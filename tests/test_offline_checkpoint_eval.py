# ABOUTME: Tests the offline checkpoint eval wrapper and the metadata-watching submitter.
# ABOUTME: Covers budget-position readback, W&B grouping/flattening, and exactly-once submission.

import json
import sys
from pathlib import Path
from unittest.mock import patch

from transformers import TrainerState

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import offline_checkpoint_eval as oce  # noqa: E402
import submit_offline_evals as soe  # noqa: E402


def _write_trainer_state(checkpoint_dir, total_minutes):
    state = TrainerState()
    state.stateful_callbacks = {
        "OffsetAwareWandbCallback": {
            "args": {"initial_global_step": 0},
            "attributes": {"total_seconds": total_minutes * 60.0},
        }
    }
    state.save_to_json(str(checkpoint_dir / "trainer_state.json"))


def test_checkpoint_total_minutes_reads_the_saved_clock(tmp_path):
    _write_trainer_state(tmp_path, total_minutes=7680.0)
    assert oce.checkpoint_total_minutes(str(tmp_path)) == 7680.0


def test_total_minutes_argument_wins_over_the_saved_clock(tmp_path):
    """RL marks exported by verl carry no trainer_state.json; the row's value is passed in."""
    args = oce.parse_args([
        "--checkpoint", str(tmp_path), "--model-family", "qwen3-8B",
        "--wandb-project", "longcot", "--total-minutes", "11520",
    ])
    assert oce.resolve_total_minutes(args) == 11520.0


def test_total_minutes_falls_back_to_the_saved_clock(tmp_path):
    _write_trainer_state(tmp_path, total_minutes=7680.0)
    args = oce.parse_args([
        "--checkpoint", str(tmp_path), "--model-family", "qwen3-8B",
        "--wandb-project", "longcot",
    ])
    assert oce.resolve_total_minutes(args) == 7680.0


def _write_tokenizer_config(checkpoint_dir, chat_template):
    (checkpoint_dir / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": chat_template}))


def test_template_auto_follows_the_checkpoints_saved_template(tmp_path):
    from tuning.utils.utils import SIMPLE_TEMPLATE

    simple = tmp_path / "simple"
    simple.mkdir()
    _write_tokenizer_config(simple, SIMPLE_TEMPLATE)
    family = tmp_path / "family"
    family.mkdir()
    _write_tokenizer_config(family, "{{ bos_token }}<|im_start|>{{ messages }}")

    assert oce.resolve_template(str(simple), "auto") == "simple"
    assert oce.resolve_template(str(family), "auto") == "repo"
    assert oce.resolve_template(str(family), "native") == "native"


def test_template_auto_refuses_a_checkpoint_without_a_template(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        oce.resolve_template(str(tmp_path), "auto")


def test_wandb_group_is_the_run_id_suffix():
    path = "/models/qwen3-8B_budget-7680m_sft-123456_ab12cd34"
    assert oce.wandb_group_from_checkpoint(path) == "ab12cd34"


def test_flatten_report_prefixes_benchmark_and_metric():
    report = {"benchmarks": {"aime25": {"pass_at_1": 0.4, "maj_at_32": 0.6}}}
    flat = oce.flatten_report(report)
    assert flat == {"eval/aime25/pass_at_1": 0.4, "eval/aime25/maj_at_32": 0.6}


def test_calibration_argv_round_trips_through_its_parser(tmp_path):
    import external_eval_calibration as cal

    args = oce.parse_args([
        "--checkpoint", str(tmp_path), "--model-family", "qwen3-8B",
        "--wandb-project", "longcot", "--tensor-parallel-size", "2",
        "--template", "repo",
    ])
    calib = cal.parse_args(oce.build_calibration_argv(args, template="repo"))
    assert calib.model == str(tmp_path)
    assert calib.template == "repo"
    assert calib.max_tokens == 32768
    assert calib.max_model_len == 36864
    assert calib.tensor_parallel_size == 2
    assert calib.n_samples == 1
    assert calib.amc_n_samples == 8
    assert calib.aime_n_samples == 16
    assert calib.save_generations is True
    assert calib.benchmarks == oce.DEFAULT_BENCHMARKS


def test_default_suite_leads_with_the_hard_math_sets():
    """AIME26 waits on a contamination check of the SFT corpus; OlympiadBench is
    the low-variance hard set; MATH-500 and GSM8K trail as saturating references."""
    assert oce.DEFAULT_BENCHMARKS == (
        "olympiadbench,amc,aime24,aime25,hmmt_feb25,gsm8k,math500")


def test_only_the_small_competition_sets_are_sample_averaged():
    """AIME/HMMT are 30 problems and AMC 83, so their pass@1 needs averaging over
    samples; OlympiadBench, MATH-500 and GSM8K are large enough to read at n=1."""
    args = oce.parse_args([
        "--checkpoint", "/ckpt", "--model-family", "qwen3-8B",
        "--wandb-project", "longcot",
    ])
    assert args.n_samples == 1
    assert args.amc_n_samples == 8
    assert args.aime_n_samples == 16
    assert max(args.k_values) <= args.aime_n_samples


def _metadata_file(tmp_path, rows):
    path = tmp_path / "marks.json"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _watch_args(tmp_path, metadata_file):
    return soe.parse_args([
        "--metadata-files", metadata_file,
        "--model-family", "qwen3-8B", "--wandb-project", "longcot",
    ])


def test_pending_rows_skip_submitted_and_missing_checkpoints(tmp_path):
    real = tmp_path / "ckpt_a"
    real.mkdir()
    metadata_file = _metadata_file(tmp_path, [
        {"checkpoint_path": str(real)},
        {"checkpoint_path": str(real), "eval_submitted": True},
        {"checkpoint_path": str(tmp_path / "missing")},
    ])
    rows = soe.pending_rows(metadata_file)
    assert [r.get("eval_submitted") for r in rows] == [None]


def test_submit_marks_the_row_and_calls_sbatch_once(tmp_path):
    real = tmp_path / "ckpt_a"
    real.mkdir()
    metadata_file = _metadata_file(
        tmp_path, [{"checkpoint_path": str(real), "sft_wandb_run_id": "ab12cd34",
                    "total_minutes": 7680.0}])
    args = _watch_args(tmp_path, metadata_file)

    with patch.object(soe.subprocess, "run") as run:
        run.return_value.stdout = "Submitted batch job 1\n"
        submitted = soe.scan_once(args)

    assert submitted == 1
    command = run.call_args.args[0]
    assert command[0] == "sbatch"
    assert "--wandb-group" in command and "ab12cd34" in command
    assert command[command.index("--total-minutes") + 1] == "7680.0"
    rows = [json.loads(l) for l in open(metadata_file)]
    assert rows[0]["eval_submitted"] is True

    with patch.object(soe.subprocess, "run") as run2:
        assert soe.scan_once(args) == 0
    run2.assert_not_called()


def test_default_allocation_is_one_h100_for_twelve_hours(tmp_path):
    """One GPU matches the eval's tensor_parallel_size of 1, and 12h fits the
    b2 partition tier."""
    real = tmp_path / "ckpt_a"
    real.mkdir()
    metadata_file = _metadata_file(tmp_path, [{"checkpoint_path": str(real)}])
    args = _watch_args(tmp_path, metadata_file)

    with patch.object(soe.subprocess, "run") as run:
        run.return_value.stdout = "Submitted batch job 1\n"
        soe.scan_once(args)

    command = run.call_args.args[0]
    assert "--gres=gpu:h100:1" in command
    assert "--time=12:00:00" in command


def test_sbatch_script_defaults_match_the_watcher():
    """A hand-submitted eval must get the same allocation as a watched one."""
    script = open(soe.OFFLINE_EVAL_SBATCH).read()
    assert "#SBATCH --gres=gpu:h100:1" in script
    assert "#SBATCH --time=12:00:00" in script


def test_watcher_partition_flag_reaches_sbatch(tmp_path):
    real = tmp_path / "ckpt_a"
    real.mkdir()
    metadata_file = _metadata_file(tmp_path, [{"checkpoint_path": str(real)}])
    args = soe.parse_args([
        "--metadata-files", metadata_file, "--model-family", "qwen3-8B",
        "--wandb-project", "longcot", "--partition", "gpubase_l40s_b1",
    ])
    with patch.object(soe.subprocess, "run") as run:
        run.return_value.stdout = "Submitted batch job 1\n"
        soe.scan_once(args)
    command = run.call_args.args[0]
    assert "--partition=gpubase_l40s_b1" in command
    assert command.index("--partition=gpubase_l40s_b1") < command.index(args.sbatch_script)
