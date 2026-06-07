# ABOUTME: Tests for the unified SFT+DPO pipeline's CLI parsing and checkpoint loading.
# ABOUTME: Covers parse_early_tuple, load_checkpoints, and argument defaults.

import argparse
import json
import sys
import pytest
from pathlib import Path
from types import SimpleNamespace

from tuning.training.pipeline.cli import parse_early_tuple, _parse_args
from tuning.training.pipeline.checkpoint_metadata import (
    load_checkpoints, next_checkpoint, claim_next_checkpoint, mark_completed,
    print_metadata_paths, parse_metadata_from_output, record_wandb_run_id,
)
from tuning.training.pipeline.orchestrator import (
    _build_base_cmd, _submit_sbatch_worker, _dispatch_parallel_workers,
)


# ---------------------------------------------------------------------------
# parse_early_tuple
# ---------------------------------------------------------------------------

class TestParseEarlyTuple:
    def test_valid_int_float(self):
        assert parse_early_tuple("2:0.02") == (2, 0.02)

    def test_valid_int_int_coerced_to_float(self):
        result = parse_early_tuple("5:1")
        assert result == (5, 1.0)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_invalid_no_colon(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_early_tuple("abc")

    def test_invalid_too_many_colons(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_early_tuple("1:0.02:extra")

    def test_invalid_non_numeric_patience(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_early_tuple("x:0.02")

    def test_invalid_non_numeric_delta(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_early_tuple("1:abc")


# ---------------------------------------------------------------------------
# load_checkpoints
# ---------------------------------------------------------------------------

def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


PASSK_ROW = {
    "checkpoint_path": "/models/cp1",
    "data_points_seen": 512,
    "threshold_type": "pass_at_1",
    "threshold_value": 0.3,
}
PPL_ROW = {
    "checkpoint_path": "/models/cp2",
    "data_points_seen": 256,
    "threshold_type": "perplexity",
    "threshold_value": 3.0,
}
PASSK_ROW_2 = {
    "checkpoint_path": "/models/cp3",
    "data_points_seen": 768,
    "threshold_type": "pass_at_4",
    "threshold_value": 0.5,
}


class TestLoadCheckpoints:
    def test_union_returns_all(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        result = load_checkpoints([str(f)], "union")
        assert len(result) == 2

    def test_passk_filter(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW, PASSK_ROW_2])
        result = load_checkpoints([str(f)], "passk")
        assert all(r["threshold_type"].startswith("pass_at_") for r in result)
        assert len(result) == 2

    def test_ppl_filter(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        result = load_checkpoints([str(f)], "ppl")
        assert all(r["threshold_type"] == "perplexity" for r in result)
        assert len(result) == 1

    def test_deduplicates_by_checkpoint_path(self, tmp_path):
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        _write_jsonl(f1, [PASSK_ROW])
        _write_jsonl(f2, [PASSK_ROW])  # same checkpoint_path
        result = load_checkpoints([str(f1), str(f2)], "union")
        assert len(result) == 1

    def test_first_occurrence_wins_on_dedup(self, tmp_path):
        row_a = {**PASSK_ROW, "data_points_seen": 100}
        row_b = {**PASSK_ROW, "data_points_seen": 999}  # same path, different data
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [row_a, row_b])
        result = load_checkpoints([str(f)], "union")
        assert result[0]["data_points_seen"] == 100

    def test_empty_result_exits(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PPL_ROW])
        with pytest.raises(SystemExit):
            load_checkpoints([str(f)], "passk")

    def test_merges_multiple_files(self, tmp_path):
        f1 = tmp_path / "a.jsonl"
        f2 = tmp_path / "b.jsonl"
        _write_jsonl(f1, [PASSK_ROW])
        _write_jsonl(f2, [PPL_ROW])
        result = load_checkpoints([str(f1), str(f2)], "union")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _parse_args defaults
# ---------------------------------------------------------------------------

REQUIRED = ["--model", "llama3-3B", "--wandb-project", "tuning"]


class TestParseArgs:
    def test_required_args(self):
        args = _parse_args(REQUIRED)
        assert args.model == "llama3-3B"
        assert args.wandb_project == "tuning"

    def test_default_dataset(self):
        assert _parse_args(REQUIRED).dataset == "gsm8k"

    def test_default_train_size(self):
        assert _parse_args(REQUIRED).train_size == 10000

    def test_default_sft_data_size_is_none(self):
        assert _parse_args(REQUIRED).sft_data_size is None

    def test_default_dpo_data_size_is_none(self):
        assert _parse_args(REQUIRED).dpo_data_size is None

    def test_custom_sft_data_size(self):
        args = _parse_args(REQUIRED + ["--sft-data-size", "3000"])
        assert args.sft_data_size == 3000

    def test_custom_dpo_data_size(self):
        args = _parse_args(REQUIRED + ["--dpo-data-size", "7000"])
        assert args.dpo_data_size == 7000

    def test_default_task_name(self):
        assert _parse_args(REQUIRED).task_name == "gsm8k"

    def test_default_max_seq_length(self):
        assert _parse_args(REQUIRED).max_seq_length == 1024

    def test_default_metadata_merge(self):
        assert _parse_args(REQUIRED).metadata_merge == "union"

    def test_default_sft_passk_enabled(self):
        assert _parse_args(REQUIRED).sft_enable_passk is True

    def test_default_sft_ppl_disabled(self):
        assert _parse_args(REQUIRED).sft_enable_ppl is False

    def test_default_dpo_passk_enabled(self):
        assert _parse_args(REQUIRED).dpo_enable_passk is True

    def test_default_dpo_ppl_disabled(self):
        assert _parse_args(REQUIRED).dpo_enable_ppl is False

    def test_default_sft_passk_targets(self):
        assert _parse_args(REQUIRED).sft_passk_targets == [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

    def test_default_sft_passk_k_values(self):
        assert _parse_args(REQUIRED).sft_passk_k_values == [1]

    def test_default_dpo_passk_early_is_empty(self):
        assert _parse_args(REQUIRED).dpo_passk_early == []

    def test_default_dpo_ppl_early_is_empty(self):
        assert _parse_args(REQUIRED).dpo_ppl_early == []

    def test_sft_passk_early_parsed_correctly(self):
        args = _parse_args(REQUIRED + ["--sft-passk-early", "1:0.02", "2:0.05"])
        assert args.sft_passk_early == [(1, 0.02), (2, 0.05)]

    def test_no_sft_passk_disables(self):
        args = _parse_args(REQUIRED + ["--no-sft-enable-passk"])
        assert args.sft_enable_passk is False

    def test_sft_ppl_can_be_enabled(self):
        args = _parse_args(REQUIRED + ["--sft-enable-ppl"])
        assert args.sft_enable_ppl is True

    def test_metadata_file_is_repeatable(self):
        args = _parse_args(REQUIRED + ["--metadata-file", "a.jsonl", "--metadata-file", "b.jsonl"])
        assert args.metadata_file == ["a.jsonl", "b.jsonl"]

    def test_run_sft_flag(self):
        args = _parse_args(REQUIRED + ["--run-sft"])
        assert args.run_sft is True
        assert args.run_dpo is False

    def test_default_gradient_checkpointing_enabled(self):
        assert _parse_args(REQUIRED).gradient_checkpointing is True

    def test_no_gradient_checkpointing_disables(self):
        args = _parse_args(REQUIRED + ["--no-gradient-checkpointing"])
        assert args.gradient_checkpointing is False

    def test_build_lora_config_default_keeps_unsloth_gc(self):
        from tuning.training.pipeline.stages import _build_lora_config
        cfg = _build_lora_config(_parse_args(REQUIRED))
        assert cfg.use_gradient_checkpointing == "unsloth"

    def test_build_lora_config_disables_gc(self):
        from tuning.training.pipeline.stages import _build_lora_config
        cfg = _build_lora_config(_parse_args(REQUIRED + ["--no-gradient-checkpointing"]))
        assert cfg.use_gradient_checkpointing is False

    def test_invalid_model_rejected(self):
        with pytest.raises(SystemExit):
            _parse_args(["--model", "nonexistent", "--wandb-project", "tuning"])


class TestParallelArg:
    def test_default_parallel_is_1(self):
        args = _parse_args(REQUIRED)
        assert args.parallel == 1

    def test_parallel_accepts_integer(self):
        args = _parse_args(REQUIRED + ["--parallel", "3"])
        assert args.parallel == 3


# ---------------------------------------------------------------------------
# next_checkpoint / mark_completed
# ---------------------------------------------------------------------------

class TestMetadataWorkQueue:
    def test_next_checkpoint_returns_first_row(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        result = next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp1"

    def test_next_checkpoint_skips_completed(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        completed_row = {**PASSK_ROW, "completed": True}
        _write_jsonl(f, [completed_row, PPL_ROW])
        result = next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp2"

    def test_next_checkpoint_returns_none_when_all_completed(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "completed": True}])
        assert next_checkpoint(str(f)) is None

    def test_mark_completed_sets_flag(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        mark_completed(str(f), "/models/cp1")
        with open(f) as fh:
            lines = [json.loads(l) for l in fh]
        assert lines[0]["completed"] is True
        assert "completed" not in lines[1]

    def test_mark_completed_preserves_other_fields(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW])
        mark_completed(str(f), "/models/cp1")
        with open(f) as fh:
            row = json.loads(fh.readline())
        assert row["data_points_seen"] == 512
        assert row["threshold_type"] == "pass_at_1"
        assert row["completed"] is True

    def test_record_wandb_run_id_sets_field_on_matching_row(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        record_wandb_run_id(str(f), "/models/cp1", "newrun")
        with open(f) as fh:
            lines = [json.loads(l) for l in fh]
        assert lines[0]["wandb_run_id"] == "newrun"
        assert "wandb_run_id" not in lines[1]

    def test_record_wandb_run_id_overwrites_existing(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "wandb_run_id": "old"}])
        record_wandb_run_id(str(f), "/models/cp1", "new")
        with open(f) as fh:
            row = json.loads(fh.readline())
        assert row["wandb_run_id"] == "new"

    def test_record_wandb_run_id_is_noop_when_empty(self, tmp_path):
        """An empty wandb_run_id (no active wandb run) must not touch the file."""
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW])
        record_wandb_run_id(str(f), "/models/cp1", "")
        with open(f) as fh:
            row = json.loads(fh.readline())
        assert "wandb_run_id" not in row


class TestClaimNextCheckpoint:
    def test_claims_first_available_row(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        result = claim_next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp1"

    def test_marks_row_as_claimed(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        claim_next_checkpoint(str(f))
        with open(f) as fh:
            lines = [json.loads(l) for l in fh]
        assert lines[0]["claimed"] is True
        assert lines[1].get("claimed", False) is False

    def test_skips_already_claimed_rows(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        claimed_row = {**PASSK_ROW, "claimed": True}
        _write_jsonl(f, [claimed_row, PPL_ROW])
        result = claim_next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp2"

    def test_skips_completed_rows(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        completed_row = {**PASSK_ROW, "completed": True}
        _write_jsonl(f, [completed_row, PPL_ROW])
        result = claim_next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp2"

    def test_returns_none_when_nothing_available(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [
            {**PASSK_ROW, "completed": True},
            {**PPL_ROW, "claimed": True},
        ])
        assert claim_next_checkpoint(str(f)) is None

    def test_preserves_other_fields(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW])
        claim_next_checkpoint(str(f))
        with open(f) as fh:
            row = json.loads(fh.readline())
        assert row["data_points_seen"] == 512
        assert row["threshold_type"] == "pass_at_1"
        assert row["claimed"] is True

    def test_sequential_claims_pick_different_rows(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW, PASSK_ROW_2])
        first = claim_next_checkpoint(str(f))
        second = claim_next_checkpoint(str(f))
        third = claim_next_checkpoint(str(f))
        fourth = claim_next_checkpoint(str(f))
        assert first["checkpoint_path"] == "/models/cp1"
        assert second["checkpoint_path"] == "/models/cp2"
        assert third["checkpoint_path"] == "/models/cp3"
        assert fourth is None

    def test_continue_flag_does_not_override_claim_lock(self, tmp_path):
        """A claimed row stays unclaimable even when continue=true: continue is a
        worker-side resume hint, not a claim-bypass."""
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "claimed": True, "continue": True}, PPL_ROW])
        result = claim_next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp2"


class TestWorkerExitCode:
    def test_run_dpo_exits_42_when_nothing_to_claim(self, tmp_path):
        from tuning.training.pipeline import stages
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "completed": True}])
        args = argparse.Namespace(metadata_file=[str(f)])
        with pytest.raises(SystemExit) as exc:
            stages.run_dpo(args)
        assert exc.value.code == 42

    def test_run_grpo_exits_42_when_nothing_to_claim(self, tmp_path):
        from tuning.training.pipeline import stages
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "claimed": True}])
        args = argparse.Namespace(metadata_file=[str(f)])
        with pytest.raises(SystemExit) as exc:
            stages.run_grpo(args)
        assert exc.value.code == 42


class TestSubmitSbatchWorker:
    def test_parses_job_id_from_stdout(self, monkeypatch):
        from tuning.training.pipeline import orchestrator as orch
        fake = type("R", (), {"stdout": "Submitted batch job 12345\n", "stderr": "",
                              "returncode": 0})()
        calls = []

        def fake_run(cmd, capture_output, text, env=None):
            calls.append(cmd)
            return fake

        monkeypatch.setattr(orch.subprocess, "run", fake_run)
        job_id = orch._submit_sbatch_worker(
            "tuning/slurm/unified_early_pipeline.sh",
            ["--model", "llama3-3B"],
        )
        assert job_id == "12345"
        assert calls[0][0] == "sbatch"
        assert calls[0][1] == "tuning/slurm/unified_early_pipeline.sh"
        assert "--model" in calls[0]
        assert "llama3-3B" in calls[0]

    def test_nonzero_return_code_exits(self, monkeypatch):
        from tuning.training.pipeline import orchestrator as orch
        fake = type("R", (), {"stdout": "", "stderr": "sbatch: error: boom\n",
                              "returncode": 1})()
        monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: fake)
        with pytest.raises(SystemExit):
            orch._submit_sbatch_worker("script.sh", [])

    def test_unparseable_stdout_exits(self, monkeypatch):
        from tuning.training.pipeline import orchestrator as orch
        fake = type("R", (), {"stdout": "weird output\n", "stderr": "", "returncode": 0})()
        monkeypatch.setattr(orch.subprocess, "run", lambda *a, **k: fake)
        with pytest.raises(SystemExit):
            orch._submit_sbatch_worker("script.sh", [])


class TestDispatchParallelWorkers:
    def test_parallel_1_submits_1_worker(self, monkeypatch):
        from tuning.training.pipeline import orchestrator as orch
        calls = []
        monkeypatch.setattr(orch, "_submit_sbatch_worker",
                            lambda *a, **k: (calls.append(a), "999")[1])
        args = SimpleNamespace(post_training_method="dpo", grpo_num_gpus=1)
        orch._dispatch_parallel_workers(
            parallel=1,
            base_cmd=["pipeline.py", "--model", "llama3-3B"],
            pt_flag="--run-dpo",
            metadata_files=["/tmp/a.jsonl"],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )
        assert len(calls) == 1

    def test_parallel_3_submits_3_workers(self, monkeypatch, tmp_path):
        from tuning.training.pipeline import orchestrator as orch
        calls = []
        monkeypatch.setattr(orch, "_submit_sbatch_worker",
                            lambda *a, **k: (calls.append(a), "999")[1])
        mf = tmp_path / "meta.jsonl"
        mf.write_text("{}\n")
        args = SimpleNamespace(post_training_method="dpo", grpo_num_gpus=1)
        orch._dispatch_parallel_workers(
            parallel=3,
            base_cmd=["pipeline.py", "--model", "llama3-3B"],
            pt_flag="--run-dpo",
            metadata_files=[str(mf)],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )
        assert len(calls) == 3

    def test_worker_argv_includes_pt_flag_and_metadata(self, monkeypatch, tmp_path):
        from tuning.training.pipeline import orchestrator as orch
        calls = []
        monkeypatch.setattr(orch, "_submit_sbatch_worker",
                            lambda script, argv, **kwargs: (calls.append(argv), "999")[1])
        mf = tmp_path / "meta.jsonl"
        mf.write_text("{}\n")
        args = SimpleNamespace(post_training_method="grpo", grpo_num_gpus=2)
        orch._dispatch_parallel_workers(
            parallel=2,
            base_cmd=["pipeline.py", "--model", "llama3-3B"],
            pt_flag="--run-grpo",
            metadata_files=[str(mf)],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )
        worker_argv = calls[0]
        assert "--run-grpo" in worker_argv
        assert "--run-all" in worker_argv
        assert "--metadata-file" in worker_argv
        assert str(mf) in worker_argv
        assert "--model" in worker_argv
        assert "llama3-3B" in worker_argv

    def test_strips_parallel_from_worker_argv(self, monkeypatch, tmp_path):
        from tuning.training.pipeline import orchestrator as orch
        calls = []
        monkeypatch.setattr(orch, "_submit_sbatch_worker",
                            lambda script, argv, **kwargs: (calls.append(argv), "999")[1])
        mf = tmp_path / "meta.jsonl"
        mf.write_text("{}\n")
        args = SimpleNamespace(post_training_method="dpo", grpo_num_gpus=1)
        orch._dispatch_parallel_workers(
            parallel=2,
            base_cmd=orch._build_base_cmd(["pipeline.py", "--model", "llama3-3B", "--parallel", "3"]),
            pt_flag="--run-dpo",
            metadata_files=[str(mf)],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )
        worker_argv = calls[0]
        assert "--parallel" not in worker_argv
        assert "3" not in worker_argv
        assert "--model" in worker_argv

    def test_skips_missing_metadata_files(self, monkeypatch, tmp_path):
        from tuning.training.pipeline import orchestrator as orch
        calls = []
        monkeypatch.setattr(orch, "_submit_sbatch_worker",
                            lambda script, argv, **kwargs: (calls.append(argv), "999")[1])
        real_mf = tmp_path / "real.jsonl"
        real_mf.write_text("{}\n")
        missing_mf = str(tmp_path / "missing.jsonl")
        args = SimpleNamespace(post_training_method="dpo", grpo_num_gpus=1)
        orch._dispatch_parallel_workers(
            parallel=2,
            base_cmd=["pipeline.py"],
            pt_flag="--run-dpo",
            metadata_files=[str(real_mf), missing_mf],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )
        worker_argv = calls[0]
        assert str(real_mf) in worker_argv
        assert missing_mf not in worker_argv


# ---------------------------------------------------------------------------
# print_metadata_paths / parse_metadata_from_output
# ---------------------------------------------------------------------------

class TestMetadataIPC:
    def test_print_metadata_paths(self, capsys, tmp_path):
        paths = [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
        print_metadata_paths(paths)
        captured = capsys.readouterr()
        lines = [l for l in captured.out.splitlines() if l.startswith("METADATA_FILE:")]
        assert len(lines) == 2
        assert lines[0] == f"METADATA_FILE:{paths[0]}"
        assert lines[1] == f"METADATA_FILE:{paths[1]}"

    def test_parse_metadata_from_output(self, tmp_path):
        output = f"Some log\nMETADATA_FILE:{tmp_path}/a.jsonl\nMore logs\nMETADATA_FILE:{tmp_path}/b.jsonl\n"
        result = parse_metadata_from_output(output)
        assert result == [f"{tmp_path}/a.jsonl", f"{tmp_path}/b.jsonl"]

    def test_parse_metadata_empty_output(self):
        assert parse_metadata_from_output("just logs\nno metadata\n") == []

    def test_print_empty_list(self, capsys):
        print_metadata_paths([])
        captured = capsys.readouterr()
        assert "METADATA_FILE:" not in captured.out


# ---------------------------------------------------------------------------
# _build_base_cmd
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# OffsetAwareWandbCallback
# ---------------------------------------------------------------------------

class TestOffsetAwareWandbCallback:
    def test_on_log_injects_total_global_step(self):
        from tuning.training.callback_utils import OffsetAwareWandbCallback
        from transformers import TrainerState

        callback = OffsetAwareWandbCallback(initial_global_step=100)
        callback._wandb = None  # makes super().on_log a no-op
        state = TrainerState()
        state.global_step = 5
        logs = {}

        callback.on_log(args=None, state=state, control=None, logs=logs)
        assert logs["total_global_step"] == 105

    def test_does_not_mutate_global_step(self):
        from tuning.training.callback_utils import OffsetAwareWandbCallback
        from transformers import TrainerState

        callback = OffsetAwareWandbCallback(initial_global_step=100)
        callback._wandb = None
        state = TrainerState()
        state.global_step = 5

        callback.on_log(args=None, state=state, control=None, logs={})
        assert state.global_step == 5

    def test_zero_offset_still_injects(self):
        from tuning.training.callback_utils import OffsetAwareWandbCallback
        from transformers import TrainerState

        callback = OffsetAwareWandbCallback(initial_global_step=0)
        callback._wandb = None
        state = TrainerState()
        state.global_step = 5
        logs = {}

        callback.on_log(args=None, state=state, control=None, logs=logs)
        assert logs["total_global_step"] == 5

    def test_none_offset_still_injects(self):
        from tuning.training.callback_utils import OffsetAwareWandbCallback
        from transformers import TrainerState

        callback = OffsetAwareWandbCallback(initial_global_step=None)
        callback._wandb = None
        state = TrainerState()
        state.global_step = 5
        logs = {}

        callback.on_log(args=None, state=state, control=None, logs=logs)
        assert logs["total_global_step"] == 5

    def test_setup_redefines_step_metric_after_super(self):
        """The override's define_metric calls must run AFTER the parent's,
        so that wandb's last-write-wins resolution lands on train/total_global_step."""
        from tuning.training.callback_utils import OffsetAwareWandbCallback
        from transformers.integrations import WandbCallback
        from unittest.mock import MagicMock, patch

        callback = OffsetAwareWandbCallback(initial_global_step=100)
        mock_wandb = MagicMock()
        callback._wandb = mock_wandb

        # Stub super().setup() to record the parent's define_metric calls
        # without needing to mock all of HF's wandb init internals.
        def fake_super_setup(self, args, state, model, **kwargs):
            self._wandb.define_metric("train/global_step")
            self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

        with patch.object(WandbCallback, "setup", fake_super_setup):
            callback.setup(args=None, state=None, model=None)

        glob_calls = [
            c for c in mock_wandb.define_metric.call_args_list
            if c.args == ("*",)
        ]
        assert len(glob_calls) >= 2, "Expected both super and override to define '*'"
        last_glob_call = glob_calls[-1]
        assert last_glob_call.kwargs.get("step_metric") == "train/total_global_step", (
            f"Expected last define_metric('*', ...) to use train/total_global_step, "
            f"got {last_glob_call}"
        )

    def test_setup_with_zero_offset_does_not_redefine(self):
        from tuning.training.callback_utils import OffsetAwareWandbCallback
        from transformers.integrations import WandbCallback
        from unittest.mock import MagicMock, patch

        callback = OffsetAwareWandbCallback(initial_global_step=0)
        mock_wandb = MagicMock()
        callback._wandb = mock_wandb

        def fake_super_setup(self, args, state, model, **kwargs):
            self._wandb.define_metric("*", step_metric="train/global_step", step_sync=True)

        with patch.object(WandbCallback, "setup", fake_super_setup):
            callback.setup(args=None, state=None, model=None)

        glob_calls = [
            c for c in mock_wandb.define_metric.call_args_list
            if c.args == ("*",)
        ]
        assert len(glob_calls) == 1
        assert glob_calls[0].kwargs.get("step_metric") == "train/global_step"


# ---------------------------------------------------------------------------
# _build_base_cmd
# ---------------------------------------------------------------------------

class TestBuildBaseCmd:
    def test_strips_run_all(self):
        original = ["/usr/bin/python", "pipeline.py", "--model", "llama3-3B", "--run-all", "--wandb-project", "tuning"]
        result = _build_base_cmd(original)
        assert "--run-all" not in result
        assert "--model" in result

    def test_preserves_other_args(self):
        original = ["/usr/bin/python", "pipeline.py", "--model", "llama3-3B", "--run-all", "--train-size", "5000"]
        result = _build_base_cmd(original)
        assert "--train-size" in result
        assert "5000" in result

    def test_no_run_all_unchanged(self):
        original = ["/usr/bin/python", "pipeline.py", "--model", "llama3-3B"]
        assert _build_base_cmd(original) == original


# ---------------------------------------------------------------------------
# Task-name dispatch
# ---------------------------------------------------------------------------

class TestTaskNameDispatch:
    def test_gsm8k_task_name_accepted(self):
        args = _parse_args(REQUIRED + ["--task-name", "gsm8k"])
        assert args.task_name == "gsm8k"

    def test_default_task_name_is_gsm8k(self):
        args = _parse_args(REQUIRED)
        assert args.task_name == "gsm8k"

    def test_math500_task_name_accepted(self):
        args = _parse_args(REQUIRED + ["--task-name", "math500"])
        assert args.task_name == "math500"

    def test_monitor_evals_default_empty(self):
        args = _parse_args(REQUIRED)
        assert args.monitor_evals == []

    def test_monitor_evals_single(self):
        args = _parse_args(REQUIRED + ["--monitor-evals", "math500"])
        assert args.monitor_evals == ["math500"]

    def test_monitor_evals_multiple(self):
        args = _parse_args(REQUIRED + ["--monitor-evals", "math500", "ifeval"])
        assert args.monitor_evals == ["math500", "ifeval"]

    def test_default_sft_warmup_ratio(self):
        args = _parse_args(REQUIRED)
        assert args.sft_warmup_ratio == 0.0

    def test_custom_sft_warmup_ratio(self):
        args = _parse_args(REQUIRED + ["--sft-warmup-ratio", "0.05"])
        assert args.sft_warmup_ratio == 0.05

    def test_default_sft_lr_scheduler_type(self):
        args = _parse_args(REQUIRED)
        assert args.sft_lr_scheduler_type == "cosine"

    def test_custom_sft_lr_scheduler_type(self):
        args = _parse_args(REQUIRED + ["--sft-lr-scheduler-type", "linear"])
        assert args.sft_lr_scheduler_type == "linear"


# ---------------------------------------------------------------------------
# GRPO pipeline args
# ---------------------------------------------------------------------------

class TestGRPOArgs:
    def test_default_post_training_method(self):
        args = _parse_args(REQUIRED)
        assert args.post_training_method == "dpo"

    def test_grpo_post_training_method(self):
        args = _parse_args(REQUIRED + ["--post-training-method", "grpo"])
        assert args.post_training_method == "grpo"

    def test_run_grpo_flag(self):
        args = _parse_args(REQUIRED + ["--run-grpo"])
        assert args.run_grpo is True
        assert args.run_dpo is False

    def test_grpo_defaults(self):
        args = _parse_args(REQUIRED)
        assert args.grpo_num_epochs == 1
        assert args.grpo_eval_steps == 64
        assert args.grpo_batch_size == 4
        assert args.grpo_grad_accum == 32
        assert args.grpo_num_generations == 8
        assert args.grpo_max_completion_length == 2048
        assert args.grpo_beta == 0.0
        assert args.grpo_temperature == 1.0
        assert args.grpo_loss_type == "dapo"
        assert args.grpo_data_size is None
        assert args.grpo_warmup_ratio == 0.0
        assert args.grpo_lr_scheduler_type == "cosine"

    def test_custom_grpo_warmup_ratio(self):
        args = _parse_args(REQUIRED + ["--grpo-warmup-ratio", "0.1"])
        assert args.grpo_warmup_ratio == 0.1

    def test_custom_grpo_lr_scheduler_type(self):
        args = _parse_args(REQUIRED + ["--grpo-lr-scheduler-type", "linear"])
        assert args.grpo_lr_scheduler_type == "linear"

    def test_grpo_custom_args(self):
        args = _parse_args(REQUIRED + [
            "--grpo-num-generations", "16",
            "--grpo-beta", "0.1",
            "--grpo-loss-type", "dr_grpo",
            "--grpo-data-size", "5000",
        ])
        assert args.grpo_num_generations == 16
        assert args.grpo_beta == 0.1
        assert args.grpo_loss_type == "dr_grpo"
        assert args.grpo_data_size == 5000

    def test_grpo_enable_passk_default(self):
        args = _parse_args(REQUIRED)
        assert args.grpo_enable_passk is True

    def test_grpo_passk_args(self):
        args = _parse_args(REQUIRED + [
            "--grpo-passk-targets", "0.5", "0.8",
            "--grpo-passk-early", "1:0.02",
        ])
        assert args.grpo_passk_targets == [0.5, 0.8]
        assert args.grpo_passk_early == [(1, 0.02)]

    def test_grpo_data_budget_with_fixed_split(self):
        """When --sft-data-size and --grpo-data-size are both set, GRPO uses fixed dataset."""
        args = _parse_args(REQUIRED + [
            "--sft-data-size", "3000",
            "--grpo-data-size", "7000",
            "--post-training-method", "grpo",
        ])
        assert args.sft_data_size == 3000
        assert args.grpo_data_size == 7000


# ---------------------------------------------------------------------------
# --sft-passk-max-checkpoint-gap CLI arg
# ---------------------------------------------------------------------------

class TestMaxCheckpointGapArg:
    def test_default_is_none(self):
        args = _parse_args(REQUIRED)
        assert args.sft_passk_max_checkpoint_gap is None

    def test_custom_value(self):
        args = _parse_args(REQUIRED + ["--sft-passk-max-checkpoint-gap", "1024"])
        assert args.sft_passk_max_checkpoint_gap == 1024


# ---------------------------------------------------------------------------
# --sft-passk-target-data-points CLI arg
# ---------------------------------------------------------------------------

class TestTargetDataPointsArg:
    def test_default_is_none(self):
        args = _parse_args(REQUIRED)
        assert args.sft_passk_target_data_points is None

    def test_single_value(self):
        args = _parse_args(REQUIRED + ["--sft-passk-target-data-points", "4000"])
        assert args.sft_passk_target_data_points == [4000]

    def test_multiple_values(self):
        args = _parse_args(REQUIRED + [
            "--sft-passk-target-data-points", "4000", "8000", "12000",
        ])
        assert args.sft_passk_target_data_points == [4000, 8000, 12000]


# ---------------------------------------------------------------------------
# --simple-template CLI arg
# ---------------------------------------------------------------------------

class TestSimpleTemplateArg:
    def test_default_simple_template_is_false(self):
        args = _parse_args(REQUIRED)
        assert args.simple_template is False

    def test_simple_template_flag(self):
        args = _parse_args(REQUIRED + ["--simple-template"])
        assert args.simple_template is True

    def test_no_simple_template_flag(self):
        args = _parse_args(REQUIRED + ["--no-simple-template"])
        assert args.simple_template is False

    def test_grpo_simple_template_removed(self):
        """Old --grpo-simple-template flag should no longer be accepted."""
        with pytest.raises(SystemExit):
            _parse_args(REQUIRED + ["--grpo-simple-template"])


import tuning.config


@pytest.fixture(autouse=True)
def restore_seed_globals():
    orig_seed = tuning.config.DEFAULT_SEED
    orig_eval = tuning.config.DEFAULT_EVAL_SEED
    yield
    tuning.config.DEFAULT_SEED = orig_seed
    tuning.config.DEFAULT_EVAL_SEED = orig_eval


class TestSeedArgs:
    def test_seed_default(self):
        args = _parse_args(["--model", "llama3-1B", "--wandb-project", "test"])
        assert args.seed == 42

    def test_seed_override(self):
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--seed", "7",
        ])
        assert args.seed == 7

    def test_eval_seed_default_none(self):
        args = _parse_args(["--model", "llama3-1B", "--wandb-project", "test"])
        assert args.eval_seed is None

    def test_eval_seed_override(self):
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--eval_seed", "99",
        ])
        assert args.eval_seed == 99


class TestEffectiveEvalSeed:
    def test_effective_eval_seed_falls_back_to_seed(self):
        from tuning.training.pipeline.cli import effective_eval_seed
        assert effective_eval_seed(seed=42, eval_seed=None) == 42

    def test_effective_eval_seed_override_wins(self):
        from tuning.training.pipeline.cli import effective_eval_seed
        assert effective_eval_seed(seed=42, eval_seed=7) == 7


class TestInitSeeds:
    def test_init_seeds_sets_globals(self):
        from tuning.training.pipeline.cli import _init_seeds
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--seed", "7", "--eval_seed", "99",
        ])
        _init_seeds(args)
        assert tuning.config.DEFAULT_SEED == 7
        assert tuning.config.DEFAULT_EVAL_SEED == 99
        assert tuning.config.get_eval_seed() == 99

    def test_init_seeds_eval_seed_falls_back(self):
        from tuning.training.pipeline.cli import _init_seeds
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--seed", "7",
        ])
        _init_seeds(args)
        assert tuning.config.DEFAULT_SEED == 7
        assert tuning.config.get_eval_seed() == 7

    def test_init_seeds_seeds_random(self):
        import random
        from tuning.training.pipeline.cli import _init_seeds
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--seed", "123",
        ])
        _init_seeds(args)
        val1 = random.random()
        # Re-seed and verify same stream
        random.seed(123)
        assert random.random() == val1


class TestSbatchScriptResolution:
    def test_default_uses_long_script(self):
        args = _parse_args(["--model", "llama3-3B", "--wandb-project", "p"])
        assert args.sbatch_script == "tuning/slurm/unified_early_pipeline.sh"

    def test_short_flag_uses_short_script(self):
        args = _parse_args(["--model", "llama3-3B", "--wandb-project", "p", "--short"])
        assert args.sbatch_script == "tuning/slurm/unified_early_pipeline_short.sh"
