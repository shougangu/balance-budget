# ABOUTME: Tests the verl-side pieces that run outside the verl venv: the union-grader
# ABOUTME: reward and the rlvr -> verl parquet conversion (prompt byte-parity included).

import json
from unittest.mock import patch

from datasets import Dataset, DatasetDict, load_dataset


PROMPT = [
    {"role": "system", "content": "Solve the problem."},
    {"role": "user", "content": "Problem: 6*7?\nAnswer:"},
]


def test_reward_accepts_boxed_answer():
    from tuning.verl.reward import compute_score
    assert compute_score("balance-budget/rlvr-dapo", "<think>\nx\n</think>\n\nSo $\\boxed{42}$.", "42") == 1.0


def test_reward_rejects_wrong_answer():
    from tuning.verl.reward import compute_score
    assert compute_score("balance-budget/rlvr-dapo", "The answer is $\\boxed{41}$.", "42") == 0.0


def test_reward_accepts_hash_marker_answer():
    """The lm-eval numeric side of the union grader: #### extraction, no boxed."""
    from tuning.verl.reward import compute_score
    assert compute_score("balance-budget/rlvr-dapo", "Working...\n#### 42", "42") == 1.0


def test_bank_marks_are_the_rows_between_mark_and_budget():
    from tuning.verl.run_verl_grpo import bank_marks
    marks = bank_marks(sft_total_minutes=7680, budget_minutes=30720,
                       bank_at=[15360, 30720])
    assert marks == [15360.0, 30720.0]


def test_bank_marks_always_include_the_budget_and_drop_passed_rows():
    from tuning.verl.run_verl_grpo import bank_marks
    assert bank_marks(15360, 61440, bank_at=[7680, 15360, 30720]) == [30720.0, 61440.0]


def test_bank_marks_empty_when_budget_already_spent():
    from tuning.verl.run_verl_grpo import bank_marks
    assert bank_marks(15360, 15360, bank_at=[15360]) == []


def test_entry_args_defaults():
    from tuning.verl.run_verl_grpo import parse_args
    args = parse_args([
        "--metadata-file", "m.json", "--claim-checkpoint", "/ckpt/a",
        "--budget-minutes", "15360", "--config", "c.yaml",
        "--wandb-project", "longcot",
    ])
    assert args.bank_at == []
    assert args.budget_minutes == 15360.0
    # verl's resumable tree (sharded fp32 weights + optimizer + hf export, ~130 GB
    # per save at 8B) lives beside the repo, never on purge-prone scratch.
    import os
    from tuning.config import ROOT_DIR
    repo_parent = os.path.dirname(os.path.dirname(ROOT_DIR))
    assert args.local_ckpt_root == os.path.join(repo_parent, "verl_ckpts")


def _metadata_file(tmp_path, row):
    path = tmp_path / "marks.json"
    path.write_text(json.dumps(row) + "\n")
    return str(path)


def _entry_argv(metadata_file, checkpoint):
    return [
        "--metadata-file", metadata_file, "--claim-checkpoint", checkpoint,
        "--budget-minutes", "15360", "--config", "c.yaml",
        "--wandb-project", "longcot",
    ]


def test_main_leaves_a_completed_row_alone(tmp_path):
    """A resubmitted worker must not retrain a cell that already finished."""
    import tuning.verl.run_verl_grpo as entry

    metadata_file = _metadata_file(tmp_path, {
        "checkpoint_path": "/ckpt/a", "total_minutes": 7680.0,
        "claimed": True, "completed": True,
    })
    with patch.object(entry, "build_config",
                      side_effect=AssertionError("must not train")):
        entry.main(_entry_argv(metadata_file, "/ckpt/a"))


def test_main_completes_a_mark_that_already_meets_the_budget(tmp_path):
    import tuning.verl.run_verl_grpo as entry

    metadata_file = _metadata_file(tmp_path, {
        "checkpoint_path": "/ckpt/a", "total_minutes": 15360.0,
    })
    with patch.object(entry, "build_config",
                      side_effect=AssertionError("must not train")):
        entry.main(_entry_argv(metadata_file, "/ckpt/a"))
    row = json.loads(open(metadata_file).readline())
    assert row["completed"] is True


def test_convert_writes_verl_schema_with_byte_identical_prompts(tmp_path):
    import tuning.verl.convert_rlvr_to_parquet as conv

    source = DatasetDict({
        "train": Dataset.from_dict({
            "prompt": [PROMPT], "reference_answer": ["42"],
        }),
        "test": Dataset.from_dict({
            "prompt": [PROMPT], "reference_answer": ["7"],
        }),
    })
    source.save_to_disk(str(tmp_path / "rlvr-fake"))

    with patch.object(conv, "DATASETS_DIR", str(tmp_path)):
        paths = conv.convert("fake", str(tmp_path / "verl"))

    train = load_dataset("parquet", data_files=paths["train"], split="train")
    row = train[0]
    assert row["data_source"] == "balance-budget/rlvr-fake"
    assert row["ability"] == "math"
    assert row["reward_model"] == {"style": "rule", "ground_truth": "42"}
    assert row["extra_info"]["split"] == "train"
    assert row["prompt"] == PROMPT

    test = load_dataset("parquet", data_files=paths["test"], split="train")
    assert test[0]["reward_model"]["ground_truth"] == "7"


def test_decision_engine_imports_without_the_eval_stack():
    """The verl venv has no eval dependencies; budget_trainer must reach the
    decision engine without the passk package importing its callback."""
    import subprocess
    import sys

    code = (
        "import sys, tuning.training.passk.decisions; "
        "assert 'tuning.training.passk.callback' not in sys.modules; "
        "assert 'tuning.training.eval_strategy' not in sys.modules; "
        "from tuning.training.passk import PassAtKStoppingCallback; "
        "assert 'tuning.training.passk.callback' in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_task_runner_survives_rays_by_value_class_transfer():
    """ray.remote ships its actor wrapper to the actor process by value; the
    reconstructed class must still construct (the upstream TaskRunner's zero-arg
    super() does not, once its class is cloned in transit)."""
    import pytest
    pytest.importorskip("verl")
    import ray.cloudpickle as cloudpickle
    from tuning.verl.budget_trainer import BudgetTaskRunner

    wrapper = BudgetTaskRunner.__ray_metadata__.modified_class
    reloaded = cloudpickle.loads(cloudpickle.dumps(wrapper))
    runner = reloaded()
    assert runner.role_worker_mapping == {}
    assert reloaded.run.__qualname__.startswith("_BudgetTaskRunner.run")


def test_build_config_installs_the_reward_function_where_verl_reads_it(tmp_path):
    import pytest
    pytest.importorskip("verl")
    from tuning.verl.run_verl_grpo import build_config, parse_args

    args = parse_args([
        "--metadata-file", "m.json", "--claim-checkpoint", "/ckpt/a",
        "--budget-minutes", "16", "--bank-at", "8",
        "--config", "tuning/verl/configs/smoke_qwen3_grpo.yaml",
        "--wandb-project", "utilities", "--local-ckpt-root", str(tmp_path),
    ])
    config = build_config(args, {"checkpoint_path": "/ckpt/a", "total_minutes": 4.0})
    assert config.reward.custom_reward_function.path == "tuning/verl/reward.py"
    assert config.reward.custom_reward_function.name == "compute_score"
    assert list(config.budget.marks) == [8.0, 16.0]
    # reward_kwargs is read with .get() by the DAPO manager and is not in verl's
    # reward schema; the merge must still carry it.
    assert config.reward.reward_manager.name == "dapo"
    assert config.reward.reward_kwargs.max_resp_len == config.data.max_response_length


def test_reward_runs_in_a_worker_thread():
    """verl's reward loop scores from worker threads, where math-verify's
    signal-based timeout raises; the reward must grade there anyway."""
    import threading

    from tuning.verl.reward import compute_score

    results = []

    def grade():
        try:
            results.append(compute_score("x", "So $\\boxed{\\frac{3}{4}}$.", "\\frac{3}{4}"))
        except Exception as exc:  # noqa: BLE001 - the failure itself is the finding
            results.append(exc)

    thread = threading.Thread(target=grade)
    thread.start()
    thread.join()
    assert results == [1.0], results


def test_campaign_configs_penalize_overlong_responses():
    """A response cut off at the cap grades like a wrong answer; DAPO's soft
    overlong buffer tells the policy the length was the problem, which is the
    guard against the length-collapse mode of the TRL long runs."""
    import yaml

    for name in ("qwen3_8b_grpo", "qwen3_14b_grpo", "llama3_8b_grpo", "smoke_qwen3_grpo"):
        with open(f"tuning/verl/configs/{name}.yaml") as fh:
            config = yaml.safe_load(fh)
        reward = config["reward"]
        assert reward["reward_manager"]["name"] == "dapo", name
        buffer = reward["reward_kwargs"]["overlong_buffer_cfg"]
        assert buffer["enable"] is True, name
        assert 0 < buffer["len"] < config["data"]["max_response_length"], name
        assert reward["reward_kwargs"]["max_resp_len"] == config["data"]["max_response_length"], name
