# ABOUTME: Tests for the unified SFT+DPO pipeline's CLI parsing and checkpoint loading.
# ABOUTME: Covers parse_early_tuple, load_checkpoints, and argument defaults.

import argparse
import json
import sys
import pytest
from pathlib import Path

from tuning.training.unified_early_pipeline import (
    parse_early_tuple,
    load_checkpoints,
    _parse_args,
    next_checkpoint,
    mark_completed,
    print_metadata_paths,
    parse_metadata_from_output,
    _build_base_cmd,
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
        assert _parse_args(REQUIRED).sft_passk_targets == [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

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

    def test_invalid_model_rejected(self):
        with pytest.raises(SystemExit):
            _parse_args(["--model", "nonexistent", "--wandb-project", "tuning"])


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
# GlobalStepOffsetCallback
# ---------------------------------------------------------------------------

class TestGlobalStepOffsetCallback:
    def test_does_not_mutate_global_step(self):
        from tuning.training.callback_utils import GlobalStepOffsetCallback
        from transformers import TrainerState

        callback = GlobalStepOffsetCallback(initial_global_step=100)
        state = TrainerState()
        state.global_step = 5

        callback.on_log(args=None, state=state, control=None, logs={})
        assert state.global_step == 5, "global_step must not be mutated"

    def test_adds_total_global_step_to_logs(self):
        from tuning.training.callback_utils import GlobalStepOffsetCallback
        from transformers import TrainerState

        callback = GlobalStepOffsetCallback(initial_global_step=100)
        state = TrainerState()
        state.global_step = 5
        logs = {}

        callback.on_log(args=None, state=state, control=None, logs=logs)
        assert logs["train/total_global_step"] == 105

    def test_zero_offset_skips_injection(self):
        from tuning.training.callback_utils import GlobalStepOffsetCallback
        from transformers import TrainerState

        callback = GlobalStepOffsetCallback(initial_global_step=0)
        state = TrainerState()
        state.global_step = 5
        logs = {}

        callback.on_log(args=None, state=state, control=None, logs=logs)
        assert "train/total_global_step" not in logs

    def test_none_offset_skips_injection(self):
        from tuning.training.callback_utils import GlobalStepOffsetCallback
        from transformers import TrainerState

        callback = GlobalStepOffsetCallback(initial_global_step=None)
        state = TrainerState()
        state.global_step = 5
        logs = {}

        callback.on_log(args=None, state=state, control=None, logs=logs)
        assert "train/total_global_step" not in logs

    def test_on_train_begin_calls_define_metric(self):
        from tuning.training.callback_utils import GlobalStepOffsetCallback
        from unittest.mock import patch, MagicMock

        callback = GlobalStepOffsetCallback(initial_global_step=100)
        mock_wandb = MagicMock()
        mock_wandb.run = True

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            callback.on_train_begin(args=None, state=None, control=None)

        mock_wandb.define_metric.assert_any_call("train/total_global_step")
        mock_wandb.define_metric.assert_any_call("*", step_metric="train/total_global_step")

    def test_on_train_begin_skips_when_no_offset(self):
        from tuning.training.callback_utils import GlobalStepOffsetCallback
        from unittest.mock import patch, MagicMock

        callback = GlobalStepOffsetCallback(initial_global_step=0)
        mock_wandb = MagicMock()
        mock_wandb.run = True

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            callback.on_train_begin(args=None, state=None, control=None)

        mock_wandb.define_metric.assert_not_called()


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

    def test_default_sft_warmup_ratio(self):
        args = _parse_args(REQUIRED)
        assert args.sft_warmup_ratio == 0.0

    def test_custom_sft_warmup_ratio(self):
        args = _parse_args(REQUIRED + ["--sft-warmup-ratio", "0.05"])
        assert args.sft_warmup_ratio == 0.05
