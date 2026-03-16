# ABOUTME: Tests for the warmup ablation script's CLI parsing and defaults.
# ABOUTME: Covers argument parsing, model choices, and warmup-ratio handling.

import pytest
from tuning.training.warmup_ablation import _parse_args


REQUIRED = ["--model", "llama3-3B", "--warmup-ratio", "0.05",
            "--wandb-project", "ablations"]


class TestParseArgs:
    def test_required_args(self):
        args = _parse_args(REQUIRED)
        assert args.model == "llama3-3B"
        assert args.warmup_ratio == 0.05
        assert args.wandb_project == "ablations"

    def test_warmup_ratio_zero(self):
        args = _parse_args(["--model", "qwen2-3B", "--warmup-ratio", "0.0",
                            "--wandb-project", "ablations"])
        assert args.warmup_ratio == 0.0

    def test_warmup_ratio_baseline(self):
        args = _parse_args(["--model", "llama3-3B", "--warmup-ratio", "0.1",
                            "--wandb-project", "ablations"])
        assert args.warmup_ratio == 0.1

    def test_default_dataset(self):
        assert _parse_args(REQUIRED).dataset == "gsm8k"

    def test_default_train_size(self):
        assert _parse_args(REQUIRED).train_size == 10000

    def test_default_task_name(self):
        assert _parse_args(REQUIRED).task_name == "gsm8k"

    def test_default_max_seq_length(self):
        assert _parse_args(REQUIRED).max_seq_length == 1024

    def test_default_num_epochs(self):
        assert _parse_args(REQUIRED).num_epochs == 2

    def test_default_eval_steps(self):
        assert _parse_args(REQUIRED).eval_steps == 64

    def test_default_batch_size(self):
        assert _parse_args(REQUIRED).batch_size == 16

    def test_default_grad_accum(self):
        assert _parse_args(REQUIRED).grad_accum == 1

    def test_default_passk_targets(self):
        targets = _parse_args(REQUIRED).passk_targets
        assert targets == [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                           0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
                           0.85, 0.9, 0.95]

    def test_default_passk_temperature(self):
        assert _parse_args(REQUIRED).passk_temperature == 0.5

    def test_default_passk_num_prompts(self):
        assert _parse_args(REQUIRED).passk_num_prompts == 541

    def test_custom_dataset_tuluif(self):
        args = _parse_args(REQUIRED + ["--dataset", "tuluif"])
        assert args.dataset == "tuluif"

    def test_custom_task_name_ifeval(self):
        args = _parse_args(REQUIRED + ["--task-name", "ifeval"])
        assert args.task_name == "ifeval"

    def test_custom_num_epochs(self):
        args = _parse_args(REQUIRED + ["--num-epochs", "3"])
        assert args.num_epochs == 3

    def test_custom_passk_targets(self):
        args = _parse_args(REQUIRED + ["--passk-targets", "0.5", "0.8"])
        assert args.passk_targets == [0.5, 0.8]

    def test_invalid_model_rejected(self):
        with pytest.raises(SystemExit):
            _parse_args(["--model", "nonexistent", "--warmup-ratio", "0.1",
                         "--wandb-project", "ablations"])

    def test_qwen_model_accepted(self):
        args = _parse_args(["--model", "qwen2-3B", "--warmup-ratio", "0.0",
                            "--wandb-project", "ablations"])
        assert args.model == "qwen2-3B"
