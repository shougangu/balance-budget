# ABOUTME: Tests the SFT optimizer/checkpoint knobs (grad clipping, Adam beta2, save
# ABOUTME: cadence) reach TrainingArguments from the config model and the pipeline CLI.

import sys
from types import ModuleType
from unittest.mock import MagicMock

if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())

from tuning.training.config_training import TrainingArgumentsConfig
from tuning.training.pipeline.cli import _parse_args


def test_config_defaults_match_hf_trainer():
    d = TrainingArgumentsConfig().to_hf_args(output_dir="/tmp/x")
    assert d["max_grad_norm"] == 1.0
    assert d["adam_beta2"] == 0.999


def test_config_overrides_reach_hf_args():
    config = TrainingArgumentsConfig(max_grad_norm=0.0, adam_beta2=0.98,
                                     save_steps=651, save_total_limit=6)
    d = config.to_hf_args(output_dir="/tmp/x")
    assert (d["max_grad_norm"], d["adam_beta2"], d["save_steps"], d["save_total_limit"]) == (0.0, 0.98, 651, 6)


def test_cli_defaults_leave_config_alone():
    args = _parse_args(["--model", "llama3-8B", "--wandb-project", "t"])
    assert args.sft_max_grad_norm == 1.0
    assert args.sft_adam_beta2 == 0.999
    assert args.sft_save_steps is None
    assert args.sft_save_total_limit is None


def test_cli_parses_overrides():
    args = _parse_args(["--model", "llama3-8B", "--wandb-project", "t",
                        "--sft-max-grad-norm", "0", "--sft-adam-beta2", "0.98",
                        "--sft-save-steps", "651", "--sft-save-total-limit", "6"])
    assert (args.sft_max_grad_norm, args.sft_adam_beta2, args.sft_save_steps, args.sft_save_total_limit) == (0.0, 0.98, 651, 6)
