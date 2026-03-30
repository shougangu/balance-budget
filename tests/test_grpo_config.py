# ABOUTME: Tests for GRPOTrainingConfig and PTRunConfig GRPO naming.
# ABOUTME: Validates config defaults, to_hf_args output, and run name generation.

import sys
from types import ModuleType


# Stub unsloth before importing config (crashes without GPU)
if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub

from tuning.training.config_training import GRPOTrainingConfig, PTRunConfig, SFTRunConfig, DatasetConfig


def test_grpo_config_defaults():
    config = GRPOTrainingConfig()
    assert config.num_generations == 8
    assert config.max_completion_length == 1024
    assert config.beta == 0.01
    assert config.temperature == 1.0
    assert config.epsilon == 0.2
    assert config.epsilon_high == 0.28
    assert config.loss_type == "grpo"
    assert config.scale_rewards == "group"
    assert config.use_vllm is True
    assert config.learning_rate == 1e-6
    assert config.num_train_epochs == 1
    assert config.per_device_train_batch_size == 8
    assert config.gradient_accumulation_steps == 1


def test_grpo_config_to_hf_args():
    config = GRPOTrainingConfig()
    d = config.to_hf_args(output_dir="/tmp/test")
    assert d["output_dir"] == "/tmp/test"
    assert d["num_generations"] == 8
    assert d["max_completion_length"] == 1024
    assert d["beta"] == 0.01
    assert d["temperature"] == 1.0
    assert d["epsilon"] == 0.2
    assert d["epsilon_high"] == 0.28
    assert d["loss_type"] == "grpo"
    assert d["scale_rewards"] == "group"
    assert d["save_strategy"] == "no"
    # Should not contain fields that are not GRPOConfig params
    assert "eval_accumulation_steps" not in d


def test_pt_run_config_grpo_naming():
    sft_config = SFTRunConfig(
        model_name="llama3-8B",
        model_name_hf="unsloth/Meta-Llama-3.1-8B",
        dataset_config=DatasetConfig(dataset="gsm8k", dataset_type="sft", train_size=1000),
    )
    run_config = PTRunConfig(
        model_name="llama3-8B",
        model_name_hf="unsloth/Meta-Llama-3.1-8B",
        sft_run_config=sft_config,
        dataset_config=DatasetConfig(dataset="gsm8k", dataset_type="rlvr", train_size=500),
        pft_method="grpo",
    )
    assert "grpo" in run_config.run_name
    assert "rlvr-gsm8k-500" in run_config.run_name
