# ABOUTME: Tests for GRPOTrainingConfig and PTRunConfig GRPO naming.
# ABOUTME: Validates config defaults, to_hf_args output, and run name generation.

import sys
from types import ModuleType


# Stub unsloth before importing config (crashes without GPU)
if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub

import pytest
import tuning.config
from tuning.training.config_training import GRPOTrainingConfig, PTRunConfig, SFTRunConfig, DatasetConfig
from tuning.training.pipeline.cli import _parse_args


@pytest.fixture(autouse=True)
def restore_seed_globals():
    orig = tuning.config.DEFAULT_SEED
    yield
    tuning.config.DEFAULT_SEED = orig


def test_grpo_config_defaults():
    config = GRPOTrainingConfig()
    assert config.num_generations == 8
    assert config.max_completion_length == 2048
    assert config.beta == 0.0
    assert config.temperature == 1.0
    assert config.epsilon == 0.2
    assert config.epsilon_high == 0.28
    assert config.loss_type == "dapo"
    assert config.scale_rewards == "group"
    assert config.use_vllm is True
    assert config.vllm_enable_sleep_mode is True
    assert config.learning_rate == 1e-5
    assert config.num_train_epochs == 1
    assert config.per_device_train_batch_size == 4
    assert config.gradient_accumulation_steps == 32
    assert config.vllm_max_model_length == 6144
    assert config.zero_variance_filter is True
    assert config.zero_variance_filter_epsilon == 0.0
    assert config.precision == "auto"


def test_grpo_config_to_hf_args():
    config = GRPOTrainingConfig()
    d = config.to_hf_args(output_dir="/tmp/test")
    assert d["output_dir"] == "/tmp/test"
    assert d["num_generations"] == 8
    assert d["max_completion_length"] == 2048
    assert d["beta"] == 0.0
    assert d["temperature"] == 1.0
    assert d["epsilon"] == 0.2
    assert d["epsilon_high"] == 0.28
    assert d["loss_type"] == "dapo"
    assert d["scale_rewards"] == "group"
    assert d["vllm_enable_sleep_mode"] is True
    assert d["vllm_max_model_length"] == 6144
    assert d["save_strategy"] == "steps"
    # Should not contain fields that are not GRPOConfig params
    assert "eval_accumulation_steps" not in d
    assert "zero_variance_filter" not in d
    assert "zero_variance_filter_epsilon" not in d
    assert "precision" not in d


def test_grpo_config_crash_recovery_defaults():
    config = GRPOTrainingConfig()
    d = config.to_hf_args(output_dir="/tmp/test")
    assert d["save_strategy"] == "steps", "GRPO should save periodically for crash recovery"
    assert d["save_steps"] > 4, "GRPO save_steps should be larger than the SFT default"


def test_grpo_config_upcast_lm_head_fp32_default_off():
    config = GRPOTrainingConfig()
    assert config.upcast_lm_head_fp32 is False


def test_grpo_config_upcast_lm_head_fp32_excluded_from_hf_args():
    config = GRPOTrainingConfig(upcast_lm_head_fp32=True)
    d = config.to_hf_args(output_dir="/tmp/test")
    assert "upcast_lm_head_fp32" not in d


def test_grpo_config_zero_variance_filter_excluded_from_hf_args():
    config = GRPOTrainingConfig(zero_variance_filter=False, zero_variance_filter_epsilon=1e-6)
    d = config.to_hf_args(output_dir="/tmp/test")
    assert "zero_variance_filter" not in d
    assert "zero_variance_filter_epsilon" not in d


def test_grpo_vllm_sleep_mode_cli_default_on():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test"])
    assert args.grpo_vllm_sleep_mode is True


def test_grpo_vllm_sleep_mode_cli_explicit_disable():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--no-grpo-vllm-sleep-mode"])
    assert args.grpo_vllm_sleep_mode is False


def test_grpo_upcast_lm_head_fp32_cli_default_off():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test"])
    assert args.grpo_upcast_lm_head_fp32 is False


def test_grpo_upcast_lm_head_fp32_cli_enable():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--grpo-upcast-lm-head-fp32"])
    assert args.grpo_upcast_lm_head_fp32 is True


def test_grpo_upcast_lm_head_fp32_cli_explicit_disable():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--no-grpo-upcast-lm-head-fp32"])
    assert args.grpo_upcast_lm_head_fp32 is False


def test_grpo_precision_cli_default_auto():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test"])
    assert args.grpo_precision == "auto"


def test_grpo_precision_cli_fp16():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--grpo-precision", "fp16"])
    assert args.grpo_precision == "fp16"


def test_grpo_precision_cli_bf16():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--grpo-precision", "bf16"])
    assert args.grpo_precision == "bf16"


def test_grpo_precision_fp16_sets_hf_mixed_precision():
    d = GRPOTrainingConfig(precision="fp16").to_hf_args(output_dir="/tmp/test")
    assert d["fp16"] is True
    assert d["bf16"] is False


def test_grpo_precision_bf16_sets_hf_mixed_precision():
    d = GRPOTrainingConfig(precision="bf16").to_hf_args(output_dir="/tmp/test")
    assert d["fp16"] is False
    assert d["bf16"] is True


def test_grpo_zero_variance_filter_cli_default_on():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test"])
    assert args.grpo_zero_variance_filter is True
    assert args.grpo_zero_variance_filter_epsilon == 0.0


def test_grpo_zero_variance_filter_cli_disable():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--no-grpo-zero-variance-filter"])
    assert args.grpo_zero_variance_filter is False


def test_grpo_zero_variance_filter_cli_epsilon():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--grpo-zero-variance-filter-epsilon", "1e-6"])
    assert args.grpo_zero_variance_filter_epsilon == 1e-6


def test_grpo_eval_batch_size_cli_default_none():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test"])
    assert args.grpo_eval_batch_size is None


def test_grpo_eval_batch_size_cli_explicit():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--grpo-eval-batch-size", "3"])
    assert args.grpo_eval_batch_size == 3


def test_grpo_eval_batch_defaults_to_config_value_when_flag_absent():
    from tuning.training.pipeline.stages import _build_post_training_configs
    args = _parse_args(["--model", "llama3-8B", "--wandb-project", "test",
                        "--grpo-batch-size", "1"])
    checkpoint = {"checkpoint_path": "/x/llama3-8B_sft-0", "data_points_seen": 0}
    configs = _build_post_training_configs(args, "grpo", checkpoint, train_size=1000)
    assert configs.training_args.per_device_eval_batch_size == GRPOTrainingConfig().per_device_eval_batch_size


def test_grpo_eval_batch_explicit_override_wins():
    from tuning.training.pipeline.stages import _build_post_training_configs
    args = _parse_args(["--model", "llama3-8B", "--wandb-project", "test",
                        "--grpo-batch-size", "1", "--grpo-eval-batch-size", "5"])
    checkpoint = {"checkpoint_path": "/x/llama3-8B_sft-0", "data_points_seen": 0}
    configs = _build_post_training_configs(args, "grpo", checkpoint, train_size=1000)
    assert configs.training_args.per_device_eval_batch_size == 5


def test_dataset_cli_accepts_dapo():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--dataset", "dapo"])
    assert args.dataset == "dapo"


def test_grpo_profile_cli_toggle_is_not_supported():
    with pytest.raises(SystemExit):
        _parse_args([
            "--model", "qwen2-3B",
            "--wandb-project", "test",
            "--grpo-profile",
        ])


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


def test_pt_run_config_no_duplicate_model_name_with_dynamic_path():
    """When sft_run_config.dataset_config.dynamic_path already encodes a full SFT folder
    name (e.g. "llama3-3B_sft-gsm8k-1000"), PTRunConfig.run_name must not prepend the
    model_name a second time."""
    sft_folder_name = "llama3-3B_sft-gsm8k-1000"
    sft_config = SFTRunConfig(
        model_name="llama3-3B",
        model_name_hf="unsloth/Meta-Llama-3.1-3B",
        dataset_config=DatasetConfig(
            dataset="gsm8k", dataset_type="sft", train_size=1000,
            dynamic_path=sft_folder_name,
        ),
    )
    run_config = PTRunConfig(
        model_name="llama3-3B",
        model_name_hf="unsloth/Meta-Llama-3.1-3B",
        sft_run_config=sft_config,
        dataset_config=DatasetConfig(dataset="gsm8k", dataset_type="rlvr", train_size=500),
        pft_method="grpo",
    )
    assert "llama3-3B_llama3-3B" not in run_config.run_name
    assert run_config.run_name.startswith("llama3-3B_sft-gsm8k-1000_")
    assert run_config.run_name.endswith("_grpo")


def test_pt_run_config_output_dir_includes_wandb_run_id():
    """PTRunConfig.output_dir appends wandb_run_id when set."""
    run_config = PTRunConfig(
        model_name="llama3-3B",
        model_name_hf="unsloth/Meta-Llama-3.1-3B",
        dataset_config=DatasetConfig(dataset="gsm8k", dataset_type="rlvr", train_size=500),
        pft_method="grpo",
        wandb_run_id="abc123",
    )
    assert run_config.output_dir.endswith("_abc123")


def test_pt_run_config_output_dir_omits_empty_wandb_run_id():
    """An empty wandb_run_id must not produce a trailing underscore."""
    run_config = PTRunConfig(
        model_name="llama3-3B",
        model_name_hf="unsloth/Meta-Llama-3.1-3B",
        dataset_config=DatasetConfig(dataset="gsm8k", dataset_type="rlvr", train_size=500),
        pft_method="grpo",
    )
    assert not run_config.output_dir.endswith("_")
    assert run_config.output_dir.endswith("_grpo")


def test_sft_run_config_output_dir_includes_wandb_run_id():
    """SFTRunConfig.output_dir appends wandb_run_id when set."""
    sft_config = SFTRunConfig(
        model_name="llama3-3B",
        model_name_hf="unsloth/Meta-Llama-3.1-3B",
        dataset_config=DatasetConfig(dataset="gsm8k", dataset_type="sft", train_size=1000),
        wandb_run_id="xyz789",
    )
    assert sft_config.output_dir.endswith("_xyz789")


def test_sft_run_config_output_dir_omits_empty_wandb_run_id():
    """An empty wandb_run_id must not produce a trailing underscore on the SFT path."""
    sft_config = SFTRunConfig(
        model_name="llama3-3B",
        model_name_hf="unsloth/Meta-Llama-3.1-3B",
        dataset_config=DatasetConfig(dataset="gsm8k", dataset_type="sft", train_size=1000),
    )
    assert not sft_config.output_dir.endswith("_")
    assert sft_config.output_dir.endswith("sft-gsm8k-1000")


def test_sft_run_config_run_name_uses_dynamic_path_directly():
    """When dynamic_path is set, SFTRunConfig.run_name returns just the dynamic_path
    basename — model_name must not be prepended again."""
    sft_config = SFTRunConfig(
        model_name="llama3-3B",
        model_name_hf="unsloth/Meta-Llama-3.1-3B",
        dataset_config=DatasetConfig(
            dataset="gsm8k", dataset_type="sft", train_size=1000,
            dynamic_path="llama3-3B_sft-gsm8k-1000",
        ),
    )
    assert sft_config.run_name == "llama3-3B_sft-gsm8k-1000"


def test_sft_learning_rate_cli_arg():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--sft-learning-rate", "2e-4"])
    assert args.sft_learning_rate == 2e-4


def test_dpo_learning_rate_cli_arg():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test",
                        "--dpo-learning-rate", "1e-5"])
    assert args.dpo_learning_rate == 1e-5


def test_sft_learning_rate_default():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test"])
    assert args.sft_learning_rate == 5e-5


def test_dpo_learning_rate_default():
    args = _parse_args(["--model", "qwen2-3B", "--wandb-project", "test"])
    assert args.dpo_learning_rate == 5e-6


def test_training_arguments_config_seed_uses_global_default():
    tuning.config.set_seed(42)
    from tuning.training.config_training import TrainingArgumentsConfig
    d = TrainingArgumentsConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 42


def test_training_arguments_restore_callback_state_on_resume():
    from tuning.training.config_training import TrainingArgumentsConfig

    d = TrainingArgumentsConfig().to_hf_args(output_dir="/tmp/test")
    assert d["restore_callback_states_from_checkpoint"] is True


def test_training_arguments_config_seed_follows_set_seed():
    tuning.config.set_seed(7)
    from tuning.training.config_training import TrainingArgumentsConfig
    d = TrainingArgumentsConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 7


def test_dpo_config_seed_follows_set_seed():
    tuning.config.set_seed(13)
    from tuning.training.config_training import DPOTrainingConfig
    d = DPOTrainingConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 13


def test_grpo_config_seed_follows_set_seed():
    tuning.config.set_seed(99)
    from tuning.training.config_training import GRPOTrainingConfig
    d = GRPOTrainingConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 99


def test_lora_config_random_state_resolves_from_global():
    tuning.config.set_seed(7)
    from tuning.training.config_training import LoraConfig
    config = LoraConfig()
    assert config.random_state == 7


def test_lora_config_random_state_default_is_42():
    tuning.config.set_seed(42)
    from tuning.training.config_training import LoraConfig
    config = LoraConfig()
    assert config.random_state == 42


def test_lora_config_random_state_explicit_overrides_global():
    tuning.config.set_seed(7)
    from tuning.training.config_training import LoraConfig
    config = LoraConfig(random_state=99)
    assert config.random_state == 99
