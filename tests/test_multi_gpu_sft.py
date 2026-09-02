# ABOUTME: Tests the multi-GPU SFT settings: FSDP sharding with one unit per decoder
# ABOUTME: layer, and the budget clock scaled by the GPU count, as accelerate receives them.

import os

import pytest

from tuning.training.config_training import TrainingArgumentsConfig, multi_gpu_training_args


@pytest.fixture
def clean_fsdp_env():
    """TrainingArguments hands FSDP settings to accelerate through FSDP_* env vars."""
    saved = dict(os.environ)
    for key in list(os.environ):
        if key.startswith("FSDP_") or key == "ACCELERATE_USE_FSDP":
            del os.environ[key]
    yield os.environ
    os.environ.clear()
    os.environ.update(saved)


def test_multi_gpu_wraps_each_decoder_layer_in_its_own_fsdp_unit(tmp_path, clean_fsdp_env):
    """Without an auto-wrap policy the whole model is one FSDP unit, so every step
    all-gathers the full bf16 weights and gradients (28 GB each at 14B)."""
    from trl import SFTConfig

    training_args = multi_gpu_training_args(TrainingArgumentsConfig(), num_gpus=8)
    hf_args = training_args.to_hf_args(output_dir=str(tmp_path))
    hf_args.update(fp16=False, bf16=False, use_cpu=True)
    SFTConfig(**hf_args)

    assert clean_fsdp_env["FSDP_AUTO_WRAP_POLICY"] == "TRANSFORMER_BASED_WRAP"
    # No class list: accelerate falls back to the model's _no_split_modules.
    assert "FSDP_TRANSFORMER_CLS_TO_WRAP" not in clean_fsdp_env
    assert clean_fsdp_env["FSDP_SHARDING_STRATEGY"] == "1"
    assert training_args.gpu_minute_multiplier == 8.0


def test_multi_gpu_runs_fsdp2_with_sharded_periodic_checkpoints(tmp_path, clean_fsdp_env):
    """transformers forwards no version key from fsdp_config, so accelerate only
    learns the version from the env; sharded periodic saves keep the rank-0
    CPU gather of fp32 weights plus Adam state (~168 GB at 14B) off the step path."""
    from accelerate.utils import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp import StateDictType
    from trl import SFTConfig

    training_args = multi_gpu_training_args(TrainingArgumentsConfig(), num_gpus=8)
    hf_args = training_args.to_hf_args(output_dir=str(tmp_path))
    hf_args.update(fp16=False, bf16=False, use_cpu=True)
    SFTConfig(**hf_args)

    plugin = FullyShardedDataParallelPlugin()
    assert plugin.fsdp_version == 2
    assert plugin.state_dict_type == StateDictType.SHARDED_STATE_DICT


class _FakeModel:
    def __init__(self):
        self.saved = None

    def save_pretrained(self, path, state_dict=None):
        import json
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as fh:
            json.dump({"torch_dtype": "float32"}, fh)
        self.saved = {"path": path, "state_dict": state_dict}


class _FakeTokenizer:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = path


class _GatheringAccelerator:
    """accelerate.get_state_dict under FSDP: the full dict lands on rank 0 only."""

    def __init__(self, state_dict):
        self._state_dict = state_dict

    def get_state_dict(self, model):
        return self._state_dict

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        pass


def _fsdp_trainer(process_index, state_dict):
    from types import SimpleNamespace
    from transformers import TrainerState

    def refuse(*_args, **_kwargs):
        raise AssertionError("trainer.save_model writes no weights under a sharded state dict")

    return SimpleNamespace(
        args=SimpleNamespace(fsdp="full_shard auto_wrap", process_index=process_index, to_dict=lambda: {}),
        state=TrainerState(),
        accelerator=_GatheringAccelerator(state_dict),
        save_model=refuse,
    )


def test_final_save_under_fsdp_gathers_bf16_weights_on_rank_zero(tmp_path):
    import json
    import torch
    from tuning.training.model_utils import save_trained_model

    model, tokenizer = _FakeModel(), _FakeTokenizer()
    gathered = {"w": torch.ones(2, dtype=torch.float32), "step": torch.tensor(3)}
    save_trained_model(model, tokenizer, _fsdp_trainer(0, gathered), str(tmp_path))

    assert model.saved["path"] == str(tmp_path)
    assert model.saved["state_dict"]["w"].dtype == torch.bfloat16
    assert model.saved["state_dict"]["step"].dtype == torch.int64
    assert json.load(open(tmp_path / "config.json"))["torch_dtype"] == "bfloat16"
    assert tokenizer.saved_to == str(tmp_path)


def test_final_save_under_fsdp_writes_nothing_off_rank_zero(tmp_path):
    from tuning.training.model_utils import save_trained_model

    model, tokenizer = _FakeModel(), _FakeTokenizer()
    save_trained_model(model, tokenizer, _fsdp_trainer(1, {}), str(tmp_path))

    assert model.saved is None
    assert tokenizer.saved_to is None
