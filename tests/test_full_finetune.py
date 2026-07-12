# ABOUTME: Tests full fine-tuning SFT: config flag, PEFT-free model loading, and
# ABOUTME: vLLM eval runners serving full-model checkpoints instead of base+LoRA.

import sys
from types import ModuleType
from unittest.mock import MagicMock

# Stub unsloth before importing config (crashes without GPU)
if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())

import pytest

from tuning.training.config_training import (
    LoraConfig, ModelLoadConfig, TrainingArgumentsConfig,
)
from tuning.training.pipeline.cli import _parse_args


# ---------------------------------------------------------------------------
# Config flag
# ---------------------------------------------------------------------------

def test_full_finetune_defaults_false():
    assert TrainingArgumentsConfig().full_finetune is False


def test_to_hf_args_pops_full_finetune():
    config = TrainingArgumentsConfig(full_finetune=True)
    d = config.to_hf_args(output_dir="/tmp/test")
    assert "full_finetune" not in d


# ---------------------------------------------------------------------------
# Model loading without PEFT wrapping
# ---------------------------------------------------------------------------

@pytest.fixture
def hf_load_mocks(monkeypatch, tmp_path):
    """Patch HF/PEFT entry points used by the non-unsloth load branch."""
    import transformers
    import peft

    model = MagicMock(name="base_model")
    tokenizer = MagicMock(name="tokenizer")
    from_pretrained_model = MagicMock(return_value=model)
    from_pretrained_tok = MagicMock(return_value=tokenizer)
    get_peft_model = MagicMock(side_effect=AssertionError(
        "get_peft_model must not be called for full fine-tuning"))

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained_model)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", from_pretrained_tok)
    monkeypatch.setattr(peft, "get_peft_model", get_peft_model)

    (tmp_path / "config.json").write_text("{}")
    return model, tokenizer, str(tmp_path)


def test_load_full_finetune_skips_peft(hf_load_mocks):
    from tuning.training.model_utils import load_model_with_lora

    base_model, base_tokenizer, model_path = hf_load_mocks
    model, tokenizer = load_model_with_lora(
        model_path=model_path,
        model_name="gemma3-12B",
        model_load_config=ModelLoadConfig(),
        lora_config=LoraConfig(),
        use_unsloth=False,
        full_finetune=True,
    )
    assert model is base_model
    assert tokenizer is base_tokenizer
    model.gradient_checkpointing_enable.assert_called_once()


def test_load_full_finetune_rejects_unsloth(hf_load_mocks):
    from tuning.training.model_utils import load_model_with_lora

    _, _, model_path = hf_load_mocks
    with pytest.raises(ValueError, match="unsloth"):
        load_model_with_lora(
            model_path=model_path,
            model_name="gemma3-12B",
            model_load_config=ModelLoadConfig(),
            lora_config=LoraConfig(),
            use_unsloth=True,
            full_finetune=True,
        )


def test_load_full_finetune_rejects_4bit(hf_load_mocks):
    from tuning.training.model_utils import load_model_with_lora

    _, _, model_path = hf_load_mocks
    with pytest.raises(ValueError, match="4bit|4-bit"):
        load_model_with_lora(
            model_path=model_path,
            model_name="gemma3-12B",
            model_load_config=ModelLoadConfig(load_in_4bit=True),
            lora_config=LoraConfig(),
            use_unsloth=False,
            full_finetune=True,
        )


# ---------------------------------------------------------------------------
# Runner checkpoint resolution: adapter vs full checkpoint
# ---------------------------------------------------------------------------

def _runner_config():
    from tuning.training.passk.runners import RunnerConfig
    return RunnerConfig(
        base_model_hf="base/model",
        vllm_gpu_memory_utilization=0.4,
        lora_max_rank=32,
        chat_template="{{ messages }}",
        temperature=0.5,
        max_tokens=64,
        available_gpus=["0"],
        num_inference_gpus=1,
    )


def test_resolve_checkpoint_adapter(tmp_path):
    from tuning.training.passk.runners import _resolve_checkpoint

    (tmp_path / "adapter_config.json").write_text("{}")
    model_path, lora_path = _resolve_checkpoint("base/model", str(tmp_path))
    assert model_path == "base/model"
    assert lora_path == str(tmp_path)


def test_resolve_checkpoint_full_model(tmp_path):
    from tuning.training.passk.runners import _resolve_checkpoint

    (tmp_path / "config.json").write_text("{}")
    model_path, lora_path = _resolve_checkpoint("base/model", str(tmp_path))
    assert model_path == str(tmp_path)
    assert lora_path is None


def test_resolve_checkpoint_none():
    from tuning.training.passk.runners import _resolve_checkpoint

    assert _resolve_checkpoint("base/model", None) == ("base/model", None)


def test_make_llm_full_checkpoint_disables_lora(tmp_path):
    import vllm
    from tuning.training.passk.runners import _make_llm

    vllm.LLM.reset_mock()
    _make_llm(_runner_config(), model_path=str(tmp_path), enable_lora=False)
    kwargs = vllm.LLM.call_args.kwargs
    assert kwargs["model"] == str(tmp_path)
    assert not kwargs.get("enable_lora", False)


def test_make_llm_default_serves_base_with_lora():
    import vllm
    from tuning.training.passk.runners import _make_llm

    vllm.LLM.reset_mock()
    _make_llm(_runner_config())
    kwargs = vllm.LLM.call_args.kwargs
    assert kwargs["model"] == "base/model"
    assert kwargs["enable_lora"] is True


def test_persistent_runner_rejects_full_checkpoint(tmp_path):
    from tuning.training.passk.runners import PersistentVLLMRunner

    (tmp_path / "config.json").write_text("{}")
    runner = PersistentVLLMRunner(_runner_config())
    with pytest.raises(RuntimeError, match="full"):
        runner.run(MagicMock(), MagicMock(), str(tmp_path))


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def test_cli_full_finetune_defaults_off():
    args = _parse_args(["--model", "gemma3-12B", "--wandb-project", "test"])
    assert args.sft_full_finetune is False
    assert args.sft_optim == "adamw_8bit"


def test_cli_full_finetune_flag():
    args = _parse_args([
        "--model", "gemma3-12B", "--wandb-project", "test",
        "--sft-full-finetune", "--sft-optim", "paged_adamw_8bit",
    ])
    assert args.sft_full_finetune is True
    assert args.sft_optim == "paged_adamw_8bit"
