# ABOUTME: Regression tests for restoring PEFT policy weights from GRPO checkpoints.
# ABOUTME: Covers the root default adapter plus nested TRL reference-adapter layout.

from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM, Trainer


def _tiny_peft_model():
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    # TRL does this whenever a PeftModel is passed to GRPOTrainer with beta != 0.
    model.add_adapter("ref", model.peft_config["default"])
    return model


def _lora_b(model, adapter):
    suffix = f"lora_B.{adapter}.weight"
    return next(param for name, param in model.named_parameters() if name.endswith(suffix))


def test_grpo_resume_restores_root_policy_with_nested_ref_adapter(tmp_path):
    """The trained root adapter must not be skipped merely because ref/ exists."""
    from tuning.training.model_utils import load_trainable_adapter

    saved = _tiny_peft_model()
    with torch.no_grad():
        _lora_b(saved, "default").fill_(0.75)
        _lora_b(saved, "ref").fill_(0.25)
    saved.save_pretrained(tmp_path)

    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "ref" / "adapter_model.safetensors").is_file()

    resumed = _tiny_peft_model()
    assert torch.count_nonzero(_lora_b(resumed, "default")) == 0
    assert torch.count_nonzero(_lora_b(resumed, "ref")) == 0

    # Reproduce Transformers 4.57's mixed-layout behavior: ref/ is restored,
    # but the trained adapter_model.safetensors at the root is ignored.
    loader = SimpleNamespace(
        model=resumed,
        args=SimpleNamespace(save_safetensors=True),
        is_fsdp_enabled=False,
    )
    Trainer._load_from_checkpoint(loader, str(tmp_path))
    assert torch.count_nonzero(_lora_b(resumed, "default")) == 0
    torch.testing.assert_close(
        _lora_b(resumed, "ref"), torch.full_like(_lora_b(resumed, "ref"), 0.25)
    )

    load_trainable_adapter(resumed, tmp_path)
    assert resumed.active_adapters == ["default"]
    assert _lora_b(resumed, "default").requires_grad
    assert not _lora_b(resumed, "ref").requires_grad
    torch.testing.assert_close(
        _lora_b(resumed, "default"), torch.full_like(_lora_b(resumed, "default"), 0.75)
    )
    torch.testing.assert_close(
        _lora_b(resumed, "ref"), torch.full_like(_lora_b(resumed, "ref"), 0.25)
    )
