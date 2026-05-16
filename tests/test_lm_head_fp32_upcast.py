# ABOUTME: Unit tests for the fp32 lm_head upcast helper used by GRPO training.
# ABOUTME: Covers tied/untied embedding handling and forward-time dtype guarantees.

import sys
from types import ModuleType

import pytest
import torch
import torch.nn as nn


if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub


def _tiny_model(*, tied: bool, vocab: int = 64, hidden: int = 8) -> nn.Module:
    """Build a minimal HF-shaped module: .model.embed_tokens + .lm_head, in bf16."""
    inner = nn.Module()
    inner.embed_tokens = nn.Embedding(vocab, hidden)
    outer = nn.Module()
    outer.model = inner
    outer.lm_head = nn.Linear(hidden, vocab, bias=False)
    if tied:
        outer.lm_head.weight = outer.model.embed_tokens.weight
    for p in outer.parameters():
        p.data = p.data.to(torch.bfloat16)
    outer.get_input_embeddings = lambda: outer.model.embed_tokens
    outer.get_output_embeddings = lambda: outer.lm_head
    return outer


def test_upcast_untied_casts_lm_head_only():
    from tuning.training.model_utils import upcast_lm_head_to_fp32

    model = _tiny_model(tied=False)
    upcast_lm_head_to_fp32(model)

    assert model.lm_head.weight.dtype == torch.float32
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16


def test_upcast_tied_unties_and_casts_lm_head_only():
    from tuning.training.model_utils import upcast_lm_head_to_fp32

    model = _tiny_model(tied=True)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()

    upcast_lm_head_to_fp32(model)

    assert model.lm_head.weight.dtype == torch.float32
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()


def test_upcast_forward_yields_fp32_logits():
    from tuning.training.model_utils import upcast_lm_head_to_fp32

    model = _tiny_model(tied=False)
    upcast_lm_head_to_fp32(model)

    hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    logits = model.lm_head(hidden)
    assert logits.dtype == torch.float32
    assert logits.shape == (2, 4, 64)


def test_upcast_preserves_requires_grad_state():
    from tuning.training.model_utils import upcast_lm_head_to_fp32

    model = _tiny_model(tied=False)
    model.lm_head.weight.requires_grad = False
    upcast_lm_head_to_fp32(model)
    assert model.lm_head.weight.requires_grad is False

    model = _tiny_model(tied=True)
    model.lm_head.weight.requires_grad = True
    upcast_lm_head_to_fp32(model)
    assert model.lm_head.weight.requires_grad is True


def test_upcast_preserves_logit_values_within_tolerance():
    from tuning.training.model_utils import upcast_lm_head_to_fp32

    torch.manual_seed(0)
    baseline = _tiny_model(tied=False)
    upcast = _tiny_model(tied=False)
    upcast.lm_head.weight.data.copy_(baseline.lm_head.weight.data)
    upcast.model.embed_tokens.weight.data.copy_(baseline.model.embed_tokens.weight.data)

    upcast_lm_head_to_fp32(upcast)

    hidden = torch.randn(2, 4, 8, dtype=torch.bfloat16)
    bf16_logits = baseline.lm_head(hidden).to(torch.float32)
    fp32_logits = upcast.lm_head(hidden)
    assert torch.allclose(bf16_logits, fp32_logits, atol=5e-2, rtol=5e-2)
