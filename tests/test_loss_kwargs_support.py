# ABOUTME: Unit tests for keeping the SFT loss invariant to gradient accumulation.
# ABOUTME: Covers models that cannot normalize their own loss by the batch token count.

import sys
from types import ModuleType

import torch.nn as nn


if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub


class _Accelerator:
    def unwrap_model(self, model):
        return model


class _Trainer:
    """Minimal stand-in exposing the attributes the helper reads and writes."""

    def __init__(self, model, model_accepts_loss_kwargs):
        self.model = model
        self.accelerator = _Accelerator()
        self.model_accepts_loss_kwargs = model_accepts_loss_kwargs


class _DeclinesLossKwargs(nn.Module):
    accepts_loss_kwargs = False


class _AcceptsLossKwargs(nn.Module):
    accepts_loss_kwargs = True


class _Undeclared(nn.Module):
    pass


def test_model_declining_loss_kwargs_disables_them():
    from tuning.training.model_utils import disable_loss_kwargs_if_unsupported

    trainer = _Trainer(_DeclinesLossKwargs(), model_accepts_loss_kwargs=True)

    assert disable_loss_kwargs_if_unsupported(trainer) is True
    assert trainer.model_accepts_loss_kwargs is False


def test_model_accepting_loss_kwargs_is_left_alone():
    from tuning.training.model_utils import disable_loss_kwargs_if_unsupported

    trainer = _Trainer(_AcceptsLossKwargs(), model_accepts_loss_kwargs=True)

    assert disable_loss_kwargs_if_unsupported(trainer) is False
    assert trainer.model_accepts_loss_kwargs is True


def test_model_without_a_declaration_is_left_alone():
    from tuning.training.model_utils import disable_loss_kwargs_if_unsupported

    trainer = _Trainer(_Undeclared(), model_accepts_loss_kwargs=True)

    assert disable_loss_kwargs_if_unsupported(trainer) is False
    assert trainer.model_accepts_loss_kwargs is True


def test_peft_wrapped_model_declaration_is_read_from_the_base_model():
    from tuning.training.model_utils import disable_loss_kwargs_if_unsupported

    base = _DeclinesLossKwargs()

    class _PeftModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = nn.Module()
            self.base_model.model = base

        def get_base_model(self):
            return base

    trainer = _Trainer(_PeftModel(), model_accepts_loss_kwargs=True)

    assert disable_loss_kwargs_if_unsupported(trainer) is True
    assert trainer.model_accepts_loss_kwargs is False


def test_multimodal_gemma3_declines_loss_kwargs_upstream():
    """Guards the assumption the helper exists for: Gemma-3 cannot honor loss kwargs.

    Only the multimodal classes declare this; the text-only Gemma3ForCausalLM
    normalizes its own loss and is left alone.
    """
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3ForCausalLM,
        Gemma3ForConditionalGeneration,
        Gemma3Model,
    )

    assert Gemma3ForConditionalGeneration.accepts_loss_kwargs is False
    assert Gemma3Model.accepts_loss_kwargs is False
    assert getattr(Gemma3ForCausalLM, "accepts_loss_kwargs", True) is True
