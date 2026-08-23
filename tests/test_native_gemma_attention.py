# ABOUTME: Tests restoring transformers' own Gemma-3 attention over Unsloth's patch.
# ABOUTME: Unsloth's forward computes attention in fp32 and reaches only flex or SDPA.

import types

import pytest

from tuning.training.model_utils import restore_native_attention

STASH = "_original_modeling_gemma3_Gemma3Attention_forward"


@pytest.fixture
def attention():
    modeling_gemma3 = pytest.importorskip("transformers.models.gemma3.modeling_gemma3")
    cls = modeling_gemma3.Gemma3Attention
    patched = cls.forward
    yield cls
    cls.forward = patched
    if hasattr(cls, STASH):
        delattr(cls, STASH)


def _model(implementation="flex_attention"):
    text_config = types.SimpleNamespace(_attn_implementation=implementation)
    config = types.SimpleNamespace(_attn_implementation=implementation, text_config=text_config)
    return types.SimpleNamespace(config=config)


def test_restores_the_stashed_forward(attention):
    def native(self):
        return "native"

    setattr(attention, STASH, native)
    attention.forward = lambda self: "unsloth"

    model = _model()
    assert restore_native_attention(model, "sdpa") is True
    assert attention.forward is native
    assert model.config._attn_implementation == "sdpa"
    assert model.config.text_config._attn_implementation == "sdpa"


def test_refuses_when_no_original_was_stashed(attention):
    with pytest.raises(RuntimeError, match="did not stash"):
        restore_native_attention(_model(), "sdpa")
