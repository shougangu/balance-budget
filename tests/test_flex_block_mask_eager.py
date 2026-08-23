# ABOUTME: Tests that flex attention's block mask is built eagerly instead of compiled.
# ABOUTME: The compiled builder faults with an illegal memory access on Gemma-3.

import pytest
import torch

import types

from tuning.training.model_utils import (
    enable_eager_block_mask,
    install_eager_block_mask_patch,
)


@pytest.fixture
def masking_utils():
    mu = pytest.importorskip("transformers.masking_utils")
    original = mu.create_block_mask
    yield mu
    mu.create_block_mask = original
    mu._balance_budget_eager_block_mask = False


def _model(implementation, text_implementation=None):
    text_config = (
        types.SimpleNamespace(_attn_implementation=text_implementation)
        if text_implementation is not None
        else None
    )
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            _attn_implementation=implementation, text_config=text_config
        )
    )


def test_leaves_the_builder_alone_when_attention_is_not_flex(masking_utils):
    from torch.nn.attention.flex_attention import create_block_mask

    assert enable_eager_block_mask(_model("sdpa", "sdpa")) is False
    assert masking_utils.create_block_mask is create_block_mask


def test_patches_when_the_text_tower_attends_through_flex(masking_utils):
    """Gemma-3 records the implementation on text_config, not the outer config."""
    from torch.nn.attention.flex_attention import create_block_mask

    assert enable_eager_block_mask(_model("eager", "flex_attention")) is True
    assert masking_utils.create_block_mask is not create_block_mask


def test_patches_when_the_outer_config_attends_through_flex(masking_utils):
    assert enable_eager_block_mask(_model("flex_attention")) is True


def test_patch_replaces_the_builder(masking_utils):
    from torch.nn.attention.flex_attention import create_block_mask

    assert masking_utils.create_block_mask is create_block_mask
    assert install_eager_block_mask_patch() is True
    assert masking_utils.create_block_mask is not create_block_mask
    assert install_eager_block_mask_patch() is False


def test_builder_never_compiles(masking_utils, monkeypatch):
    install_eager_block_mask_patch()

    def refuse(*args, **kwargs):
        raise AssertionError("create_block_mask was compiled")

    monkeypatch.setattr(torch, "compile", refuse)

    attention_mask = torch.ones(2, 384, dtype=torch.long)
    attention_mask[1, 300:] = 0
    masking_utils.flex_attention_mask(
        batch_size=2,
        cache_position=torch.arange(384),
        kv_length=384,
        attention_mask=attention_mask,
    )


def test_eager_mask_matches_the_dense_truth(masking_utils):
    install_eager_block_mask_patch()

    attention_mask = torch.ones(2, 384, dtype=torch.long)
    attention_mask[1, 300:] = 0
    block_mask = masking_utils.flex_attention_mask(
        batch_size=2,
        cache_position=torch.arange(384),
        kv_length=384,
        attention_mask=attention_mask,
    )

    query = torch.arange(384).view(-1, 1)
    key = torch.arange(384).view(1, -1)
    causal = query >= key
    expected = torch.stack([causal & attention_mask[b].bool().view(1, -1) for b in range(2)])
    expected = expected.view(2, 1, 3, 128, 3, 128).any(dim=-1).any(dim=-2)

    assert torch.equal(block_mask.to_dense().bool().view(2, 1, 3, 3), expected)


def test_sliding_window_mask_builds_eagerly(masking_utils, monkeypatch):
    install_eager_block_mask_patch()

    def refuse(*args, **kwargs):
        raise AssertionError("create_block_mask was compiled")

    monkeypatch.setattr(torch, "compile", refuse)

    sliding = masking_utils.sliding_window_overlay(128)
    block_mask = masking_utils.flex_attention_mask(
        batch_size=2,
        cache_position=torch.arange(384),
        kv_length=384,
        mask_function=masking_utils.and_masks(masking_utils.causal_mask_function, sliding),
        attention_mask=torch.ones(2, 384, dtype=torch.long),
    )

    query = torch.arange(384).view(-1, 1)
    key = torch.arange(384).view(1, -1)
    expected = ((query >= key) & (key > query - 128)).view(1, 1, 3, 128, 3, 128)
    expected = expected.any(dim=-1).any(dim=-2).expand(2, 1, 3, 3)

    assert torch.equal(block_mask.to_dense().bool().view(2, 1, 3, 3), expected)
