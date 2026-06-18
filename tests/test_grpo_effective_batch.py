# ABOUTME: Tests GRPO effective-batch helpers for zero-variance prompt filtering.

import pytest
import torch

from tuning.training.grpo_effective_batch import (
    nonzero_variance_row_mask,
    select_row_aligned,
)


def test_nonzero_variance_row_mask_keeps_whole_prompt_groups():
    advantages = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    mask = nonzero_variance_row_mask(advantages, num_generations=3)

    assert mask.tolist() == [True, True, True, False, False, False]


def test_nonzero_variance_row_mask_supports_token_shaped_advantages():
    advantages = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.2],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    mask = nonzero_variance_row_mask(advantages, num_generations=2)

    assert mask.tolist() == [True, True, False, False]


def test_nonzero_variance_row_mask_rejects_partial_groups():
    with pytest.raises(ValueError):
        nonzero_variance_row_mask(torch.zeros(3), num_generations=2)


def test_select_row_aligned_filters_only_batch_aligned_values():
    inputs = {
        "batch_tensor": torch.arange(6).reshape(3, 2),
        "scalar": torch.tensor(7),
        "short_tensor": torch.arange(2),
        "batch_list": ["a", "b", "c"],
        "other": "unchanged",
    }

    selected = select_row_aligned(
        inputs,
        torch.tensor([True, False, True]),
        row_count=3,
    )

    assert selected["batch_tensor"].tolist() == [[0, 1], [4, 5]]
    assert selected["scalar"].item() == 7
    assert selected["short_tensor"].tolist() == [0, 1]
    assert selected["batch_list"] == ["a", "c"]
    assert selected["other"] == "unchanged"
