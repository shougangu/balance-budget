"""Helpers for GRPO effective-batch filtering."""

from collections.abc import Sequence
from typing import Any

import torch


def nonzero_variance_row_mask(
    advantages: torch.Tensor,
    num_generations: int,
    epsilon: float = 0.0,
) -> torch.Tensor:
    """Return a row mask that keeps whole prompt groups with any nonzero advantage."""
    if num_generations < 1:
        raise ValueError("num_generations must be positive")
    if advantages.ndim == 0:
        raise ValueError("advantages must have a batch dimension")
    if advantages.shape[0] % num_generations != 0:
        raise ValueError(
            f"batch rows ({advantages.shape[0]}) must be divisible by "
            f"num_generations ({num_generations})"
        )

    row_signal = advantages.detach().reshape(advantages.shape[0], -1).abs().amax(dim=1)
    group_signal = row_signal.view(-1, num_generations).amax(dim=1)
    return (group_signal > epsilon).repeat_interleave(num_generations)


def select_row_aligned(
    inputs: dict[str, Any],
    selector: torch.Tensor | slice,
    row_count: int,
) -> dict[str, Any]:
    """Select rows for tensors/lists whose leading dimension matches row_count."""
    selected = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] == row_count:
                selected[key] = value[selector]
            else:
                selected[key] = value
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == row_count:
                selected[key] = value[selector] if isinstance(selector, slice) else [
                    item for item, keep in zip(value, selector.tolist(), strict=True) if keep
                ]
            else:
                selected[key] = value
        else:
            selected[key] = value
    return selected
