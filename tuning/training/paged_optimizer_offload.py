# ABOUTME: Trainer callback that migrates bitsandbytes paged optimizer state to host
# ABOUTME: after every optimizer step so the states only occupy the GPU during the step.

import torch
from transformers import TrainerCallback

from tuning.training.passk.runners import (
    _offload_paged_optimizer_state,
    _unwrap_paged_optimizer,
)


# bitsandbytes pages a state buffer only when its parameter has at least this many
# elements (`BaseOptimizer.get_state_buffer`); smaller states stay ordinary tensors.
_MIN_PAGED_ELEMENTS = 1e5


def _paged_buffer(tensor):
    """Allocate a bitsandbytes paged buffer shaped like tensor."""
    import bitsandbytes.functional as F

    return F.get_paged(*tensor.shape, dtype=tensor.dtype, device=tensor.device)


def repage_optimizer_state(optimizer) -> int:
    """Move optimizer state restored from a checkpoint back into paged memory.

    `load_state_dict` replaces the managed allocations bitsandbytes made in
    `init_state` with ordinary CUDA tensors, so a resumed run keeps its whole
    optimizer state resident and `prefetch_state` skips it. Reallocating each
    state as a paged buffer restores the migration the memory budget depends on.
    """
    paged_optimizer = _unwrap_paged_optimizer(optimizer)
    if paged_optimizer is None:
        return 0
    repaged = 0
    for state in paged_optimizer.state.values():
        for key in ("state1", "state2"):
            tensor = state.get(key) if hasattr(state, "get") else None
            if tensor is None or getattr(tensor, "is_paged", False):
                continue
            if tensor.numel() < _MIN_PAGED_ELEMENTS:
                continue
            buffer = _paged_buffer(tensor)
            buffer.copy_(tensor)
            state[key] = buffer
            paged_optimizer.page_mng.paged_tensors.append(buffer)
            repaged += 1
    return repaged


class PagedOptimizerOffloadCallback(TrainerCallback):
    """Keep paged AdamW state off the GPU during gradient accumulation.

    CUDA managed memory is not evicted when PyTorch's allocator runs out of
    device memory, so the 8-bit states stay resident after the first step and
    an fp32 full fine-tune (params + grads = 64 GB for 8B) has no room left
    for activations. The next optimizer step pages each state tensor back in
    as it is used, and those pages can only land in memory PyTorch's caching
    allocator has not reserved, so the cache is released before every step.
    """

    def __init__(self):
        self._active = True

    def on_train_begin(self, args, state, control, **kwargs):
        repaged = repage_optimizer_state(kwargs.get("optimizer"))
        if repaged:
            torch.cuda.empty_cache()
            print(f"[PagedOptimizer] Repaged {repaged} restored state tensors")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._active:
            torch.cuda.empty_cache()

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self._active:
            return
        self._active = _offload_paged_optimizer_state(kwargs["optimizer"])
