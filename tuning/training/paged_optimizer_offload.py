# ABOUTME: Trainer callback that migrates bitsandbytes paged optimizer state to host
# ABOUTME: after every optimizer step so the states only occupy the GPU during the step.

import torch
from transformers import TrainerCallback

from tuning.training.passk.runners import _offload_paged_optimizer_state


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

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._active:
            torch.cuda.empty_cache()

    def on_optimizer_step(self, args, state, control, **kwargs):
        if not self._active:
            return
        self._active = _offload_paged_optimizer_state(kwargs["optimizer"])
