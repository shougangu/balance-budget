# ABOUTME: Tests the callback that migrates paged optimizer state to host after
# ABOUTME: each optimizer step so fp32 full fine-tuning fits one GPU.

import sys
from types import ModuleType
from unittest.mock import MagicMock

if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())

from tuning.training import paged_optimizer_offload as module


def test_offloads_optimizer_after_each_step(monkeypatch):
    offload = MagicMock(return_value=True)
    monkeypatch.setattr(module, "_offload_paged_optimizer_state", offload)
    optimizer = MagicMock(name="optimizer")

    callback = module.PagedOptimizerOffloadCallback()
    callback.on_optimizer_step(args=None, state=None, control=None, optimizer=optimizer)

    offload.assert_called_once_with(optimizer)


def test_stops_after_optimizer_reports_not_paged(monkeypatch):
    offload = MagicMock(return_value=False)
    monkeypatch.setattr(module, "_offload_paged_optimizer_state", offload)
    optimizer = MagicMock(name="optimizer")

    callback = module.PagedOptimizerOffloadCallback()
    callback.on_optimizer_step(args=None, state=None, control=None, optimizer=optimizer)
    callback.on_optimizer_step(args=None, state=None, control=None, optimizer=optimizer)

    assert offload.call_count == 1


def test_releases_cached_device_memory_before_each_step(monkeypatch):
    import torch

    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    callback = module.PagedOptimizerOffloadCallback()
    callback.on_pre_optimizer_step(args=None, state=None, control=None)

    empty_cache.assert_called_once_with()


def test_stops_releasing_cache_once_optimizer_is_not_paged(monkeypatch):
    import torch

    empty_cache = MagicMock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(module, "_offload_paged_optimizer_state", MagicMock(return_value=False))

    callback = module.PagedOptimizerOffloadCallback()
    callback.on_optimizer_step(args=None, state=None, control=None, optimizer=MagicMock())
    callback.on_pre_optimizer_step(args=None, state=None, control=None)

    empty_cache.assert_not_called()
