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


class _FakePageManager:
    def __init__(self):
        self.paged_tensors = []

    def prefetch_all(self, to_cpu=False):
        pass


class _FakePagedOptimizer:
    """Stands in for a bitsandbytes paged optimizer whose state came off disk."""

    is_paged = True

    def __init__(self, state):
        self.page_mng = _FakePageManager()
        self.state = state
        self.optimizer = None


def _loaded_state(numel):
    import torch
    return {"state1": torch.zeros(numel), "state2": torch.ones(numel)}


def test_repages_optimizer_state_loaded_from_a_checkpoint(monkeypatch):
    import torch

    def fake_buffer(tensor):
        buff = torch.empty_like(tensor)
        buff.is_paged = True
        return buff

    monkeypatch.setattr(module, "_paged_buffer", fake_buffer)
    state = _loaded_state(100_000)
    optimizer = _FakePagedOptimizer({"p": state})

    callback = module.PagedOptimizerOffloadCallback()
    callback.on_train_begin(args=None, state=None, control=None, optimizer=optimizer)

    assert state["state1"].is_paged and state["state2"].is_paged
    assert torch.equal(state["state2"], torch.ones(100_000))
    assert len(optimizer.page_mng.paged_tensors) == 2


def test_leaves_already_paged_state_untouched(monkeypatch):
    monkeypatch.setattr(module, "_paged_buffer",
                        MagicMock(side_effect=AssertionError("must not reallocate")))
    state = _loaded_state(100_000)
    for tensor in state.values():
        tensor.is_paged = True
    optimizer = _FakePagedOptimizer({"p": state})

    module.PagedOptimizerOffloadCallback().on_train_begin(
        args=None, state=None, control=None, optimizer=optimizer)

    assert optimizer.page_mng.paged_tensors == []


def test_leaves_small_state_tensors_unpaged(monkeypatch):
    monkeypatch.setattr(module, "_paged_buffer",
                        MagicMock(side_effect=AssertionError("must not reallocate")))
    optimizer = _FakePagedOptimizer({"p": _loaded_state(1024)})

    module.PagedOptimizerOffloadCallback().on_train_begin(
        args=None, state=None, control=None, optimizer=optimizer)

    assert optimizer.page_mng.paged_tensors == []
