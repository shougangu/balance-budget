# ABOUTME: Tests for DDP-related changes in pipeline.cli and pipeline.orchestrator.
# ABOUTME: CPU-only; mocks heavy imports.

import os

from tuning.training.pipeline.cli import init_cuda_env


def test_init_cuda_env_noop_when_local_rank_set(monkeypatch):
    """Under torchrun (LOCAL_RANK set), init_cuda_env must not mutate CUDA_VISIBLE_DEVICES."""
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES_ALL", raising=False)

    init_cuda_env()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "CUDA_VISIBLE_DEVICES_ALL" not in os.environ


def test_init_cuda_env_pins_gpu0_without_local_rank(monkeypatch):
    """Without torchrun (no LOCAL_RANK), legacy behavior: pin GPU 0, save the rest."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES_ALL", raising=False)

    init_cuda_env()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES_ALL"] == "0,1,2,3"
