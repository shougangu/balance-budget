# ABOUTME: Tests for DDP-related changes in pipeline.cli and pipeline.orchestrator.
# ABOUTME: CPU-only; mocks heavy imports.

import os

import pytest

from tuning.training.pipeline.cli import init_cuda_env
from tuning.training.pipeline.vllm_sidecar import resolve_grpo_server_split


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


from tuning.training.pipeline.cli import _parse_args


def test_grpo_num_gpus_default():
    args = _parse_args(["--model", "qwen2-2B", "--wandb-project", "test"])
    assert args.grpo_num_gpus == 2


def test_grpo_server_split_two_gpus():
    split = resolve_grpo_server_split(grpo_num_gpus=2)
    assert split.trainer_gpus == ("0",)
    assert split.server_gpus == ("1",)
    assert len(split.server_gpus) == 1


def test_grpo_server_split_uses_all_non_trainer_gpus_as_dp_workers():
    split = resolve_grpo_server_split(grpo_num_gpus=8)
    assert split.trainer_gpus == ("0",)
    assert split.server_gpus == tuple(str(i) for i in range(1, 8))
    assert len(split.server_gpus) == 7


def test_grpo_server_split_rejects_one_gpu():
    with pytest.raises(ValueError, match=">= 2"):
        resolve_grpo_server_split(grpo_num_gpus=1)


def test_grpo_num_gpus_override():
    args = _parse_args([
        "--model", "qwen2-2B",
        "--wandb-project", "test",
        "--grpo-num-gpus", "4",
    ])
    assert args.grpo_num_gpus == 4


from unittest.mock import patch, MagicMock
from types import SimpleNamespace


def test_dispatch_grpo_passes_gres_to_sbatch(tmp_path):
    """When pt_method='grpo' and grpo_num_gpus=4, dispatch injects --gres=gpu:4."""
    from tuning.training.pipeline.orchestrator import _dispatch_parallel_workers

    metadata_file = tmp_path / "meta.jsonl"
    metadata_file.write_text("")

    args = SimpleNamespace(
        post_training_method="grpo",
        grpo_num_gpus=4,
        parallel=2,
    )

    fake_result = MagicMock(returncode=0,
                            stdout="Submitted batch job 12345\n",
                            stderr="")
    with patch("tuning.training.pipeline.orchestrator.subprocess.run",
               return_value=fake_result) as mock_run:
        _dispatch_parallel_workers(
            parallel=args.parallel,
            base_cmd=["python", "tuning/training/unified_early_pipeline.py", "--model", "qwen2-2B"],
            pt_flag="--run-grpo",
            metadata_files=[str(metadata_file)],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )

    cmd = mock_run.call_args[0][0]
    assert "--gres=gpu:4" in cmd
    assert cmd[0] == "sbatch"


def test_dispatch_dpo_does_not_pass_gres(tmp_path):
    """For DPO (or grpo_num_gpus=1), no --gres flag is injected (sbatch script default applies)."""
    from tuning.training.pipeline.orchestrator import _dispatch_parallel_workers

    metadata_file = tmp_path / "meta.jsonl"
    metadata_file.write_text("")

    args = SimpleNamespace(
        post_training_method="dpo",
        grpo_num_gpus=1,
        parallel=2,
    )

    fake_result = MagicMock(returncode=0,
                            stdout="Submitted batch job 12345\n",
                            stderr="")
    with patch("tuning.training.pipeline.orchestrator.subprocess.run",
               return_value=fake_result) as mock_run:
        _dispatch_parallel_workers(
            parallel=args.parallel,
            base_cmd=["python", "tuning/training/unified_early_pipeline.py", "--model", "qwen2-2B"],
            pt_flag="--run-dpo",
            metadata_files=[str(metadata_file)],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )

    cmd = mock_run.call_args[0][0]
    assert not any(a.startswith("--gres") for a in cmd)


def test_grpo_server_subprocess_uses_gpu0_and_injects_server_args(tmp_path, monkeypatch):
    from tuning.training.pipeline import orchestrator as orch

    metadata_file = tmp_path / "meta.jsonl"
    metadata_file.write_text("{}\n")
    captured = {}

    class FakeSidecar:
        def __init__(self, *, model_hf, split, gpu_memory_utilization, startup_timeout, host="127.0.0.1"):
            captured["sidecar"] = {
                "model_hf": model_hf,
                "split": split,
                "gpu_memory_utilization": gpu_memory_utilization,
                "host": host,
                "startup_timeout": startup_timeout,
            }
            self._split = split

        def __enter__(self):
            return SimpleNamespace(
                host="127.0.0.1",
                port=34567,
                group_port=45678,
                split=self._split,
            )

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run(cmd, env=None, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = env
        return MagicMock(returncode=0)

    monkeypatch.setattr(orch, "GRPOVLLMServerSidecar", FakeSidecar)
    monkeypatch.setattr(orch.subprocess, "run", fake_run)

    args = SimpleNamespace(
        model="qwen2-2B",
        grpo_vllm_mode="server",
        grpo_num_gpus=2,
        grpo_vllm_server_gpu_util=0.9,
        grpo_vllm_server_timeout=300.0,
    )

    result = orch._run_post_training_subprocess(
        "grpo",
        ["tuning/training/unified_early_pipeline.py", "--model", "qwen2-2B", "--wandb-project", "test"],
        "--run-grpo",
        str(metadata_file),
        args,
    )

    assert result.returncode == 0
    assert captured["sidecar"]["split"].trainer_gpus == ("0",)
    assert captured["sidecar"]["split"].server_gpus == ("1",)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    cmd = captured["cmd"]
    assert "--grpo-vllm-server-host" in cmd
    assert "127.0.0.1" in cmd
    assert "--grpo-vllm-server-port" in cmd
    assert "34567" in cmd
    assert "--grpo-vllm-group-port" in cmd
    assert "45678" in cmd
