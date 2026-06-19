# ABOUTME: GRPO server-mode vLLM sidecar launch helpers for local orchestrator runs.
# ABOUTME: Keeps GPU split validation and server lifecycle outside orchestrator control flow.

from __future__ import annotations

import os
import socket
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


DEFAULT_VLLM_MAX_MODEL_LEN = 6144


@dataclass(frozen=True)
class GRPOServerSplit:
    trainer_gpus: tuple[str, ...]
    server_gpus: tuple[str, ...]


def resolve_grpo_server_split(grpo_num_gpus: int) -> GRPOServerSplit:
    """Use GPU 0 for training and every remaining GPU as a data-parallel vLLM worker."""
    if grpo_num_gpus < 2:
        raise ValueError(
            f"--grpo-vllm-mode=server requires --grpo-num-gpus >= 2 "
            f"(got {grpo_num_gpus})"
        )
    return GRPOServerSplit(
        trainer_gpus=("0",),
        server_gpus=tuple(str(i) for i in range(1, grpo_num_gpus)),
    )


def reserve_tcp_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for an available TCP port and release it for a child process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


@dataclass(frozen=True)
class RunningVLLMServer:
    host: str
    port: int
    group_port: int
    log_path: Path


class GRPOVLLMServerSidecar:
    """Context manager that starts `trl vllm-serve` and stops it on exit."""

    def __init__(
        self,
        *,
        model_hf: str,
        split: GRPOServerSplit,
        gpu_memory_utilization: float,
        max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN,
        host: str = "127.0.0.1",
        dtype: str | None = None,
        startup_timeout: float = 300.0,
        popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self.model_hf = model_hf
        self.split = split
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.host = host
        self.dtype = dtype
        self.startup_timeout = startup_timeout
        self._popen_factory = popen_factory
        self._proc: subprocess.Popen | None = None
        self._log_handle = None
        self._running: RunningVLLMServer | None = None

    @property
    def running(self) -> RunningVLLMServer:
        if self._running is None:
            raise RuntimeError("vLLM sidecar is not running")
        return self._running

    def __enter__(self) -> RunningVLLMServer:
        port = reserve_tcp_port(self.host)
        group_port = reserve_tcp_port(self.host)
        while group_port == port:
            group_port = reserve_tcp_port(self.host)
        log_path = Path(f"vllm-serve-{os.getpid()}-{port}.out")
        self._log_handle = log_path.open("w")

        data_parallel_size = len(self.split.server_gpus)
        cmd = [
            "trl", "vllm-serve",
            "--model", self.model_hf,
            "--host", self.host,
            "--port", str(port),
            "--data-parallel-size", str(data_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--max-model-len", str(self.max_model_len),
        ]
        if self.dtype is not None:
            cmd += ["--dtype", self.dtype]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(self.split.server_gpus)

        print(
            "[orchestrator] Starting GRPO vLLM server: "
            f"gpus={env['CUDA_VISIBLE_DEVICES']} "
            f"dp={data_parallel_size} max_model_len={self.max_model_len} "
            f"dtype={self.dtype or 'auto'} "
            f"http={self.host}:{port} group_port={group_port} "
            f"log={log_path}"
        )
        try:
            self._proc = self._popen_factory(
                cmd,
                env=env,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
            self._running = RunningVLLMServer(
                host=self.host,
                port=port,
                group_port=group_port,
                log_path=log_path,
            )
            self._wait_for_health()
            return self._running
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def _wait_for_health(self) -> None:
        url = f"http://{self.running.host}:{self.running.port}/health/"
        deadline = time.monotonic() + self.startup_timeout
        while True:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    "vllm-serve exited before becoming healthy "
                    f"(code {self._proc.returncode}); see {self.running.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        print(f"[orchestrator] vLLM server healthy at {url}")
                        return
            except Exception:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"vllm-serve did not become healthy within "
                        f"{self.startup_timeout} seconds; see {self.running.log_path}"
                    )
                time.sleep(2)

    def close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            print(f"[orchestrator] Terminating vLLM server pid={proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        self._proc = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
