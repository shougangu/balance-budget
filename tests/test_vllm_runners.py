# ABOUTME: Tests for VLLMRunner strategy — selection, fallback, and per-runner behavior.
# ABOUTME: vLLM is mocked; we test the dispatch shape, not real generation.

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from tuning.training.passk.runners import RunnerConfig, VLLMRunner


def test_runner_config_defaults_are_explicit():
    cfg = RunnerConfig(
        base_model_hf="m",
        vllm_gpu_memory_utilization=0.6,
        lora_max_rank=32,
        chat_template="t",
        temperature=0.5,
        max_tokens=256,
        available_gpus=["0"],
        num_inference_gpus=1,
    )
    assert cfg.base_model_hf == "m"
    assert cfg.vllm_gpu_memory_utilization == 0.6


def test_base_runner_is_abstract():
    cfg = RunnerConfig(
        base_model_hf="m", vllm_gpu_memory_utilization=0.6, lora_max_rank=32,
        chat_template="t", temperature=0.5, max_tokens=256,
        available_gpus=["0"], num_inference_gpus=1,
    )
    with __import__("pytest").raises(NotImplementedError):
        VLLMRunner(cfg).run(model=None, eval_strategy=None, adapter_path=None)
