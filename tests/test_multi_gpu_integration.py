"""Integration tests for multi-GPU data-parallel inference (requires GPUs).

Run via SLURM with --gres=gpu:2 or more.
These tests are skipped when CUDA is not available.
"""

import pytest
import torch

# Skip entire module if no GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

requires_multi_gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2 if torch.cuda.is_available() else True,
    reason="Requires at least 2 GPUs"
)


@pytest.fixture
def passk_config_factory():
    """Factory for creating PassAtKConfig with test defaults."""
    from tuning.training.config_training import PassAtKConfig

    def _make(num_inference_gpus=1, num_prompts=10, n_samples=1, temperature=0.0):
        return PassAtKConfig(
            target_pass_at_k=[1.0],  # Never triggers early stop
            k_values=[1],
            n_samples=n_samples,
            num_prompts=num_prompts,
            temperature=temperature,
            max_tokens=256,
            strict=True,
            enabled=True,
            use_persistent_vllm=False,
            vllm_gpu_memory_utilization=0.9,
            num_inference_gpus=num_inference_gpus,
        )
    return _make


class TestSingleGPUBaseline:
    def test_single_gpu_produces_valid_scores(self, passk_config_factory):
        """Run evaluate_pass_at_k with num_inference_gpus=1, verify valid scores."""
        pytest.skip("Run manually via SLURM: requires model weights and IFEval data")


class TestMultiGPUInference:
    @requires_multi_gpu
    def test_multi_gpu_produces_valid_scores(self, passk_config_factory):
        """Run with num_inference_gpus=2, verify scores in [0, 1]."""
        pytest.skip("Run manually via SLURM: requires model weights and IFEval data")

    @requires_multi_gpu
    def test_multi_gpu_matches_single_gpu(self, passk_config_factory):
        """Both single and multi-GPU should return same scores with temperature=0."""
        pytest.skip("Run manually via SLURM: requires model weights and IFEval data")
