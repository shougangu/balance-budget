# ABOUTME: Tests optimizer-step GRPO timing aggregation for W&B metrics.

import math

import pytest

from tuning.training.grpo_timing import GRPOStepTiming


def test_grpo_step_timing_builds_phase_breakdown():
    timing = GRPOStepTiming()
    timing.add("step", 40.0)
    timing.add("rollout", 15.0)
    timing.add("vllm_generate", 7.0)
    timing.add("weight_sync", 0.2)
    timing.add("rollout_logps", 6.0)
    timing.add("reward", 0.1)
    timing.add("policy_forward_loss", 8.0)
    timing.add("backward", 15.0)
    timing.add("generated_tokens", 7000)

    metrics = timing.finish()

    assert metrics["timing/rollout_overhead_seconds"] == pytest.approx(1.7)
    assert metrics["timing/unattributed_seconds"] == pytest.approx(2.0)
    assert metrics["timing/vllm_tokens_per_second"] == pytest.approx(1000.0)
    assert metrics["timing/rollout_fraction"] == pytest.approx(0.375)
    assert metrics["timing/vllm_fraction"] == pytest.approx(0.175)
    assert metrics["timing/policy_forward_loss_fraction"] == pytest.approx(0.2)
    assert metrics["timing/backward_fraction"] == pytest.approx(0.375)


def test_grpo_step_timing_accumulates_and_resets():
    timing = GRPOStepTiming()
    timing.add("backward", 1.25)
    timing.add("backward", 2.75)

    assert timing.finish()["timing/backward_seconds"] == pytest.approx(4.0)
    assert timing.finish()["timing/backward_seconds"] == 0.0


def test_grpo_step_timing_ignores_invalid_values():
    timing = GRPOStepTiming()
    timing.add("step", -1)
    timing.add("step", math.nan)
    timing.add("step", "invalid")

    assert timing.finish()["timing/step_seconds"] == 0.0
