import math
from collections import defaultdict


class GRPOStepTiming:
    """Accumulate disjoint GRPO phase timings for one optimizer step."""

    def __init__(self):
        self._totals = defaultdict(float)

    def add(self, name: str, value: float) -> None:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(value) and value >= 0:
            self._totals[name] += value

    def finish(self) -> dict[str, float]:
        totals = dict(self._totals)
        self._totals.clear()

        step = totals.get("step", 0.0)
        rollout = totals.get("rollout", 0.0)
        vllm = totals.get("vllm_generate", 0.0)
        sync = totals.get("weight_sync", 0.0)
        rollout_logps = totals.get("rollout_logps", 0.0)
        reward = totals.get("reward", 0.0)
        policy_forward = totals.get("policy_forward_loss", 0.0)
        backward = totals.get("backward", 0.0)
        generated_tokens = totals.get("generated_tokens", 0.0)

        rollout_overhead = max(
            rollout - vllm - sync - rollout_logps - reward,
            0.0,
        )
        unattributed = max(
            step - rollout - policy_forward - backward,
            0.0,
        )

        metrics = {
            "timing/step_seconds": step,
            "timing/rollout_seconds": rollout,
            "timing/vllm_generate_seconds": vllm,
            "timing/weight_sync_seconds": sync,
            "timing/rollout_logps_seconds": rollout_logps,
            "timing/reward_seconds": reward,
            "timing/rollout_overhead_seconds": rollout_overhead,
            "timing/policy_forward_loss_seconds": policy_forward,
            "timing/backward_seconds": backward,
            "timing/unattributed_seconds": unattributed,
            "timing/generated_tokens": generated_tokens,
        }
        if step > 0:
            metrics.update({
                "timing/rollout_fraction": rollout / step,
                "timing/vllm_fraction": vllm / step,
                "timing/policy_forward_loss_fraction": policy_forward / step,
                "timing/backward_fraction": backward / step,
            })
        if vllm > 0:
            metrics["timing/vllm_tokens_per_second"] = generated_tokens / vllm
        return metrics
