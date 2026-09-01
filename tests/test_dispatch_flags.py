# ABOUTME: Tests computed live-dispatch allocations and verl worker submission.
# ABOUTME: Allocations are sized from the remaining budget and the worker's GPU count.

from types import SimpleNamespace
from unittest.mock import patch

import tuning.training.pipeline.orchestrator as orch


class TestDispatchFlags:
    def test_gpu_type_selects_the_matching_partition_family_and_gres(self):
        flags, _ = orch._dispatch_flags(remaining_gpu_minutes=100, num_gpus=2, gpu_type="l40s")
        assert flags[0].startswith("--partition=gpubase_l40s_b1,")
        assert "h100" not in flags[0]

    def test_small_remaining_budget_reaches_the_shallow_tiers(self):
        flags, wall = orch._dispatch_flags(remaining_gpu_minutes=960, num_gpus=8)
        partition = flags[0]
        assert partition.startswith("--partition=")
        assert "gpubase_h100_b1" in partition
        assert wall <= 3 * 60

    def test_large_remaining_budget_uses_only_deep_tiers(self):
        # 46080 GPU-min on 8 GPUs ~ 4 days of wall: only b5 fits.
        flags, wall = orch._dispatch_flags(remaining_gpu_minutes=46080, num_gpus=8)
        assert flags[0] == "--partition=gpubase_h100_b5"
        assert wall > 3 * 24 * 60

    def test_wall_time_is_capped_at_the_deepest_tier(self):
        flags, wall = orch._dispatch_flags(remaining_gpu_minutes=10_000_000, num_gpus=8)
        assert flags[0] == "--partition=gpubase_h100_b5"
        assert wall == 7 * 24 * 60
        assert flags[1] == "--time=7-00:00:00"

    def test_more_gpus_shrink_the_wall(self):
        _, wall8 = orch._dispatch_flags(remaining_gpu_minutes=7680, num_gpus=8)
        _, wall2 = orch._dispatch_flags(remaining_gpu_minutes=7680, num_gpus=2)
        assert wall8 < wall2

    def test_time_format_days_hours_minutes(self):
        flags, wall = orch._dispatch_flags(remaining_gpu_minutes=7680, num_gpus=8)
        assert flags[1].startswith("--time=")
        # 7680/(8*0.85)+30 ~ 1160 min ~ 19h20m, zero-padded H:MM
        assert flags[1] == f"--time={wall // 1440}-{(wall % 1440) // 60:02d}:{wall % 60:02d}:00"


def _args(**overrides):
    defaults = dict(
        wandb_project="longcot",
        qos=None,
        verl_num_gpus=8,
        verl_gpu_type="h100",
        verl_config="tuning/verl/configs/qwen3_14b_grpo.yaml",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestSubmitVerlWorker:
    def test_submits_with_sized_allocation_and_typed_gres(self):
        with patch.object(orch, "_submit_sbatch_worker", return_value="123") as submit:
            job = orch.submit_verl_worker_for_metadata(
                _args(), "meta.json", "/ckpt/a",
                budget_minutes=30720, sft_total_minutes=7680, bank_at=[15360],
            )
        assert job == "123"
        script, worker_argv = submit.call_args.args[:2]
        assert script == orch.VERL_SBATCH_SCRIPT
        assert worker_argv[worker_argv.index("--claim-checkpoint") + 1] == "/ckpt/a"
        assert worker_argv[worker_argv.index("--budget-minutes") + 1] == "30720"
        assert worker_argv[worker_argv.index("--bank-at") + 1] == "15360"
        flags = submit.call_args.kwargs["sbatch_flags"]
        assert "--gres=gpu:h100:8" in flags
        assert any(f.startswith("--partition=gpubase_h100") for f in flags)

    def test_gpu_type_flows_into_gres_and_partitions(self):
        with patch.object(orch, "_submit_sbatch_worker", return_value="7") as submit:
            orch.submit_verl_worker_for_metadata(
                _args(verl_num_gpus=2, verl_gpu_type="l40s"), "meta.json", "/ckpt/a",
                budget_minutes=16, sft_total_minutes=4, bank_at=[8],
            )
        flags = submit.call_args.kwargs["sbatch_flags"]
        assert "--gres=gpu:l40s:2" in flags
        assert any(f.startswith("--partition=gpubase_l40s") for f in flags)

    def test_sbatch_failure_is_swallowed(self):
        with patch.object(orch, "_submit_sbatch_worker", side_effect=SystemExit("boom")):
            job = orch.submit_verl_worker_for_metadata(
                _args(), "meta.json", "/ckpt/a",
                budget_minutes=15360, sft_total_minutes=7680, bank_at=[],
            )
        assert job is None
