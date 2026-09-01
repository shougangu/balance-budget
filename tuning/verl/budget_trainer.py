# ABOUTME: verl RayPPOTrainer variant that spends a GPU-minute budget: ticks a wall x nGPU
# ABOUTME: clock once per step, banks HF checkpoints + metadata rows at marks, stops at budget.

import json
import os
import shutil
import time

import ray

# verl 0.9.0 (pinned; see SETUP.md). The v0 stack is deprecated upstream but is the
# battle-tested synchronous colocate path; revisit on any verl upgrade.
from verl.trainer import main_ppo_v0
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from tuning.training.passk.decisions import CheckpointDecisionEngine
from tuning.training.pipeline.checkpoint_metadata import append_metadata_row
from tuning.verl.export import carry_parent_config

CLOCK_FILENAME = "budget_clock.json"


class BudgetTrackedTrainer(RayPPOTrainer):
    """RayPPOTrainer whose stopping condition is a GPU-minute budget, not a step count.

    The clock advances by (wall seconds since the previous tick) x nGPU, seeded with
    the claimed SFT mark's total_minutes, so train/total_minutes is the cumulative
    cell budget exactly as on the SFT side. Hooks are deliberately narrow:
    _get_gen_batch runs once per step at step start and decides; _save_checkpoint is
    verl's own end-of-step save, which is where a decided mark actually banks so the
    banked weights, dataloader state, and step label all agree; total_training_steps
    controls is_last_step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        budget = self.config.budget
        self._budget_minutes = float(budget.budget_minutes)
        self._total_seconds = float(budget.initial_total_minutes) * 60.0
        self._metadata_path = budget.metadata_path
        self._models_dir = budget.models_dir
        self._bank_prefix = budget.bank_prefix
        self._sft_wandb_run_id = budget.get("sft_wandb_run_id", "")
        self._engine = CheckpointDecisionEngine(
            target_thresholds=[],
            early_tuples=None,
            max_checkpoint_gap=None,
            target_total_minutes=list(budget.marks),
        )
        self._tick = None
        self._pending_banks = []
        self._native_save_freq = self.config.trainer.save_freq
        assert self._native_save_freq > 0, (
            "budget banking rides on verl's end-of-step save; set trainer.save_freq > 0"
        )

    # -- clock ---------------------------------------------------------------

    def _n_gpus(self):
        return self.resource_pool_manager.get_n_gpus()

    def total_minutes(self):
        return self._total_seconds / 60.0

    def _advance_clock(self):
        now = time.perf_counter()
        if self._tick is not None:
            self._total_seconds += (now - self._tick) * self._n_gpus()
        self._tick = now

    def _get_gen_batch(self, batch):
        self._budget_tick()
        return super()._get_gen_batch(batch)

    def _budget_tick(self):
        if self._tick is None:
            # First step (fresh or resumed): never re-fire marks already passed.
            self._engine.target_total_minutes = [
                t for t in self._engine.target_total_minutes if t > self.total_minutes()
            ]
        self._advance_clock()

        total_minutes = self.total_minutes()
        self._log_total_minutes(total_minutes)

        decisions = self._engine.decide(
            primary_metric=0.0, history=[], data_points_seen=0,
            last_checkpoint_data_points=0, total_minutes=total_minutes,
        )
        if decisions:
            for decision in decisions:
                print(f"[Budget] {decision.metadata_value:g}m mark crossed "
                      f"(actual {total_minutes:.2f}m); banking at this step's save")
            self._pending_banks.extend(decisions)
            # Pull verl's periodic save onto this step; _save_checkpoint restores it.
            self.config.trainer.save_freq = 1

        if total_minutes >= self._budget_minutes:
            # Make the step now starting the last one; fit() saves and returns.
            print(f"[Budget] budget {self._budget_minutes:g}m reached at "
                  f"{total_minutes:.2f}m; stopping after this step")
            self.total_training_steps = min(self.total_training_steps, self.global_steps)

    def _log_total_minutes(self, total_minutes):
        import wandb
        if wandb.run is not None:
            wandb.log({"train/total_minutes": total_minutes},
                      step=self.global_steps, commit=False)

    # -- banking on verl's own save -------------------------------------------

    def _global_step_dir(self):
        return os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}",
        )

    def _save_checkpoint(self):
        # Fold the step just finished into the clock so the persisted clock and any
        # banked row describe the weights being saved.
        self._advance_clock()
        super()._save_checkpoint()
        with open(os.path.join(self._global_step_dir(), CLOCK_FILENAME), "w") as fh:
            json.dump({"total_seconds": self._total_seconds}, fh)
        for decision in self._pending_banks:
            self._bank(decision)
        self._pending_banks = []
        self.config.trainer.save_freq = self._native_save_freq

    def _bank(self, decision):
        source = os.path.join(self._global_step_dir(), "actor", "huggingface")
        import wandb
        run_id = wandb.run.id if wandb.run is not None else ""
        name = f"{self._bank_prefix}_rl-{decision.label}_step-{self.global_steps}"
        if run_id:
            name = f"{name}_{run_id}"
        destination = os.path.join(self._models_dir, name)
        # Banked marks live on /project; verl's own checkpoint tree stays on scratch.
        shutil.copytree(source, destination, dirs_exist_ok=True)
        carry_parent_config(self.config.actor_rollout_ref.model.path, destination)
        append_metadata_row(self._metadata_path, {
            "global_step": self.global_steps,
            "checkpoint_path": destination,
            "total_minutes": self.total_minutes(),
            "threshold_type": decision.metadata_type,
            "threshold_value": decision.metadata_value,
            # RL checkpoints are evaluated offline but never seed another worker.
            "eval_only": True,
            "claimed": True,
            "wandb_run_id": run_id,
            "rl_wandb_run_id": run_id,
            "sft_wandb_run_id": self._sft_wandb_run_id,
        })
        print(f"[Budget] banked {destination}")

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps <= 0:
            return
        clock_path = os.path.join(self._global_step_dir(), CLOCK_FILENAME)
        if os.path.isfile(clock_path):
            with open(clock_path) as fh:
                self._total_seconds = float(json.load(fh)["total_seconds"])
            print(f"[Budget] resumed clock at {self.total_minutes():.2f}m "
                  f"(step {self.global_steps})")


class _BudgetTaskRunner(main_ppo_v0.BaseTaskRunner):
    """Upstream TaskRunner.run with BudgetTrackedTrainer in place of RayPPOTrainer.

    Not a subclass of the upstream TaskRunner: ray.remote copies every method into
    its actor wrapper and ships the wrapper to the actor process by value, and the
    copied __init__ then carries a by-value clone of TaskRunner in its super()
    closure that is not the class in the actor's MRO. Instead run() resolves the
    upstream function inside the actor, where main_ppo_v0 is the real module, and
    rebinds the module-level RayPPOTrainer name that the upstream body constructs.
    """

    def run(self, config):
        upstream = main_ppo_v0.TaskRunner.__ray_actor_class__.run
        assert main_ppo_v0.RayPPOTrainer is RayPPOTrainer, (
            "verl no longer builds RayPPOTrainer by that name in main_ppo_v0; "
            "re-diff TaskRunner.run against the pinned release"
        )
        main_ppo_v0.RayPPOTrainer = BudgetTrackedTrainer
        return upstream(self, config)


BudgetTaskRunner = ray.remote(_BudgetTaskRunner)
