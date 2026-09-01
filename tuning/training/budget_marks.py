# ABOUTME: Trainer callback that banks checkpoints at GPU-minute budget marks with no in-loop eval.
# ABOUTME: Each crossing saves an HF checkpoint (FSDP-gathered, bf16) plus one metadata JSONL row.

import datetime
import os

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from tuning.config import MODELS_METADATA_DIR
from tuning.training.callback_utils import (
    get_total_minutes_from_state,
    save_sweetspot_checkpoint,
)
from tuning.training.passk.decisions import CheckpointDecisionEngine


class BudgetMarkCallback(TrainerCallback):
    """Watch train/total_minutes and bank a checkpoint at each budget mark.

    Evaluation happens offline on the banked checkpoints, so this callback never
    pauses training beyond the save itself.
    """

    def __init__(self, model_name, tokenizer, target_total_minutes,
                 eval_only_minutes=None, metadata_path=None):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.metadata_path = metadata_path
        self._engine = CheckpointDecisionEngine(
            target_thresholds=[],
            early_tuples=None,
            max_checkpoint_gap=None,
            target_total_minutes=list(target_total_minutes),
            eval_only_minutes=eval_only_minutes,
        )
        self._trainer = None

    def set_trainer(self, trainer):
        """Attach the built trainer so FSDP saves can gather through its accelerator."""
        self._trainer = trainer

    def _accelerator(self):
        return getattr(self._trainer, "accelerator", None) if self._trainer else None

    def _is_rank_zero(self):
        return not dist.is_initialized() or dist.get_rank() == 0

    def _synchronized_total_minutes(self, state):
        """Return rank 0's total_minutes on every rank.

        Per-rank wall clocks drift; a mark must cross on every rank in the same
        step or the ranks issue divergent collectives during the save and hang.
        """
        total_minutes = get_total_minutes_from_state(state)
        if dist.is_initialized() and dist.get_world_size() > 1:
            payload = [total_minutes]
            dist.broadcast_object_list(payload, src=0)
            total_minutes = payload[0]
        return total_minutes

    def on_train_begin(self, args, state, control, **kwargs):
        if self.metadata_path is None:
            now = datetime.datetime.now().strftime("%m%d_%H%M%S")
            self.metadata_path = os.path.join(
                MODELS_METADATA_DIR, f"{self.model_name}_budget-marks-{now}.json"
            )
        print(f"METADATA_FILE:{self.metadata_path}")

        starting_minutes = get_total_minutes_from_state(state)
        if starting_minutes > 0 and self._engine.target_total_minutes:
            self._engine.target_total_minutes = [
                t for t in self._engine.target_total_minutes if t > starting_minutes
            ]
            print(
                f"[BudgetMarks] resumed at {starting_minutes:.1f}m; "
                f"remaining marks: {self._engine.target_total_minutes}"
            )

    def on_step_end(self, args, state, control, **kwargs):
        total_minutes = self._synchronized_total_minutes(state)
        decisions = self._engine.decide(
            primary_metric=0.0,
            history=[],
            data_points_seen=0,
            last_checkpoint_data_points=0,
            total_minutes=total_minutes,
        )
        for decision in decisions:
            print(
                f"[BudgetMarks] {decision.metadata_value:g}m mark crossed "
                f"(actual {total_minutes:.2f}m); banking checkpoint"
            )
            self._bank(decision, state, args, kwargs.get("model"))
        return control

    def _gathered_state_dict(self, model):
        """Collect the full weights across FSDP ranks, cast to bf16 for disk.

        Collective: every rank must enter. Returns None without an accelerator
        (plain single-GPU), letting save_pretrained read the model directly.
        """
        accelerator = self._accelerator()
        if accelerator is None:
            return None
        state_dict = accelerator.get_state_dict(model)
        if not self._is_rank_zero() or state_dict is None:
            return state_dict
        return {
            key: value.to(torch.bfloat16) if value.is_floating_point() else value
            for key, value in state_dict.items()
        }

    def _bank(self, decision, state, args, model):
        state_dict = self._gathered_state_dict(model)
        accelerator = self._accelerator()
        if self._is_rank_zero():
            extra_metadata = {
                "threshold_type": decision.metadata_type,
                "threshold_value": decision.metadata_value,
            }
            if decision.eval_only:
                # Pre-claim so claim_next_checkpoint skips it: the 100%-budget
                # anchors are evaluated but never seed an RL worker.
                extra_metadata["eval_only"] = True
                extra_metadata["claimed"] = True
            save_sweetspot_checkpoint(
                model=model,
                tokenizer=self.tokenizer,
                model_name=self.model_name,
                threshold_label=f"budget-{decision.label}",
                state=state,
                args=args,
                metadata_path=self.metadata_path,
                extra_metadata=extra_metadata,
                accelerator=accelerator,
                state_dict=state_dict,
            )
        if accelerator is not None:
            accelerator.wait_for_everyone()
