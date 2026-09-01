# ABOUTME: Entry for one verl GRPO worker: claims a banked SFT mark, spends the remainder
# ABOUTME: of the cell budget under BudgetTrackedTrainer, and completes the claimed row.

import argparse
import os
import sys

from tuning.config import ROOT_DIR
from tuning.training.pipeline.checkpoint_metadata import claim_checkpoint, mark_completed

# verl's resumable tree (sharded fp32 weights + optimizer + hf export, ~130 GB per
# save at 8B) sits beside the repo on the same filesystem, never on purged scratch.
DEFAULT_CHECKPOINT_ROOT = os.path.join(os.path.dirname(os.path.dirname(ROOT_DIR)), "verl_ckpts")


def bank_marks(sft_total_minutes, budget_minutes, bank_at):
    """Cumulative GPU-minute marks this worker banks: the budget rows it passes
    on the way to its own budget, plus the budget itself."""
    marks = {float(m) for m in bank_at} | {float(budget_minutes)}
    return sorted(m for m in marks if sft_total_minutes < m <= budget_minutes)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one budget-tracked verl GRPO worker.")
    parser.add_argument("--metadata-file", required=True)
    parser.add_argument("--claim-checkpoint", required=True,
                        help="checkpoint_path of the SFT mark row to claim")
    parser.add_argument("--budget-minutes", type=float, required=True,
                        help="Total cell budget in GPU-minutes (SFT mark + RL share)")
    parser.add_argument("--config", required=True,
                        help="Partial yaml merged over verl's ppo_trainer defaults")
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--bank-at", type=float, nargs="*", default=[],
                        help="Cumulative GPU-minute marks (smaller cell budgets this run "
                             "passes through) that bank an HF checkpoint for offline "
                             "eval (~28GB each at 14B); the budget itself always banks")
    parser.add_argument("--local-ckpt-root", default=DEFAULT_CHECKPOINT_ROOT,
                        help="verl's own (resumable, optimizer-bearing) checkpoint tree")
    parser.add_argument("--models-dir", default=None,
                        help="Where banked HF marks land; defaults to tuning MODELS_DIR")
    return parser.parse_args(argv)


def build_config(args, row):
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    config_dir = os.path.join(os.path.dirname(verl.__file__), "trainer", "config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")
    # reward.reward_kwargs (the DAPO manager's overlong buffer) is read with
    # .get() and is absent from verl's reward schema, so only that node opens
    # for the merge; a misspelled key anywhere else still fails loudly.
    with open_dict(config.reward):
        config = OmegaConf.merge(config, OmegaConf.load(args.config))

    sft_minutes = float(row.get("total_minutes") or 0.0)
    parent = os.path.basename(os.path.normpath(row["checkpoint_path"]))
    experiment = f"{parent}_B{int(args.budget_minutes)}"

    from tuning.config import MODELS_DIR
    with open_dict(config):
        config.trainer.use_v1 = False
        config.trainer.project_name = args.wandb_project
        config.trainer.experiment_name = experiment
        config.trainer.logger = ["console", "wandb"]
        # Eval is offline on banked marks; the loop never validates.
        config.trainer.val_before_train = False
        config.trainer.test_freq = -1
        config.trainer.total_epochs = 10000  # the budget clock terminates the run
        config.trainer.resume_mode = "auto"
        config.trainer.default_local_dir = os.path.join(args.local_ckpt_root, experiment)
        config.actor_rollout_ref.model.path = row["checkpoint_path"]
        # hf_model makes every verl save carry a full HF copy for banking.
        config.actor_rollout_ref.actor.checkpoint.save_contents = [
            "model", "optimizer", "extra", "hf_model",
        ]
        config.budget = OmegaConf.create({
            "budget_minutes": args.budget_minutes,
            "initial_total_minutes": sft_minutes,
            "metadata_path": args.metadata_file,
            "models_dir": args.models_dir or MODELS_DIR,
            "bank_prefix": parent,
            "sft_wandb_run_id": row.get("sft_wandb_run_id", ""),
            "marks": bank_marks(sft_minutes, args.budget_minutes, args.bank_at),
        })
    return config


def main(argv=None):
    args = parse_args(argv)
    row = claim_checkpoint(args.metadata_file, args.claim_checkpoint)
    if row is None:
        print(f"No row for {args.claim_checkpoint} in {args.metadata_file}; nothing to do.")
        sys.exit(42)
    if row.get("completed"):
        print(f"{args.claim_checkpoint} is already completed; nothing to do.")
        return
    if float(row.get("total_minutes") or 0.0) >= args.budget_minutes:
        print("Claimed mark already meets the cell budget; marking completed.")
        mark_completed(args.metadata_file, args.claim_checkpoint)
        return

    config = build_config(args, row)

    from verl.trainer.main_ppo import run_ppo
    from tuning.verl.budget_trainer import BudgetTaskRunner

    run_ppo(config, task_runner_class=BudgetTaskRunner)
    mark_completed(args.metadata_file, args.claim_checkpoint)
    print(f"[run_verl_grpo] completed {args.claim_checkpoint} "
          f"at budget {args.budget_minutes:g}m")


if __name__ == "__main__":
    main()
