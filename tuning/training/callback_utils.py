import os
import json
from transformers import TrainerCallback, TrainerState
from transformers.integrations import WandbCallback
from transformers.training_args import TrainingArguments
from tuning.config import MODELS_DIR


class OffsetAwareWandbCallback(WandbCallback):
    """WandbCallback that bridges train/global_step across chained runs.

    Injects train/total_global_step (= global_step + offset) into every log dict.
    Use train/total_global_step as the x-axis in W&B to compare chained runs.
    """

    def __init__(self, initial_global_step=0):
        super().__init__()
        self._offset = int(initial_global_step or 0)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._offset and logs is not None:
            logs["train/total_global_step"] = state.global_step + self._offset
        return super().on_log(args, state, control, model=model, logs=logs, **kwargs)


class CompletionsIntervalCallback(TrainerCallback):
    """Gates trainer.log_completions to fire only every N steps; accumulates all rows into one growing table."""

    def __init__(self, trainer, interval):
        self.trainer = trainer
        self.interval = interval
        self._accumulated_df = None
        self._original_wandb_log = None

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import wandb
            import pandas as pd
        except ImportError:
            return
        if wandb.run is None:
            return

        self._original_wandb_log = wandb.log

        def patched_log(data, *args, **kwargs):
            if isinstance(data, dict) and "completions" in data:
                current_table = data["completions"]
                current_df = pd.DataFrame(current_table.data, columns=current_table.columns)
                self._accumulated_df = (
                    current_df if self._accumulated_df is None
                    else pd.concat([self._accumulated_df, current_df], ignore_index=True)
                )
                data = {**data, "completions": wandb.Table(dataframe=self._accumulated_df)}
            return self._original_wandb_log(data, *args, **kwargs)

        wandb.log = patched_log

    def on_train_end(self, args, state, control, **kwargs):
        if self._original_wandb_log is not None:
            import wandb
            wandb.log = self._original_wandb_log
            self._original_wandb_log = None

    def on_log(self, args, state, control, **kwargs):
        self.trainer.log_completions = (state.global_step % self.interval == 0)


def compute_data_points_seen(state: TrainerState, args: TrainingArguments) -> int:
    """Compute total unique data points seen (accounts for 2 epochs by dividing by 2)."""
    bs = args.per_device_train_batch_size
    ga = args.gradient_accumulation_steps
    ws = getattr(args, "world_size", 1)
    return int(state.global_step * bs * ga * ws)


def save_sweetspot_checkpoint(
    model,
    tokenizer,
    model_name: str,
    threshold_label: str,
    state: TrainerState,
    args: TrainingArguments,
    metadata_path: str,
    extra_metadata: dict = None,
    accelerator=None,
) -> str:
    """Save a sweetspot checkpoint with metadata.

    Args:
        model: The model to save (merged 16bit).
        tokenizer: Tokenizer to save alongside.
        model_name: Base model name for checkpoint naming.
        threshold_label: Label for the threshold (e.g., "ppl-2.50", "pass@1-0.3").
        state: Current TrainerState.
        args: Current TrainingArguments.
        metadata_path: Path to append JSONL metadata to.
        extra_metadata: Additional metadata keys to include.
        accelerator: If provided, unwrap the model and use PEFT save_pretrained instead of unsloth's merged_16bit save.

    Returns:
        Path to the saved checkpoint directory.
    """
    data_points_seen = compute_data_points_seen(state, args)

    checkpoint_name = f"{model_name}_{threshold_label}_sft-{data_points_seen}"
    checkpoint_path = os.path.join(MODELS_DIR, checkpoint_name)

    print(f"[Callback] Saving sweetspot checkpoint to {checkpoint_path}")
    if accelerator is not None:
        target = accelerator.unwrap_model(model)
        target.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    else:
        model.save_pretrained_merged(checkpoint_path, tokenizer, save_method="merged_16bit")

    with open(f"{checkpoint_path}/training_config.json", "w") as f:
        json.dump(args.to_dict(), f, indent=4)

    metadata = {
        "global_step": state.global_step,
        "checkpoint_path": checkpoint_path,
        "data_points_seen": data_points_seen,
        **(extra_metadata or {}),
    }
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata) + "\n")

    print(f"[Callback] Sweetspot checkpoint saved with metadata at {metadata_path}")
    return checkpoint_path
