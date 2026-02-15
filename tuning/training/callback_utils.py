import os
import json
from transformers import TrainerState
from transformers.training_args import TrainingArguments
from tuning.config import MODELS_DIR


def compute_data_points_seen(state: TrainerState, args: TrainingArguments) -> int:
    """Compute total unique data points seen (accounts for 2 epochs by dividing by 2)."""
    bs = args.per_device_train_batch_size
    ga = args.gradient_accumulation_steps
    ws = getattr(args, "world_size", 1)
    return int(state.global_step * bs * ga * ws / 2)


def save_sweetspot_checkpoint(
    model,
    tokenizer,
    model_name: str,
    threshold_label: str,
    state: TrainerState,
    args: TrainingArguments,
    metadata_path: str,
    extra_metadata: dict = None,
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

    Returns:
        Path to the saved checkpoint directory.
    """
    data_points_seen = compute_data_points_seen(state, args)

    checkpoint_name = f"{model_name}_{threshold_label}_sft-{data_points_seen}"
    checkpoint_path = os.path.join(MODELS_DIR, checkpoint_name)

    print(f"[Callback] Saving sweetspot checkpoint to {checkpoint_path}")
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
