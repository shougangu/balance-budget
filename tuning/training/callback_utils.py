import json
import math
import os
import time
import warnings

import wandb
from transformers import TrainerCallback, TrainerState
from transformers.integrations import WandbCallback
from transformers.trainer_callback import ExportableState
from transformers.training_args import TrainingArguments
from tuning.config import MODELS_DIR


TRAINER_STATE_FILENAME = "trainer_state.json"
_CALLBACK_STATE_NAMES = ("OffsetAwareWandbCallback", "TrainingTimeCallback")


def _valid_seconds(value):
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _read_timing_state(stateful_callbacks):
    """Return (stored name, total seconds, step offset) across supported states."""
    for name in _CALLBACK_STATE_NAMES:
        callback_state = (stateful_callbacks or {}).get(name)
        if isinstance(callback_state, list):
            callback_state = callback_state[-1] if callback_state else None
        if not isinstance(callback_state, dict):
            continue

        attributes = callback_state.get("attributes", {})
        total_seconds = _valid_seconds(attributes.get("total_seconds"))
        if total_seconds is None:
            total_seconds = _valid_seconds(attributes.get("_cumulative_train_seconds"))

        try:
            step_offset = int(
                callback_state.get("args", {}).get("initial_global_step", 0) or 0
            )
        except (TypeError, ValueError):
            step_offset = None
        return name, total_seconds, step_offset
    return None, None, None


class OffsetAwareWandbCallback(WandbCallback, ExportableState):
    """Log chained-run step offsets and completed training-step time to W&B."""

    def __init__(self, initial_global_step=0, initial_total_seconds=0.0, time_multiplier=1.0):
        super().__init__()
        self.step_offset = int(initial_global_step or 0)
        self.total_seconds = _valid_seconds(initial_total_seconds) or 0.0
        self.time_multiplier = time_multiplier
        self.step_start = None

    def state(self):
        return {
            "args": {"initial_global_step": self.step_offset},
            "attributes": {"total_seconds": self.total_seconds},
        }

    def _sync_state(self, state):
        state.stateful_callbacks[self.__class__.__name__] = self.state()

    def setup(self, args, state, model, **kwargs):
        super().setup(args, state, model, **kwargs)
        if (
            self._wandb is not None
            and self.step_offset
            and state.is_world_process_zero
        ):
            self._wandb.define_metric("train/total_global_step")
            self._wandb.define_metric(
                "*",
                step_metric="train/total_global_step",
                step_sync=True,
            )

    def on_train_begin(self, args, state, control, **kwargs):
        stored_name, total_seconds, step_offset = _read_timing_state(
            state.stateful_callbacks,
        )
        if total_seconds is not None:
            self.total_seconds = total_seconds
        elif state.global_step > 0:
            warnings.warn(
                "No valid OffsetAwareWandbCallback timing state found in resumed "
                "TrainerState; starting train/total_minutes at 0.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.total_seconds = 0.0
        if step_offset is not None:
            self.step_offset = step_offset
        if stored_name and stored_name != self.__class__.__name__:
            state.stateful_callbacks.pop(stored_name, None)

        # Trainer rebuilds this callback as type(cb)(**state()["args"]) when it
        # resumes, so the constructor's multiplier is gone by now; take it from
        # the training arguments, which the launch config rebuilds every run.
        multiplier = getattr(args, "gpu_minute_multiplier", None)
        if multiplier:
            self.time_multiplier = float(multiplier)

        self.step_start = None
        self._sync_state(state)
        return super().on_train_begin(args, state, control, **kwargs)

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.perf_counter()
        return super().on_step_begin(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start is not None:
            self.total_seconds += max(time.perf_counter() - self.step_start, 0.0) * self.time_multiplier
            self.step_start = None
            self._sync_state(state)
        return super().on_step_end(args, state, control, **kwargs)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs is not None:
            logs["total_global_step"] = state.global_step + self.step_offset
            logs["total_minutes"] = self.total_seconds / 60.0
            self._sync_state(state)
        return super().on_log(args, state, control, model=model, logs=logs, **kwargs)


def remove_default_wandb_callback(trainer) -> None:
    for callback in list(trainer.callback_handler.callbacks):
        if type(callback) is WandbCallback:
            trainer.remove_callback(callback)
            return


def get_total_seconds_from_state(state: TrainerState) -> float:
    _, total_seconds, _ = _read_timing_state(
        getattr(state, "stateful_callbacks", None),
    )
    return total_seconds or 0.0


def get_total_minutes_from_state(state: TrainerState) -> float:
    return get_total_seconds_from_state(state) / 60.0


def load_total_seconds_from_checkpoint(checkpoint_path: str, warn: bool = True) -> float:
    state_path = os.path.join(checkpoint_path, TRAINER_STATE_FILENAME)
    try:
        state = TrainerState.load_from_json(state_path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if warn:
            warnings.warn(
                f"Could not load {state_path}: {exc}; "
                "starting train/total_minutes at 0.",
                RuntimeWarning,
                stacklevel=2,
            )
        return 0.0

    stored_name, total_seconds, _ = _read_timing_state(state.stateful_callbacks)
    if total_seconds is not None:
        return total_seconds
    if warn:
        reason = "No timing callback state" if stored_name is None else "No valid total_seconds"
        warnings.warn(
            f"{reason} found in {state_path}; starting train/total_minutes at 0.",
            RuntimeWarning,
            stacklevel=2,
        )
    return 0.0


class CompletionsIntervalCallback(TrainerCallback):
    """Gates trainer.log_completions to fire only every N steps; accumulates all rows into one growing table."""

    def __init__(self, trainer, interval):
        self.trainer = trainer
        self.interval = interval
    def on_log(self, args, state, control, **kwargs):
        self.trainer.log_completions = (state.global_step % self.interval == 0)


def compute_data_points_seen(state: TrainerState, args: TrainingArguments) -> int:
    """Compute total unique data points seen (accounts for 2 epochs by dividing by 2)."""
    bs = args.per_device_train_batch_size
    ga = args.gradient_accumulation_steps
    ws = getattr(args, "world_size", 1)
    return int(state.global_step * bs * ga * ws)


def save_trainer_state(state: TrainerState, output_dir: str) -> None:
    if not isinstance(state, TrainerState):
        raise TypeError("state must be a transformers.TrainerState")
    os.makedirs(output_dir, exist_ok=True)
    state.save_to_json(os.path.join(output_dir, TRAINER_STATE_FILENAME))


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
        model: The model whose LoRA adapter to save.
        tokenizer: Tokenizer to save alongside.
        model_name: Base model name for checkpoint naming.
        threshold_label: Label for the threshold (e.g., "ppl-2.50", "pass@1-0.3").
        state: Current TrainerState.
        args: Current TrainingArguments.
        metadata_path: Path to append JSONL metadata to.
        extra_metadata: Additional metadata keys to include.
        accelerator: If provided, unwrap the model before saving the adapter (DDP).

    Returns:
        Path to the saved checkpoint directory.
    """
    data_points_seen = compute_data_points_seen(state, args)
    wandb_run_id = wandb.run.id if wandb.run else ""

    checkpoint_name = f"{model_name}_{threshold_label}_sft-{data_points_seen}"
    if wandb_run_id:
        checkpoint_name = f"{checkpoint_name}_{wandb_run_id}"
    checkpoint_path = os.path.join(MODELS_DIR, checkpoint_name)

    print(f"[Callback] Saving sweetspot checkpoint to {checkpoint_path}")
    target = accelerator.unwrap_model(model) if accelerator is not None else model
    if hasattr(target, 'save_pretrained'):
        target.save_pretrained(checkpoint_path)
    else:
        target.save_pretrained_merged(checkpoint_path, tokenizer, save_method="lora")
    tokenizer.save_pretrained(checkpoint_path)

    os.makedirs(checkpoint_path, exist_ok=True)
    with open(f"{checkpoint_path}/training_config.json", "w") as f:
        json.dump(args.to_dict(), f, indent=4)
    save_trainer_state(state, checkpoint_path)

    metadata = {
        "global_step": state.global_step,
        "checkpoint_path": checkpoint_path,
        "data_points_seen": data_points_seen,
        "total_minutes": get_total_minutes_from_state(state),
        **(extra_metadata or {}),
        "wandb_run_id": wandb_run_id,
        "sft_wandb_run_id": wandb_run_id,
    }
    with open(metadata_path, "a") as f:
        f.write(json.dumps(metadata) + "\n")

    print(f"[Callback] Sweetspot checkpoint saved with metadata at {metadata_path}")
    return checkpoint_path
