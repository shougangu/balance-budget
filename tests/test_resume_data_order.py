# ABOUTME: Tests that an SFT resume continues the current epoch's sample order instead of
# ABOUTME: replaying epoch 0's, which accelerate's batch-skipping dataloader otherwise does.

import shutil

import pytest
import torch
from datasets import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from tuning.training.model_utils import install_resume_epoch_patch

N, BATCH, ACCUM, EPOCHS = 24, 2, 2, 3   # 12 micro-batches, 6 optimizer steps per epoch
STOP_STEP = 8                           # epoch 1, two optimizer steps (4 micro-batches) in


class _StopAt(TrainerCallback):
    def __init__(self, step):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            control.should_training_stop = True
        return control


def _train(outdir, seen, resume=None, stop=None):
    dataset = Dataset.from_dict({
        "idx": list(range(N)),
        "input_ids": [[(i * 7 + j) % 97 for j in range(8)] for i in range(N)],
    })

    def collate(features):
        seen.append([f["idx"] for f in features])
        ids = torch.tensor([f["input_ids"] for f in features])
        return {"input_ids": ids, "labels": ids}

    set_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=100, n_positions=16, n_embd=16, n_layer=1, n_head=2))
    args = TrainingArguments(
        output_dir=str(outdir), per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM, num_train_epochs=EPOCHS, seed=42, data_seed=3407,
        save_strategy="steps", save_steps=stop or 10**6, report_to=[], logging_steps=1,
        use_cpu=True, dataloader_num_workers=0, disable_tqdm=True, lr_scheduler_type="constant",
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset, data_collator=collate,
        callbacks=[_StopAt(stop)] if stop else [],
    )
    trainer.train(resume_from_checkpoint=resume)


def test_resume_continues_the_current_epoch_order(tmp_path):
    assert install_resume_epoch_patch() is True
    full, resumed = [], []
    _train(tmp_path / "full", full)
    _train(tmp_path / "stopped", [], stop=STOP_STEP)
    _train(tmp_path / "resumed", resumed, resume=str(tmp_path / "stopped" / f"checkpoint-{STOP_STEP}"))
    expected = full[STOP_STEP * ACCUM:]
    assert resumed == expected


def test_patch_installs_once():
    install_resume_epoch_patch()
    assert install_resume_epoch_patch() is False


def test_patch_reaches_a_training_loop_executed_in_another_module(monkeypatch):
    """Unsloth re-execs the loop with a copy of transformers.trainer's names."""
    import types

    import accelerate
    import transformers.trainer as trainer_module

    monkeypatch.setattr(trainer_module, "_balance_budget_resume_epoch", False, raising=False)
    monkeypatch.setattr(trainer_module, "skip_first_batches", accelerate.skip_first_batches)
    copied = types.ModuleType("fake_unsloth_compiler")
    copied.skip_first_batches = accelerate.skip_first_batches
    exec("def _fast_inner_training_loop(self):\n    return skip_first_batches", copied.__dict__)
    monkeypatch.setattr(trainer_module.Trainer, "_inner_training_loop", copied._fast_inner_training_loop)

    assert install_resume_epoch_patch() is True
    assert copied.skip_first_batches is not accelerate.skip_first_batches
    assert copied.skip_first_batches is trainer_module.skip_first_batches
