# ABOUTME: Regression tests for padding variable-length SFT completion masks.
# ABOUTME: TRL builds this collator from completion_only_loss; masking must survive ragged batches.

import torch

from trl.trainer.sft_trainer import DataCollatorForLanguageModeling as SFTDataCollatorForLanguageModeling


def test_completion_collator_pads_variable_length_masks():
    collator = SFTDataCollatorForLanguageModeling(
        pad_token_id=0,
        completion_only_loss=True,
    )
    batch = collator(
        [
            {
                "input_ids": [1] * 107,
                "completion_mask": [0] * 50 + [1] * 57,
            },
            {
                "input_ids": [2] * 232,
                "completion_mask": [0] * 100 + [1] * 132,
            },
        ]
    )

    assert batch["input_ids"].shape == torch.Size([2, 232])
    assert batch["labels"].shape == torch.Size([2, 232])
    assert torch.all(batch["labels"][0, :50] == -100)
    assert torch.all(batch["labels"][0, 50:107] == 1)
    assert torch.all(batch["labels"][0, 107:] == -100)
    assert torch.all(batch["labels"][1, :100] == -100)
    assert torch.all(batch["labels"][1, 100:] == 2)
