# ABOUTME: Tests that SFT prompt tokens are masked out of the loss via a completion_mask column.
# ABOUTME: Uses real tokenizers because the seam behaviour under test is real BPE merge behaviour.

import pytest
from datasets import Dataset, DatasetDict

import tuning.config
from tuning.config import HF_MODEL_MAP
from tuning.utils.utils import (
    GEMMA_3_CHAT_TEMPLATE,
    LLAMA_31_SIMPLE_TEMPLATE,
    SIMPLE_TEMPLATE,
    apply_chat_template,
    tokenize_sft_dataset,
)


# A response starting with "A" makes the prompt-final ":" merge into a single ":A"
# token under the Llama BPE, which is the case a delimiter search cannot resolve.
STRADDLING_CONVO = [
    {"role": "system", "content": "You solve math problems."},
    {"role": "user", "content": "Problem: 2+2?\nAnswer:"},
    {"role": "assistant", "content": "A body of work gives \\boxed{4}."},
]

NON_STRADDLING_CONVO = [
    {"role": "system", "content": "You solve math problems."},
    {"role": "user", "content": "Problem: 3+3?\nAnswer:"},
    {"role": "assistant", "content": "Let us compute \\boxed{6}."},
]


def _tokenizer(model_key, template):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_MAP[model_key])
    tokenizer.chat_template = template
    return tokenizer


def _prepare(tokenizer, convos, max_length=128, mask_prompt=True):
    dataset = DatasetDict({"train": Dataset.from_dict({"messages": convos})})
    dataset = apply_chat_template(tokenizer, dataset, mask_prompt=mask_prompt)
    return tokenize_sft_dataset(
        tokenizer, dataset, max_length=max_length, num_proc=None, mask_prompt=mask_prompt
    )["train"]


TEMPLATE_CASES = [
    ("llama3-8B", SIMPLE_TEMPLATE, "simple"),
    ("llama3-8B", LLAMA_31_SIMPLE_TEMPLATE, "llama-3.1"),
    ("gemma3-4B", SIMPLE_TEMPLATE, "simple"),
    ("gemma3-4B", GEMMA_3_CHAT_TEMPLATE, "gemma-3"),
]


@pytest.mark.parametrize("model_key,template,template_name", TEMPLATE_CASES)
def test_prompt_tokens_masked_and_response_supervised(model_key, template, template_name):
    """Every template masks a prompt prefix and supervises a non-empty response."""
    tokenizer = _tokenizer(model_key, template)
    row = _prepare(tokenizer, [STRADDLING_CONVO, NON_STRADDLING_CONVO])[0]

    mask = row["completion_mask"]
    assert len(mask) == len(row["input_ids"])
    assert set(mask) == {0, 1}, f"{template_name} produced a degenerate mask"
    # The mask is a prefix of zeros followed by ones; no interleaving.
    assert mask == sorted(mask), f"{template_name} mask is not contiguous"


@pytest.mark.parametrize("model_key,template,template_name", TEMPLATE_CASES)
def test_final_token_stays_supervised(model_key, template, template_name):
    """The terminal token must be trained or the model never learns to stop."""
    tokenizer = _tokenizer(model_key, template)
    row = _prepare(tokenizer, [STRADDLING_CONVO])[0]

    assert row["completion_mask"][-1] == 1


@pytest.mark.parametrize("model_key,template,template_name", TEMPLATE_CASES)
def test_no_row_is_fully_masked(model_key, template, template_name):
    """A row with zero supervised tokens would silently contribute no gradient."""
    tokenizer = _tokenizer(model_key, template)
    rows = _prepare(tokenizer, [STRADDLING_CONVO, NON_STRADDLING_CONVO])

    for row in rows:
        assert any(row["completion_mask"]), f"{template_name} fully masked a row"


def test_token_straddling_the_seam_is_supervised():
    """"Answer:" + "A body" merges into one ":A" token, which must count as response.

    Assigning it to the prompt would drop the first response token from the loss.
    """
    tokenizer = _tokenizer("llama3-8B", SIMPLE_TEMPLATE)
    row = _prepare(tokenizer, [STRADDLING_CONVO])[0]

    tokens = tokenizer.convert_ids_to_tokens(row["input_ids"])
    straddling = [i for i, tok in enumerate(tokens) if tok == ":A"]
    assert straddling, f"expected a merged ':A' token, got {tokens}"
    for index in straddling:
        assert row["completion_mask"][index] == 1


def test_masking_does_not_change_input_ids():
    """Labels change; the token stream must stay byte-identical to unmasked runs."""
    tokenizer = _tokenizer("llama3-8B", SIMPLE_TEMPLATE)
    masked = _prepare(tokenizer, [STRADDLING_CONVO], mask_prompt=True)[0]
    unmasked = _prepare(tokenizer, [STRADDLING_CONVO], mask_prompt=False)[0]

    assert masked["input_ids"] == unmasked["input_ids"]


def test_single_bos_is_preserved():
    """The template supplies BOS; the tokenizer must not prepend a second one."""
    tokenizer = _tokenizer("llama3-8B", SIMPLE_TEMPLATE)
    row = _prepare(tokenizer, [STRADDLING_CONVO])[0]

    input_ids = row["input_ids"]
    assert input_ids[0] == tokenizer.bos_token_id
    assert input_ids[1] != tokenizer.bos_token_id
    # BOS is context, never a prediction target.
    assert row["completion_mask"][0] == 0


def test_mask_column_absent_when_disabled():
    """Default-off keeps the dataset identical to what existing runs consume."""
    tokenizer = _tokenizer("llama3-8B", SIMPLE_TEMPLATE)
    rows = _prepare(tokenizer, [STRADDLING_CONVO], mask_prompt=False)

    assert "completion_mask" not in rows.column_names
    assert "prompt_length" not in rows.column_names


def test_multi_turn_conversation_is_rejected():
    """Only the last assistant turn is derivable; earlier ones would be masked silently."""
    tokenizer = _tokenizer("llama3-8B", SIMPLE_TEMPLATE)
    multi_turn = [
        {"role": "user", "content": "First?"},
        {"role": "assistant", "content": "One."},
        {"role": "user", "content": "Second?"},
        {"role": "assistant", "content": "Two."},
    ]

    with pytest.raises(ValueError, match="single assistant turn"):
        _prepare(tokenizer, [multi_turn])


def test_truncation_keeps_mask_aligned_with_input_ids():
    """Rows longer than max_length must not desynchronise mask and tokens."""
    tokenizer = _tokenizer("llama3-8B", SIMPLE_TEMPLATE)
    long_convo = [
        {"role": "user", "content": "Problem: " + "x " * 200 + "\nAnswer:"},
        {"role": "assistant", "content": "A body " * 200},
    ]
    row = _prepare(tokenizer, [long_convo], max_length=64)[0]

    assert len(row["input_ids"]) == 64
    assert len(row["completion_mask"]) == 64
