# ABOUTME: Tests for RLVR prompt-only dataset loaders (GSM8K and IFEval).
# ABOUTME: Validates dataset format, column names, and deduplication.

from tuning.data.gsm8k_rlvr import GSM8KRLVR
from tuning.data.tuluif_rlvr import TuluIFRLVR


def test_gsm8k_rlvr_format_produces_prompt_and_reference():
    """RLVR dataset must have 'prompt' (list of message dicts) and 'reference_answer' columns."""
    ds = GSM8KRLVR()
    ds.load_from_huggingface("mohit-raghavendra/gsm8k-gpt4ogenerations", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()

    train = dataset["train"]
    assert "prompt" in train.column_names
    assert "reference_answer" in train.column_names
    # prompt should be a list of message dicts (system + user)
    row = train[0]
    assert isinstance(row["prompt"], list)
    assert len(row["prompt"]) == 2
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
    # reference_answer should be a string
    assert isinstance(row["reference_answer"], str)
    assert len(row["reference_answer"]) > 0


def test_gsm8k_rlvr_deduplicates_prompts():
    """RLVR dataset should have one row per unique question, not per response."""
    ds = GSM8KRLVR()
    ds.load_from_huggingface("mohit-raghavendra/gsm8k-gpt4ogenerations", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()
    train = dataset["train"]

    prompts = train["prompt"]
    prompt_texts = [p[1]["content"] for p in prompts]
    assert len(prompt_texts) == len(set(prompt_texts)), "RLVR dataset should have unique prompts"


def test_tuluif_rlvr_format_produces_prompt():
    """IFEval RLVR dataset must have 'prompt' column with conversational format."""
    ds = TuluIFRLVR()
    ds.load_from_huggingface("allenai/tulu-3-sft-personas-instruction-following", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()

    train = dataset["train"]
    assert "prompt" in train.column_names
    row = train[0]
    assert isinstance(row["prompt"], list)
    assert len(row["prompt"]) == 2
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
