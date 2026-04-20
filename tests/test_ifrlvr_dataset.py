# ABOUTME: Tests for IF-RLVR dataset loader.
# ABOUTME: Validates format, columns, and ground_truth preservation.

from tuning.data.ifrlvr_rlvr import IfrlvrRLVR


def test_ifrlvr_format_produces_prompt_and_ground_truth():
    ds = IfrlvrRLVR()
    ds.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()

    train = dataset["train"]
    assert "prompt" in train.column_names
    assert "ground_truth" in train.column_names

    row = train[0]
    assert isinstance(row["prompt"], list)
    assert len(row["prompt"]) == 2
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
    assert isinstance(row["ground_truth"], str)
    assert len(row["ground_truth"]) > 0


def test_ifrlvr_ground_truth_is_parseable():
    import ast
    ds = IfrlvrRLVR()
    ds.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()

    row = dataset["train"][0]
    parsed = ast.literal_eval(row["ground_truth"])
    assert isinstance(parsed, list)
    assert len(parsed) >= 1
    assert "instruction_id" in parsed[0]
    assert "kwargs" in parsed[0]


def test_ifrlvr_deduplicates_prompts():
    ds = IfrlvrRLVR()
    ds.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()
    train = dataset["train"]

    prompt_texts = [p[1]["content"] for p in train["prompt"]]
    assert len(prompt_texts) == len(set(prompt_texts)), "IF-RLVR dataset should have unique prompts"
