# ABOUTME: Tests for IF-RLVR dataset loader.
# ABOUTME: Validates format, columns, and ground_truth preservation.

from tuning.data.ifrlvr_rlvr import IfrlvrRLVR, MAX_PROMPT_TOKENS


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


def _make_mock_rows(prompt_contents):
    return [{"messages": [{"content": c}], "ground_truth": "[]"} for c in prompt_contents]


class FakeTokenizer:
    def apply_chat_template(self, prompt, tokenize=False, add_generation_prompt=False):
        return prompt[1]["content"]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[0] * len(text.split()) for text in texts]}


class ScalingFakeTokenizer(FakeTokenizer):
    def __init__(self, multiplier):
        self.multiplier = multiplier

    def __call__(self, texts, **kwargs):
        return {"input_ids": [[0] * len(text.split()) * self.multiplier for text in texts]}


def test_ifrlvr_filters_long_prompts():
    ds = IfrlvrRLVR(tokenizers=(("fake", FakeTokenizer()),))
    short = "This is a short prompt."
    long = " ".join(["x"] * (MAX_PROMPT_TOKENS + 1))
    rows = ds._get_rows(_make_mock_rows([short, long]))
    contents = [r["prompt"][1]["content"] for r in rows]
    assert all(len(c.split()) <= MAX_PROMPT_TOKENS for c in contents)
    assert short in contents
    assert long not in contents


def test_ifrlvr_keeps_prompts_exactly_at_limit():
    ds = IfrlvrRLVR(tokenizers=(("fake", FakeTokenizer()),))
    at_limit = " ".join(["a"] * MAX_PROMPT_TOKENS)
    rows = ds._get_rows(_make_mock_rows([at_limit]))
    assert len(rows) == 1
    assert rows[0]["prompt"][1]["content"] == at_limit


def test_ifrlvr_filters_when_any_tokenizer_exceeds_limit():
    ds = IfrlvrRLVR(
        max_prompt_tokens=3,
        tokenizers=(
            ("fits", FakeTokenizer()),
            ("over", ScalingFakeTokenizer(multiplier=2)),
        ),
    )
    rows = ds._get_rows(_make_mock_rows(["one two three"]))
    assert rows == []


def test_ifrlvr_deduplicates_prompts():
    ds = IfrlvrRLVR()
    ds.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()
    train = dataset["train"]

    prompt_texts = [p[1]["content"] for p in train["prompt"]]
    assert len(prompt_texts) == len(set(prompt_texts)), "IF-RLVR dataset should have unique prompts"
