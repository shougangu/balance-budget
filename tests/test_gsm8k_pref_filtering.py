# ABOUTME: Tests for GSM8K preference dataset filtering by chosen correctness.
# ABOUTME: Validates that only rows with verified-correct chosen answers are kept.

from datasets import Dataset, DatasetDict
from tuning.data.gsm8k_pref import filter_correct_chosen


def _make_dataset(rows):
    """Build a DatasetDict with the given rows as the train split and a dummy test split."""
    train = Dataset.from_dict({
        "system_message": [r["system_message"] for r in rows],
        "prompt": [r["prompt"] for r in rows],
        "reference_solution": [r["reference_solution"] for r in rows],
        "chosen": [r["chosen"] for r in rows],
        "chosen_rating": [r["chosen_rating"] for r in rows],
        "rejected": [r["rejected"] for r in rows],
        "rejected_rating": [r["rejected_rating"] for r in rows],
    })
    test = Dataset.from_dict({
        "system_message": ["sys"],
        "prompt": ["test prompt"],
        "reference_solution": ["5"],
        "chosen": ["#### 5"],
        "chosen_rating": ["5"],
        "rejected": ["#### 3"],
        "rejected_rating": ["1"],
    })
    return DatasetDict({"train": train, "test": test})


def _row(prompt, chosen, reference, rejected="#### 0", chosen_rating="5", rejected_rating="1"):
    return {
        "system_message": "sys",
        "prompt": prompt,
        "reference_solution": reference,
        "chosen": chosen,
        "chosen_rating": chosen_rating,
        "rejected": rejected,
        "rejected_rating": rejected_rating,
    }


# --- filter_correct_chosen tests ---


def test_filters_incorrect_chosen():
    """Rows where chosen answer doesn't match reference are removed."""
    ds = _make_dataset([
        _row("q1", "The answer is #### 10", "10"),
        _row("q2", "The answer is #### 99", "10"),       # incorrect
        _row("q3", "The answer is #### 5", "5"),
    ])
    result = filter_correct_chosen(ds)
    prompts = result["train"]["prompt"]
    assert len(prompts) == 2
    assert "q1" in prompts
    assert "q3" in prompts
    assert "q2" not in prompts


def test_keeps_duplicate_prompts_when_both_correct():
    """Multiple rows with the same prompt are all kept if chosen is correct."""
    ds = _make_dataset([
        _row("q1", "The answer is #### 10", "10", rejected="#### 0"),
        _row("q1", "Another correct #### 10", "10", rejected="#### 5"),
        _row("q2", "The answer is #### 7", "7"),
    ])
    result = filter_correct_chosen(ds)
    prompts = result["train"]["prompt"]
    assert len(prompts) == 3
    assert prompts.count("q1") == 2


def test_empty_after_filtering():
    """If all chosen answers are wrong, result has zero train rows."""
    ds = _make_dataset([
        _row("q1", "The answer is #### 99", "10"),
        _row("q2", "The answer is #### 3", "7"),
    ])
    result = filter_correct_chosen(ds)
    assert len(result["train"]) == 0


def test_preserves_test_split():
    """Test split is unchanged by filtering."""
    ds = _make_dataset([
        _row("q1", "The answer is #### 10", "10"),
    ])
    result = filter_correct_chosen(ds)
    assert len(result["test"]) == 1
    assert result["test"]["prompt"][0] == "test prompt"


def test_preserves_all_columns():
    """All columns from the original dataset are preserved."""
    ds = _make_dataset([
        _row("q1", "The answer is #### 10", "10", rejected="#### 0",
             chosen_rating="5", rejected_rating="1"),
    ])
    result = filter_correct_chosen(ds)
    row = result["train"][0]
    assert row["system_message"] == "sys"
    assert row["prompt"] == "q1"
    assert row["reference_solution"] == "10"
    assert row["chosen"] == "The answer is #### 10"
    assert row["chosen_rating"] == "5"
    assert row["rejected"] == "#### 0"
    assert row["rejected_rating"] == "1"
