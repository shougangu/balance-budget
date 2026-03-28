# ABOUTME: Tests that GSM8K SFT dataset creation includes a prompt column.
# ABOUTME: Validates the prompt column matches the user message content for unique-first sampling.

from tuning.data.gsm8k_sft import GSM8KSFT


def test_get_rows_includes_prompt_column():
    """Each row has a 'prompt' field matching the user message content."""
    dataset = {
        "prompt": ["What is 2+2?"],
        "correct": [["Step 1: 2+2=4\n\n#### 4"]],
        "reference_answer": ["4"],
    }
    sft = GSM8KSFT()
    rows = sft._get_rows(dataset)
    assert len(rows) == 1
    assert "prompt" in rows[0]
    assert rows[0]["prompt"] == rows[0]["messages"][1]["content"]
