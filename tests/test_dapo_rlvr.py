# ABOUTME: Tests DAPO-Math-17k RLVR row formatting uses the plain prompt, not source_prompt.

from tuning.data.dapo_rlvr import DAPORLVR


def test_dapo_rlvr_uses_prompt_and_ground_truth():
    fake = [{
        "prompt": "What is 2+2?",
        "source_prompt": [{"role": "user", "content": "FOLLOW THESE INSTRUCTIONS: What is 2+2?"}],
        "reward_model": {"ground_truth": "4", "style": "rule"},
        "solution": "4",
    }]
    rows = DAPORLVR()._get_rows(fake)
    assert len(rows) == 1
    assert rows[0]["reference_answer"] == "4"
    user_msg = rows[0]["prompt"][-1]["content"]
    assert "What is 2+2?" in user_msg
    assert "FOLLOW THESE INSTRUCTIONS" not in user_msg
    assert rows[0]["prompt"][0]["role"] == "system"


def test_dapo_rlvr_dedupes_identical_prompts():
    fake = [
        {"prompt": "Q1", "reward_model": {"ground_truth": "1"}},
        {"prompt": "Q1", "reward_model": {"ground_truth": "1"}},
        {"prompt": "Q2", "reward_model": {"ground_truth": "2"}},
    ]
    rows = DAPORLVR()._get_rows(fake)
    assert len(rows) == 2
