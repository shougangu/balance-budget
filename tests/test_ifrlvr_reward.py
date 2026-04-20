# ABOUTME: Tests for IF-RLVR reward function using IFEvalG constraint checkers.
# ABOUTME: Validates ground_truth parsing, null-filtering, and fractional scoring.

from tuning.training.reward_functions import ifrlvr_reward_func, _remove_thinking_section


class TestRemoveThinkingSection:
    def test_strips_think_tags(self):
        text = "<think>some reasoning</think>The actual answer."
        assert _remove_thinking_section(text) == "The actual answer."

    def test_strips_answer_tags(self):
        text = "<answer>42</answer>"
        assert _remove_thinking_section(text) == "42"

    def test_strips_assistant_prefix(self):
        text = "<|assistant|>Hello world"
        assert _remove_thinking_section(text) == "Hello world"

    def test_no_tags_returns_unchanged(self):
        text = "Plain response with no tags."
        assert _remove_thinking_section(text) == "Plain response with no tags."

    def test_empty_string(self):
        assert _remove_thinking_section("") == ""


class TestIfrlvrReward:
    def test_returns_float_between_0_and_1(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['hello']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["Include the word hello."],
            completions=["hello world"],
            ground_truth=[gt],
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0

    def test_keyword_present_gets_reward_1(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['banana']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["Include banana."],
            completions=["I like banana."],
            ground_truth=[gt],
        )
        assert rewards == [1.0]

    def test_keyword_missing_gets_reward_0(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['banana']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["Include banana."],
            completions=["I like apples."],
            ground_truth=[gt],
        )
        assert rewards == [0.0]

    def test_none_kwargs_handled(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [None]}]"
        rewards = ifrlvr_reward_func(
            prompts=["test"],
            completions=["test response"],
            ground_truth=[gt],
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0

    def test_batch_returns_correct_length(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['hello']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["p1", "p2"],
            completions=["hello", "goodbye"],
            ground_truth=[gt, gt],
        )
        assert len(rewards) == 2

    def test_conversational_format_completions(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['banana']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=[{"role": "user", "content": "test"}],
            completions=[[{"role": "assistant", "content": "I have a banana."}]],
            ground_truth=[gt],
        )
        assert rewards == [1.0]

    def test_thinking_section_stripped_before_eval(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['banana']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["Include banana."],
            completions=["<think>banana is not the answer</think>I have a banana."],
            ground_truth=[gt],
        )
        assert rewards == [1.0]

    def test_empty_completion_gets_reward_0(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['hello']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["test"],
            completions=[""],
            ground_truth=[gt],
        )
        assert rewards == [0.0]
