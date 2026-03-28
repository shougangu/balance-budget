# ABOUTME: Tests for GRPO reward functions (GSM8K and IFEval).
# ABOUTME: Validates reward interface, correctness scoring, and edge cases.

from tuning.training.reward_functions import gsm8k_reward_func, ifeval_reward_func


class TestGSM8KReward:
    def test_correct_answer_gets_reward_1(self):
        rewards = gsm8k_reward_func(
            prompts=["What is 2+2?"],
            completions=["Step 1: 2+2=4\n\n#### 4"],
            reference_answer=["4"],
        )
        assert rewards == [1.0]

    def test_incorrect_answer_gets_reward_0(self):
        rewards = gsm8k_reward_func(
            prompts=["What is 2+2?"],
            completions=["Step 1: 2+2=5\n\n#### 5"],
            reference_answer=["4"],
        )
        assert rewards == [0.0]

    def test_no_answer_gets_reward_0(self):
        rewards = gsm8k_reward_func(
            prompts=["What is 2+2?"],
            completions=["I don't know how to solve this."],
            reference_answer=["4"],
        )
        assert rewards == [0.0]

    def test_batch_of_mixed_answers(self):
        rewards = gsm8k_reward_func(
            prompts=["q1", "q2", "q3"],
            completions=[
                "#### 42",
                "#### 99",
                "#### 7",
            ],
            reference_answer=["42", "100", "7"],
        )
        assert rewards == [1.0, 0.0, 1.0]

    def test_conversational_format_completions(self):
        """GRPOTrainer may pass completions as list of message dicts."""
        rewards = gsm8k_reward_func(
            prompts=[{"role": "user", "content": "q1"}],
            completions=[[{"role": "assistant", "content": "#### 42"}]],
            reference_answer=["42"],
        )
        assert rewards == [1.0]


class TestIFEvalReward:
    def test_returns_float_between_0_and_1(self):
        rewards = ifeval_reward_func(
            prompts=["Write a poem in exactly 3 sentences."],
            completions=["The sun rose high. Birds sang aloud. Day had begun."],
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0

    def test_batch_returns_correct_length(self):
        rewards = ifeval_reward_func(
            prompts=["Write a haiku.", "Say hello."],
            completions=["Cherry blossoms fall / gently on the quiet stream / spring has come at last", "Hello!"],
        )
        assert len(rewards) == 2
        assert all(0.0 <= r <= 1.0 for r in rewards)
