# ABOUTME: Reward functions for GRPO/RLVR training matching TRL's GRPOTrainer interface.
# ABOUTME: GSM8K uses binary correctness; IFEval uses fractional instruction compliance.

from tuning.evaluation.gsm8k_scoring import is_correct


def _extract_text(completion):
    """Extract text from a completion that may be a string or list of message dicts."""
    if isinstance(completion, str):
        return completion
    # Conversational format: list of message dicts
    return completion[-1]["content"]


def gsm8k_reward_func(prompts, completions, reference_answer, **kwargs):
    """Binary reward: 1.0 if the completion's answer matches the reference, 0.0 otherwise."""
    rewards = []
    for completion, ref in zip(completions, reference_answer):
        text = _extract_text(completion)
        rewards.append(1.0 if is_correct(text, ref) else 0.0)
    return rewards


def ifeval_reward_func(prompts, completions, **kwargs):
    """Partial credit reward: fraction of instructions followed in each completion.

    Falls back to 0.0 if the prompt can't be matched to an IFEval InputExample
    (e.g. prompts not from the IFEval dataset).
    """
    from instruction_following_eval import evaluation_lib

    # Lazy-load IFEval input examples and build prompt→example index
    if not hasattr(ifeval_reward_func, "_prompt_to_example"):
        from tuning.training.eval_strategy import IFEVAL_INPUT_PATH
        examples = evaluation_lib.read_prompt_list(str(IFEVAL_INPUT_PATH))
        ifeval_reward_func._prompt_to_example = {ex.prompt: ex for ex in examples}

    prompt_to_example = ifeval_reward_func._prompt_to_example

    rewards = []
    for prompt, completion in zip(prompts, completions):
        prompt_text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        text = _extract_text(completion)
        inp = prompt_to_example.get(prompt_text)
        if inp is None:
            rewards.append(0.0)
            continue
        result = evaluation_lib.test_instruction_following_loose(inp, {inp.prompt: text})
        score = sum(result.follow_instruction_list) / len(result.follow_instruction_list)
        rewards.append(score)
    return rewards
