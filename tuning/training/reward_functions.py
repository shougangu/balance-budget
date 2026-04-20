# ABOUTME: Reward functions for GRPO/RLVR training matching TRL's GRPOTrainer interface.
# ABOUTME: GSM8K and MATH-500 use binary correctness; IFEval and IF-RLVR use fractional instruction compliance.

import ast
import re

from tuning.evaluation.gsm8k_scoring import is_correct as gsm8k_is_correct
from tuning.evaluation.math500_scoring import is_correct as math500_is_correct


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
        rewards.append(1.0 if gsm8k_is_correct(text, ref) else 0.0)
    return rewards


def math500_reward_func(prompts, completions, reference_answer, **kwargs):
    """Binary reward: 1.0 if the completion's answer matches the reference, 0.0 otherwise.

    Uses math-verify for extraction (handles \\boxed{} and $...$) with #### fallback.
    """
    rewards = []
    for completion, ref in zip(completions, reference_answer):
        text = _extract_text(completion)
        rewards.append(1.0 if math500_is_correct(text, ref) else 0.0)
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


def _remove_thinking_section(text):
    """Strip chain-of-thought markup before constraint checking."""
    text = text.replace("<|assistant|>", "")
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<answer>", "").replace("</answer>", "")
    return text.strip()


def ifrlvr_reward_func(prompts, completions, ground_truth, **kwargs):
    """Fractional reward: fraction of IF-RLVR constraints satisfied per completion."""
    if not hasattr(ifrlvr_reward_func, "_instruction_dict"):
        from ifrlvr.instructions_registry import INSTRUCTION_DICT
        ifrlvr_reward_func._instruction_dict = INSTRUCTION_DICT

    instruction_dict = ifrlvr_reward_func._instruction_dict

    rewards = []
    for completion, gt_str in zip(completions, ground_truth):
        text = _extract_text(completion)
        answer = _remove_thinking_section(text)

        if not answer.strip():
            rewards.append(0.0)
            continue

        constraint_list = ast.literal_eval(gt_str)
        constraint_dict = constraint_list[0]
        if isinstance(constraint_dict, str):
            import json
            constraint_dict = json.loads(constraint_dict)

        instruction_keys = constraint_dict["instruction_id"]
        args_list = constraint_dict["kwargs"]

        passed = []
        for inst_key, inst_args in zip(instruction_keys, args_list):
            args = {} if inst_args is None else {k: v for k, v in inst_args.items() if v is not None}
            instruction_cls = instruction_dict[inst_key]
            instruction = instruction_cls(inst_key)
            instruction.build_description(**args)
            if instruction.check_following(answer):
                passed.append(1.0)
            else:
                passed.append(0.0)

        rewards.append(sum(passed) / len(passed) if passed else 0.0)

    return rewards
