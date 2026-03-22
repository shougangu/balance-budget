# ABOUTME: ABC for eval strategies injected into the generation eval callback.
# ABOUTME: Includes IFEval and GSM8K pass@k implementations.

import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from instruction_following_eval import evaluation_lib
from tuning.data.test_dataset import get_ifeval_test_dataset, get_gsm8k_test_dataset
from tuning.evaluation.gsm8k_scoring import is_correct as gsm8k_is_correct

BASE_DIR = Path('/home/shougan/projects/aip-fredashi/shougan/balance-budget')
IFEVAL_INPUT_PATH = BASE_DIR / "instruction_following_eval/data/input_data.jsonl"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k: probability that at least one of k samples is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def format_vllm_outputs(outputs, prompts: List[str], n_samples: int = 1) -> List[Dict]:
    """Convert vLLM chat outputs to the [{prompt, responses}] format used by score_responses().

    Groups responses by prompt, merging duplicates. Preserves first-seen order.
    """
    grouped = defaultdict(list)
    for prompt, output in zip(prompts, outputs):
        if n_samples == 1:
            grouped[prompt].append(output.outputs[0].text)
        else:
            grouped[prompt].extend(r.text for r in output.outputs)
    return [{"prompt": p, "responses": grouped[p]} for p in dict.fromkeys(prompts)]


def evaluate_single_response(inp: evaluation_lib.InputExample, response: str, strict: bool = True, easy = False) -> bool:
    """Evaluate a single response using the pre-built IFEval functions -> returns 1 iff all instructions followed"""
    prompt_to_response = {inp.prompt: response}
    if strict:
        result = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    else:
        result = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
    return result.follow_all_instructions

def evaluate_single_response_instruction(inp: evaluation_lib.InputExample, response: str) -> float:
    "Evaluate a single response using the pre-built IFEval functions -> return the % of instructions followed"
    result = evaluation_lib.test_instruction_following_loose(inp, {inp.prompt: response})
    return sum(result.follow_instruction_list) / len(result.follow_instruction_list)

def evaluate_multiple_responses_instruction(inp: evaluation_lib.InputExample, responses: List[str], k: int, strict: bool = False) -> float:
    """
    Given multiple responses, calculate instruction-level pass@k.
    If there are many instruction-following criterion (ie: follow_instruction_1, ..., following_instruction_n),
    for every response, the pass@k is calculated as the average pass@k across individual criterion in the responses
    """
    eval_fn = evaluation_lib.test_instruction_following_strict if strict else evaluation_lib.test_instruction_following_loose
    result_matrix = [eval_fn(inp, {inp.prompt: r}) for r in responses]
    n = len(result_matrix)
    num_instructions = len(result_matrix[0].follow_instruction_list)
    pass_at_k_list = []
    for c in range(num_instructions):
        correct_count = sum(result_matrix[r].follow_instruction_list[c] for r in range(n))
        pass_at_k_list.append(pass_at_k(n, correct_count, k))
    return sum(pass_at_k_list) / len(pass_at_k_list)

class EvalStrategy(ABC):
    """Defines what prompts to generate and how to score responses."""
    @abstractmethod
    def get_test_messages(self) -> List[List[dict]]:
        """Chat messages to send to vLLM."""

    @abstractmethod
    def get_test_prompts(self) -> List[str]:
        """Raw prompt strings, parallel to get_test_messages()."""

    @abstractmethod
    def score_responses(self, results: List[Dict], tokenizer) -> Dict[str, float]:
        """Score vLLM outputs. Returns metric dict."""

    @abstractmethod
    def stopping_metric(self) -> str:
        """Which key from score_responses() to use for thresholds."""

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this eval strategy (e.g., 'ifeval_pass_at_1')."""

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Number of completions to generate per prompt."""

    @property
    @abstractmethod
    def label_prefix(self) -> str:
        """Prefix for checkpoint labels (e.g., 'p@1')."""

    @abstractmethod
    def wandb_metrics(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Format scores for wandb logging."""


class IFEvalStrategy(EvalStrategy):
    """IFEval pass@k evaluation strategy."""

    def __init__(self, k_values=None, n_samples=1, num_prompts=541, strict=True):
        k_values = k_values or [1]
        self.k_values = k_values
        self.stopping_k = k_values[0]
        self._n_samples = n_samples
        self.strict = strict

        self.test_dataset = get_ifeval_test_dataset()
        if num_prompts is not None:
            self.test_dataset = self.test_dataset.select(
                range(min(num_prompts, len(self.test_dataset)))
            )

        self.inputs_map = {
            inp.prompt: inp
            for inp in evaluation_lib.read_prompt_list(str(IFEVAL_INPUT_PATH))
        }

        print(f"[IFEvalStrategy] k_values={k_values}, n_samples={n_samples}, "
              f"strict={strict}, num_prompts={len(self.test_dataset)}")

    @property
    def n_samples(self) -> int:
        return self._n_samples

    def get_test_messages(self) -> List[List[dict]]:
        return self.test_dataset["messages"]

    def get_test_prompts(self) -> List[str]:
        return self.test_dataset["prompt"]

    def score_responses(self, results: List[Dict], tokenizer) -> Dict[str, float]:
        all_prompt_results = []
        all_instruction_scores = {k: [] for k in self.k_values}
        response_token_lengths = []

        for item in results:
            prompt = item["prompt"]
            responses = item["responses"]

            encoded_batch = tokenizer(
                responses, add_special_tokens=False, padding=False, truncation=False,
            )
            response_token_lengths.extend(len(ids) for ids in encoded_batch["input_ids"])

            eval_input = self.inputs_map[prompt]

            prompt_results = [evaluate_single_response(eval_input, r, self.strict) for r in responses]
            all_prompt_results.append(prompt_results)

            for k in self.k_values:
                all_instruction_scores[k].append(
                    evaluate_multiple_responses_instruction(eval_input, responses, k, strict=self.strict)
                )

            item["per_response_correct"] = prompt_results

        scores = {}

        # Main: instruction-level pass@k
        for k in self.k_values:
            scores[f"pass_at_{k}"] = np.mean(all_instruction_scores[k])

        # Prompt-level pass@k (all-or-nothing)
        for k in self.k_values:
            prompt_scores = [pass_at_k(len(r), sum(r), k) for r in all_prompt_results]
            scores[f"pass_at_{k}_prompt"] = np.mean(prompt_scores)

        scores["num_prompts_evaluated"] = len(results)
        scores["avg_response_length_tokens"] = (
            float(np.mean(response_token_lengths)) if response_token_lengths else 0.0
        )
        return scores

    @property
    def id(self) -> str:
        return "ifeval"
    
    def stopping_metric(self) -> str:
        return f"pass_at_{self.stopping_k}"

    @property
    def label_prefix(self) -> str:
        return f"p@{self.stopping_k}"

    def wandb_metrics(self, scores: Dict[str, float]) -> Dict[str, float]:
        metrics = {}
        for k in self.k_values:
            metrics[f"eval/pass_at_{k}"] = scores[f"pass_at_{k}"]
            metrics[f"eval/pass_at_{k}_prompt"] = scores[f"pass_at_{k}_prompt"]
        metrics["eval/avg_response_length_tokens"] = scores.get("avg_response_length_tokens", 0.0)
        return metrics


class GSM8KEvalStrategy(EvalStrategy):
    """GSM8K evaluation using pass@k scoring."""

    def __init__(self, k_values=None, n_samples=1, num_prompts=None):
        k_values = k_values or [1]
        self.k_values = k_values
        self._n_samples = n_samples
        self.stopping_k = k_values[0]

        self.test_dataset = get_gsm8k_test_dataset(num_prompts=num_prompts)
        self.reference_answers = {
            prompt: ref
            for prompt, ref in zip(
                self.test_dataset["prompt"],
                self.test_dataset["reference_answer"],
            )
        }

        print(f"[GSM8KEvalStrategy] k_values={k_values}, n_samples={n_samples}, "
              f"num_prompts={len(self.test_dataset)}")

    @property
    def n_samples(self) -> int:
        return self._n_samples
    
    @property
    def id(self) -> str:
        return "gsm8k"
    
    def get_test_messages(self) -> List[List[dict]]:
        return self.test_dataset["messages"]

    def get_test_prompts(self) -> List[str]:
        return self.test_dataset["prompt"]

    def score_responses(self, results: List[Dict], tokenizer) -> Dict[str, float]:
        all_results = []
        response_token_lengths = []

        for item in results:
            prompt = item["prompt"]
            responses = item["responses"]
            ref = self.reference_answers[prompt]

            encoded_batch = tokenizer(
                responses, add_special_tokens=False, padding=False, truncation=False,
            )
            response_token_lengths.extend(len(ids) for ids in encoded_batch["input_ids"])

            eval_results = [gsm8k_is_correct(r, ref) for r in responses]
            all_results.append(eval_results)

            item["per_response_correct"] = eval_results

        scores = {}
        for k in self.k_values:
            pass_at_k_scores = [pass_at_k(len(r), sum(r), k) for r in all_results]
            scores[f"pass_at_{k}"] = np.mean(pass_at_k_scores)

        scores["num_prompts_evaluated"] = len(all_results)
        scores["avg_response_length_tokens"] = (
            float(np.mean(response_token_lengths)) if response_token_lengths else 0.0
        )
        return scores

    def stopping_metric(self) -> str:
        return f"pass_at_{self.stopping_k}"

    @property
    def label_prefix(self) -> str:
        return f"gsm8k-p@{self.stopping_k}"

    def wandb_metrics(self, scores: Dict[str, float]) -> Dict[str, float]:
        metrics = {}
        for k in self.k_values:
            key = f"pass_at_{k}"
            if key in scores:
                metrics[f"eval/gsm8k_{key}"] = scores[key]
        metrics["eval/avg_response_length_tokens"] = scores.get("avg_response_length_tokens", 0.0)
        return metrics
