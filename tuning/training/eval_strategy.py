# ABOUTME: ABC for eval strategies injected into the generation eval callback.
# ABOUTME: Includes IFEval pass@k implementation.

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

from instruction_following_eval import evaluation_lib
from tuning.data.test_dataset import get_ifeval_test_dataset
from tuning.training.config_training import IFEvalConfig

BASE_DIR = Path('/home/shougan/projects/aip-fredashi/shougan/balance-budget')
IFEVAL_INPUT_PATH = BASE_DIR / "instruction_following_eval/data/input_data.jsonl"


def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k: probability that at least one of k samples is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_single_response(inp: evaluation_lib.InputExample, response: str, strict: bool = True) -> bool:
    """Evaluate a single response using the pre-built IFEval functions."""
    prompt_to_response = {inp.prompt: response}
    if strict:
        result = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    else:
        result = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
    return result.follow_all_instructions


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

    def __init__(self, config: IFEvalConfig, tokenizer):
        self.k_values = config.k_values
        self.stopping_k = config.k_values[0]
        self._n_samples = config.n_samples
        self.strict = config.strict

        self.test_dataset = get_ifeval_test_dataset()
        if config.num_prompts is not None:
            self.test_dataset = self.test_dataset.select(
                range(min(config.num_prompts, len(self.test_dataset)))
            )

        self.inputs_map = {
            inp.prompt: inp
            for inp in evaluation_lib.read_prompt_list(str(IFEVAL_INPUT_PATH))
        }

        print(f"[IFEvalStrategy] k_values={config.k_values}, n_samples={config.n_samples}, "
              f"strict={config.strict}, num_prompts={len(self.test_dataset)}")

    @property
    def n_samples(self) -> int:
        return self._n_samples

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

            encoded_batch = tokenizer(
                responses, add_special_tokens=False, padding=False, truncation=False,
            )
            response_token_lengths.extend(len(ids) for ids in encoded_batch["input_ids"])

            eval_input = self.inputs_map[prompt]
            eval_results = [evaluate_single_response(eval_input, r, self.strict) for r in responses]
            all_results.append(eval_results)

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
        return f"p@{self.stopping_k}"

    def wandb_metrics(self, scores: Dict[str, float]) -> Dict[str, float]:
        metrics = {}
        for k in self.k_values:
            metrics[f"eval/pass_at_{k}"] = scores[f"pass_at_{k}"]
        metrics["eval/avg_response_length_tokens"] = scores.get("avg_response_length_tokens", 0.0)
        return metrics
