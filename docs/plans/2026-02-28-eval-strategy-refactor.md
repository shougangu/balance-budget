# Eval Strategy Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `passk_callback.py` so eval-specific logic (prompts, scoring) is injected via strategy objects, while vLLM infrastructure and stopping logic stay in the callback.

**Architecture:** Composition pattern. The callback takes a `primary_eval: EvalStrategy` that drives stopping/forking plus optional `monitor_evals` logged to wandb only. Each `EvalStrategy` provides test prompts, response scoring, wandb metrics, and a label prefix for checkpoint naming. The callback owns all vLLM lifecycle, LoRA adapter saving, threshold/early_tuple stopping, and checkpointing.

**Tech Stack:** Python, HuggingFace TrainerCallback, vLLM, pydantic, pytest

---

### Task 1: Create `EvalStrategy` ABC and `IFEvalStrategy`

**Files:**
- Create: `tuning/training/eval_strategy.py`
- Create: `tests/test_eval_strategy.py`

**Step 1: Write the failing test**

```python
# tests/test_eval_strategy.py
# ABOUTME: Tests for eval strategy ABC and IFEval implementation.
# ABOUTME: Validates interface conformance and pass@k scoring logic.

import pytest
from unittest.mock import patch, MagicMock
from datasets import Dataset


def test_ifeval_strategy_implements_interface():
    """IFEvalStrategy must implement all EvalStrategy abstract methods."""
    from tuning.training.eval_strategy import IFEvalStrategy, EvalStrategy
    assert issubclass(IFEvalStrategy, EvalStrategy)

    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(
            k_values=[1],
            n_samples=1,
            strict=True,
            num_prompts=1,
            tokenizer=MagicMock(),
        )
        assert hasattr(strategy, "get_test_messages")
        assert hasattr(strategy, "score_responses")
        assert hasattr(strategy, "stopping_metric")
        assert hasattr(strategy, "wandb_metrics")
        assert hasattr(strategy, "label_prefix")


def test_pass_at_k_computation():
    """pass_at_k(n, c, k) should compute correct probabilities."""
    from tuning.training.eval_strategy import pass_at_k
    assert pass_at_k(5, 5, 1) == 1.0
    assert pass_at_k(5, 0, 1) == 0.0
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-6


def test_ifeval_stopping_metric():
    """stopping_metric should return the key for the first k value."""
    from tuning.training.eval_strategy import IFEvalStrategy
    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(
            k_values=[1, 5], n_samples=5, strict=True,
            num_prompts=1, tokenizer=MagicMock(),
        )
        assert strategy.stopping_metric() == "pass_at_1"


def test_ifeval_label_prefix():
    """label_prefix should return 'p@{stopping_k}'."""
    from tuning.training.eval_strategy import IFEvalStrategy
    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(
            k_values=[1], n_samples=1, strict=True,
            num_prompts=1, tokenizer=MagicMock(),
        )
        assert strategy.label_prefix == "p@1"


def test_ifeval_wandb_metrics():
    """wandb_metrics should prefix scores with eval/."""
    from tuning.training.eval_strategy import IFEvalStrategy
    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(
            k_values=[1], n_samples=1, strict=True,
            num_prompts=1, tokenizer=MagicMock(),
        )
        scores = {"pass_at_1": 0.72, "avg_response_length_tokens": 150.0, "num_prompts_evaluated": 10}
        wandb_dict = strategy.wandb_metrics(scores)
        assert "eval/pass_at_1" in wandb_dict
        assert "eval/avg_response_length_tokens" in wandb_dict
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tuning.training.eval_strategy'`

**Step 3: Write `eval_strategy.py`**

Extract from `passk_callback.py`: `pass_at_k()`, `evaluate_single_response()`, IFEval dataset loading, the scoring loop from `evaluate_pass_at_k` (lines 532-561).

```python
# tuning/training/eval_strategy.py
# ABOUTME: ABC for eval strategies injected into the generation eval callback.
# ABOUTME: Includes IFEval pass@k implementation.

import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict

from instruction_following_eval import evaluation_lib
from tuning.data.test_dataset import get_ifeval_test_dataset

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
    def label_prefix(self) -> str:
        """Prefix for checkpoint labels (e.g., 'p@1')."""

    @abstractmethod
    def wandb_metrics(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Format scores for wandb logging."""


class IFEvalStrategy(EvalStrategy):
    """IFEval pass@k evaluation strategy."""

    def __init__(self, k_values: list[int], n_samples: int, strict: bool,
                 num_prompts: int, tokenizer):
        self.k_values = k_values
        self.stopping_k = k_values[0]
        self.n_samples = n_samples
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_strategy.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add tuning/training/eval_strategy.py tests/test_eval_strategy.py
git commit -m "feat: add EvalStrategy ABC and IFEvalStrategy"
```

---

### Task 2: Refactor `PassAtKStoppingCallback` to use `EvalStrategy`

**Files:**
- Modify: `tuning/training/passk_callback.py`
- Modify: `tests/test_eval_strategy.py` (add test)

**Step 1: Write the failing test**

Add to `tests/test_eval_strategy.py`:

```python
def test_callback_accepts_eval_strategy():
    """PassAtKStoppingCallback should accept primary_eval and monitor_evals."""
    from tuning.training.passk_callback import PassAtKStoppingCallback
    import inspect
    sig = inspect.signature(PassAtKStoppingCallback.__init__)
    params = list(sig.parameters.keys())
    assert "primary_eval" in params
    assert "monitor_evals" in params
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_strategy.py::test_callback_accepts_eval_strategy -v`
Expected: FAIL — `primary_eval` not in params

**Step 3: Refactor `passk_callback.py`**

Changes to `__init__`:
- Add params: `primary_eval: EvalStrategy`, `monitor_evals: list[EvalStrategy] = None`
- Remove: `self.test_dataset`, `self.inputs_map`, `self.k_values`, `self.stopping_k`, `self.n_samples`, `self.strict`
- Keep: all vLLM config (`use_persistent_vllm`, `num_inference_gpus`, `vllm_gpu_memory_utilization`, `lora_max_rank`, `base_model_hf`)
- Keep: stopping config (`target_pass_at_k_thresholds`, `early_tuples`)
- Keep: `temperature`, `max_tokens` (vLLM sampling params)
- Store `self.primary_eval = primary_eval`, `self.monitor_evals = monitor_evals or []`

Changes to inference methods:
- `_run_vllm_inference`, `_run_data_parallel_inference`, `_format_outputs`: take an `eval_strategy` param to get prompts from `eval_strategy.get_test_messages()` and `eval_strategy.get_test_prompts()` instead of `self.test_dataset`

Replace `evaluate_pass_at_k(model)` with `_run_eval(model, eval_strategy)`:
- Handles vLLM inference (persistent/ephemeral/data-parallel) — same logic as before
- Calls `eval_strategy.score_responses(results, self.tokenizer)` for scoring
- Returns scores dict

Changes to `on_evaluate`:
- Run `_run_eval(model, self.primary_eval)` → check thresholds → checkpoint if needed
- For each `self.monitor_evals`: run `_run_eval(model, eval)` → log to wandb only

Changes to `_save_sweetspot_checkpoint`:
- Absolute threshold label: `f"{self.primary_eval.label_prefix}-{threshold}"`
- Early tuple label: `f"{self.primary_eval.label_prefix}-{patience}@{min_increase}"`

Remove from file:
- `pass_at_k()` function (moved to `eval_strategy.py`)
- `evaluate_single_response()` function (moved to `eval_strategy.py`)
- IFEval imports (`evaluation_lib`, `get_ifeval_test_dataset`)
- `IFEVAL_INPUT_PATH`, `BASE_DIR` constants

Keep in file:
- `partition_prompts()` (used by data-parallel inference)
- `_data_parallel_worker()` (subprocess worker)
- All vLLM lifecycle methods
- All stopping/checkpointing logic

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_strategy.py -v`
Expected: PASS (all 6 tests)

**Step 5: Commit**

```bash
git add tuning/training/passk_callback.py tests/test_eval_strategy.py
git commit -m "refactor: callback takes EvalStrategy for prompts and scoring"
```

---

### Task 3: Update call sites in `sft_training.py` and `dpo_training.py`

**Files:**
- Modify: `tuning/training/sft_training.py` (lines 40-47)
- Modify: `tuning/training/dpo_training.py` (lines 48-55)
- Modify: `tests/test_eval_strategy.py` (add test)

**Step 1: Write the failing test**

```python
def test_sft_training_imports_eval_strategy():
    """sft_training should import from eval_strategy."""
    import ast
    with open("tuning/training/sft_training.py") as f:
        source = f.read()
    tree = ast.parse(source)
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
        and "eval_strategy" in node.module
    ]
    assert len(imports) > 0, "sft_training.py should import from eval_strategy"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_strategy.py::test_sft_training_imports_eval_strategy -v`
Expected: FAIL

**Step 3: Update call sites**

In both files, change:

```python
passk_callback = PassAtKStoppingCallback(
    config=passk_config,
    tokenizer=tokenizer,
    model_name=run_config.model_name,
    base_model_hf=...,
)
```

To:

```python
from tuning.training.eval_strategy import IFEvalStrategy
ifeval_strategy = IFEvalStrategy(
    k_values=passk_config.k_values,
    n_samples=passk_config.n_samples,
    strict=passk_config.strict,
    num_prompts=passk_config.num_prompts,
    tokenizer=tokenizer,
)
passk_callback = PassAtKStoppingCallback(
    config=passk_config,
    tokenizer=tokenizer,
    model_name=run_config.model_name,
    base_model_hf=...,
    primary_eval=ifeval_strategy,
)
```

Note: `passk_config` still has `k_values`, `n_samples`, `strict`, `num_prompts` at this point — they're read here for constructing the strategy. They'll be moved to a separate config in Task 4.

**Step 4: Run tests**

Run: `pytest tests/test_eval_strategy.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add tuning/training/sft_training.py tuning/training/dpo_training.py tests/test_eval_strategy.py
git commit -m "refactor: sft/dpo training construct IFEvalStrategy"
```

---

### Task 4: Split `PassAtKConfig` — move eval fields to `IFEvalConfig`

**Files:**
- Modify: `tuning/training/config_training.py`
- Modify: `tuning/training/unified_early_pipeline.py` (CLI arg construction)
- Modify: `tuning/training/sft_training.py`
- Modify: `tuning/training/dpo_training.py`
- Modify: `tests/test_eval_strategy.py`

**Step 1: Write the failing test**

```python
def test_passk_config_no_eval_fields():
    """PassAtKConfig should not have eval-specific fields."""
    from tuning.training.config_training import PassAtKConfig
    config = PassAtKConfig()
    assert not hasattr(config, "strict"), "strict should be on IFEvalConfig"
    assert not hasattr(config, "k_values"), "k_values should be on IFEvalConfig"
    assert not hasattr(config, "n_samples"), "n_samples should be on IFEvalConfig"
    assert not hasattr(config, "num_prompts"), "num_prompts should be on IFEvalConfig"


def test_ifeval_config_exists():
    """IFEvalConfig should exist with the eval-specific fields."""
    from tuning.training.config_training import IFEvalConfig
    config = IFEvalConfig()
    assert hasattr(config, "k_values")
    assert hasattr(config, "n_samples")
    assert hasattr(config, "strict")
    assert hasattr(config, "num_prompts")
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_strategy.py::test_passk_config_no_eval_fields -v`
Expected: FAIL — `strict` still on PassAtKConfig

**Step 3: Update configs**

In `config_training.py`, create `IFEvalConfig` and remove those fields from `PassAtKConfig`:

```python
class IFEvalConfig(BaseModel):
    """Configuration for IFEval pass@k evaluation."""
    k_values: list[int] = [1]
    n_samples: int = 1
    num_prompts: int = 541
    strict: bool = True
```

`PassAtKConfig` keeps: `target_pass_at_k`, `early_tuples`, `temperature`, `max_tokens`, `enabled`, `use_persistent_vllm`, `vllm_gpu_memory_utilization`, `num_inference_gpus`.

Update `sft_training.py` and `dpo_training.py` to accept `ifeval_config: IFEvalConfig = None` and construct `IFEvalStrategy` from it.

Update `unified_early_pipeline.py`: split `_sft_passk_config` / `_dpo_passk_config` to also return an `IFEvalConfig`, and thread it through to the training functions.

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tuning/training/config_training.py tuning/training/sft_training.py tuning/training/dpo_training.py tuning/training/unified_early_pipeline.py tests/test_eval_strategy.py
git commit -m "refactor: split IFEvalConfig from PassAtKConfig"
```

---

### Task 5: Rename `PassAtKStoppingCallback` → `GenerationEvalCallback`

**Files:**
- Rename: `tuning/training/passk_callback.py` → `tuning/training/generation_eval_callback.py`
- Modify: all importers (`sft_training.py`, `dpo_training.py`, `unified_early_pipeline.py`, any pipeline scripts)

**Step 1: Find all importers**

Run: `grep -r "passk_callback\|PassAtKStoppingCallback" tuning/ tests/ --include="*.py" -l`

**Step 2: Rename file and class**

- `passk_callback.py` → `generation_eval_callback.py`
- `PassAtKStoppingCallback` → `GenerationEvalCallback`
- Update ABOUTME comment
- Update all `[PassAtKCallback]` log prefixes to `[GenerationEvalCallback]`

**Step 3: Update all imports**

In each file found in step 1, change:
- `from tuning.training.passk_callback import PassAtKStoppingCallback` → `from tuning.training.generation_eval_callback import GenerationEvalCallback`

**Step 4: Also rename `PassAtKConfig`**

Since the config is no longer pass@k-specific (it holds vLLM + stopping settings), rename to `GenerationEvalConfig`. Update all references.

**Step 5: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add -u
git add tuning/training/generation_eval_callback.py
git commit -m "refactor: rename to GenerationEvalCallback"
```

---

### Task 6: Final cleanup and verify

**Step 1: Check for dead imports/references**

Run: `grep -r "evaluate_single_response\|pass_at_k\|IFEVAL_INPUT_PATH" tuning/training/generation_eval_callback.py`
Expected: No matches (these are now in `eval_strategy.py`)

**Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All pass

**Step 3: Commit any cleanup**

```bash
git add -u
git commit -m "chore: cleanup dead references after eval strategy refactor"
```
