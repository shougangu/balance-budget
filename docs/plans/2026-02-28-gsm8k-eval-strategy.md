# GSM8K EvalStrategy & Pipeline Generalization

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable the unified_early_pipeline to train and evaluate on GSM8K (and future tasks) instead of being hardcoded to IFEval.

**Architecture:** Implement `GSM8KEvalStrategy` (parallel to `IFEvalStrategy`) with direct constructor params — no eval config classes (`IFEvalConfig` is deleted, `GSM8KConfig` is not created). A single composer function in the pipeline builds `PassAtKConfig` + eval strategies together, dispatching on `--task-name`. The `PassAtKStoppingCallback` is already task-agnostic — it accepts any `EvalStrategy` — so no callback changes needed.

**Tech Stack:** Python, HuggingFace datasets, pytest. GSM8K scoring follows the lm-evaluation-harness approach (strict regex + flexible fallback + normalization). No new dependencies.

---

## Key Context

- `EvalStrategy` ABC lives in `tuning/training/eval_strategy.py` — has 7 abstract members
- `IFEvalStrategy` is the only implementation; we're adding `GSM8KEvalStrategy`
- `PassAtKStoppingCallback` in `tuning/training/passk_callback.py` is already generic — takes `primary_eval: EvalStrategy`
- `sft_training.py` and `dpo_training.py` currently hardcode `IFEvalStrategy` — these need to accept any strategy
- `unified_early_pipeline.py` has separate builder functions (`_sft_passk_config`, `_sft_ifeval_config`, etc.) — these become a single composer function dispatching on `--task-name`
- GSM8K test data: the standard GSM8K test set from HuggingFace (1319 problems with reference answers in `#### <number>` format)
- GSM8K system message and prompt format already exist in `tuning/data/config.py`
- `sft-gsm8k` base dataset must be pre-processed before training (script exists in `tuning/data/gsm8k_sft.py`)

---

### Task 1: GSM8K answer extraction and scoring (matching lm-eval-harness)

The core scoring logic for GSM8K, matching the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) approach exactly.

lm-eval uses two extraction strategies and a normalization pipeline:
- **Strict extraction**: regex `#### (\-?[0-9\.\,]+)` — number right after `####`
- **Flexible fallback**: regex `(-?[$0-9.,]{2,})|(-?[0-9]+)` with last-match semantics — for when `####` is missing
- **Normalization** (applied to both extracted and reference before comparison): strip commas, `$`, everything before `#### `, trailing periods
- **Comparison**: case-insensitive exact match

**Files:**
- Create: `tuning/evaluation/gsm8k_scoring.py`
- Test: `tests/test_gsm8k_scoring.py`

**Step 1: Write failing tests for answer extraction**

```python
# tests/test_gsm8k_scoring.py

# ABOUTME: Tests for GSM8K answer extraction and scoring (lm-eval-harness compatible).
# ABOUTME: Covers strict/flexible extraction, normalization, and edge cases.

import pytest
from tuning.evaluation.gsm8k_scoring import (
    extract_gsm8k_answer_strict,
    extract_gsm8k_answer_flexible,
    normalize_answer,
    is_correct,
)


class TestStrictExtraction:
    """Strict extraction: regex '#### (\\-?[0-9\\.\\,]+)' — first match."""

    def test_standard_format(self):
        assert extract_gsm8k_answer_strict("Step 1: blah\n#### 42") == "42"

    def test_with_commas(self):
        assert extract_gsm8k_answer_strict("Step 1: blah\n#### 1,234") == "1,234"

    def test_negative_number(self):
        assert extract_gsm8k_answer_strict("Step 1: blah\n#### -7") == "-7"

    def test_decimal(self):
        assert extract_gsm8k_answer_strict("Step 1: blah\n#### 3.14") == "3.14"

    def test_no_delimiter_returns_none(self):
        assert extract_gsm8k_answer_strict("The answer is 42") is None

    def test_no_number_after_delimiter_returns_none(self):
        assert extract_gsm8k_answer_strict("#### hello") is None

    def test_multiple_delimiters_takes_first(self):
        assert extract_gsm8k_answer_strict("#### 10\n#### 20") == "10"


class TestFlexibleExtraction:
    """Flexible extraction: regex '(-?[$0-9.,]{2,})|(-?[0-9]+)' — last match."""

    def test_number_without_delimiter(self):
        assert extract_gsm8k_answer_flexible("The answer is 42") == "42"

    def test_dollar_amount(self):
        assert extract_gsm8k_answer_flexible("The total cost is $1,234") == "$1,234"

    def test_takes_last_match(self):
        assert extract_gsm8k_answer_flexible("First 10, then 20, finally 30") == "30"

    def test_negative_number(self):
        assert extract_gsm8k_answer_flexible("The result is -5") == "-5"

    def test_no_number_returns_none(self):
        assert extract_gsm8k_answer_flexible("No numbers here") is None

    def test_single_digit(self):
        assert extract_gsm8k_answer_flexible("Answer: 7") == "7"


class TestNormalizeAnswer:
    """Normalization: strip commas, $, '#### ' prefix, trailing period."""

    def test_strip_commas(self):
        assert normalize_answer("1,234,567") == "1234567"

    def test_strip_dollar_sign(self):
        assert normalize_answer("$100") == "100"

    def test_strip_hash_prefix(self):
        assert normalize_answer("some text #### 42") == "42"

    def test_strip_trailing_period(self):
        assert normalize_answer("42.") == "42"

    def test_strip_all_combined(self):
        assert normalize_answer("blah #### $1,000.") == "1000"

    def test_case_insensitive(self):
        assert normalize_answer("ABC") == "abc"

    def test_decimal_not_stripped(self):
        assert normalize_answer("3.14") == "3.14"


class TestIsCorrect:
    """Full pipeline: extract + normalize + compare."""

    def test_strict_match(self):
        assert is_correct("Step 1: blah\n#### 42", "42") is True

    def test_strict_mismatch(self):
        assert is_correct("Step 1: blah\n#### 42", "43") is False

    def test_no_answer_is_incorrect(self):
        assert is_correct("I don't know", "42") is False

    def test_comma_normalization(self):
        assert is_correct("#### 1,000", "1000") is True

    def test_reference_with_hash_prefix(self):
        # GSM8K references often have "#### " prefix
        assert is_correct("#### 42", "#### 42") is True

    def test_flexible_fallback(self):
        # No #### delimiter, but number present — flexible extraction kicks in
        assert is_correct("The answer is 42", "42") is True

    def test_dollar_sign_in_both(self):
        assert is_correct("#### $100", "$100") is True

    def test_trailing_period(self):
        assert is_correct("#### 42.", "42") is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gsm8k_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tuning.evaluation.gsm8k_scoring'`

**Step 3: Implement gsm8k_scoring.py**

```python
# tuning/evaluation/gsm8k_scoring.py

# ABOUTME: GSM8K answer extraction and scoring, matching lm-evaluation-harness.
# ABOUTME: Uses strict (#### regex) then flexible (last-number) extraction with normalization.

import re
from typing import Optional

# lm-eval strict extraction: number right after ####
STRICT_PATTERN = re.compile(r"####\s*(\-?[0-9\.\,]+)")

# lm-eval flexible extraction: last number-like token (including $ amounts)
FLEXIBLE_PATTERN = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

# lm-eval normalization regexes (applied to both extracted and reference)
NORMALIZE_REGEXES = [
    re.compile(r","),           # strip commas
    re.compile(r"\$"),          # strip dollar signs
    re.compile(r"(?s).*#### "), # strip everything before "#### "
    re.compile(r"\.$"),         # strip trailing period
]


def extract_gsm8k_answer_strict(response: str) -> Optional[str]:
    """Strict extraction: first match of '#### <number>' pattern."""
    match = STRICT_PATTERN.search(response)
    return match.group(1) if match else None


def extract_gsm8k_answer_flexible(response: str) -> Optional[str]:
    """Flexible extraction: last number-like token in the response."""
    matches = FLEXIBLE_PATTERN.findall(response)
    if not matches:
        return None
    # findall with alternation returns tuples; pick the non-empty group
    last_match = matches[-1]
    return last_match[0] if last_match[0] else last_match[1]


def normalize_answer(answer: str) -> str:
    """Normalize an answer string using lm-eval's regex pipeline."""
    result = answer.lower()
    for pattern in NORMALIZE_REGEXES:
        result = pattern.sub("", result)
    return result.strip()


def is_correct(response: str, reference: str) -> bool:
    """Check if a response matches the reference using lm-eval's approach.

    Tries strict extraction first (#### pattern), falls back to flexible
    (last number). Normalizes both sides before case-insensitive comparison.
    """
    extracted = extract_gsm8k_answer_strict(response)
    if extracted is None:
        extracted = extract_gsm8k_answer_flexible(response)
    if extracted is None:
        return False
    return normalize_answer(extracted) == normalize_answer(reference)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gsm8k_scoring.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tuning/evaluation/gsm8k_scoring.py tests/test_gsm8k_scoring.py
git commit -m "feat: add GSM8K scoring matching lm-eval-harness approach"
```

---

### Task 2: GSM8K test dataset loader

Create `get_gsm8k_test_dataset()` parallel to the existing `get_ifeval_test_dataset()` in `tuning/data/test_dataset.py`. Returns a HuggingFace Dataset with `messages`, `prompt`, and `reference_answer` columns.

**Files:**
- Modify: `tuning/data/test_dataset.py` (add `get_gsm8k_test_dataset`)
- Test: `tests/test_gsm8k_test_dataset.py`

**Step 1: Write failing tests**

```python
# tests/test_gsm8k_test_dataset.py

# ABOUTME: Tests for GSM8K test dataset loader.
# ABOUTME: Validates dataset structure: messages, prompt, and reference_answer columns.

import pytest
from tuning.data.test_dataset import get_gsm8k_test_dataset


class TestGetGSM8KTestDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        return get_gsm8k_test_dataset()

    def test_has_required_columns(self, dataset):
        assert "messages" in dataset.column_names
        assert "prompt" in dataset.column_names
        assert "reference_answer" in dataset.column_names

    def test_messages_have_system_and_user(self, dataset):
        msgs = dataset[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_prompt_is_string(self, dataset):
        assert isinstance(dataset[0]["prompt"], str)
        assert len(dataset[0]["prompt"]) > 0

    def test_reference_answer_is_string(self, dataset):
        assert isinstance(dataset[0]["reference_answer"], str)
        assert len(dataset[0]["reference_answer"]) > 0

    def test_dataset_has_rows(self, dataset):
        assert len(dataset) > 0

    def test_num_prompts_subset(self):
        dataset = get_gsm8k_test_dataset(num_prompts=10)
        assert len(dataset) == 10
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gsm8k_test_dataset.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_gsm8k_test_dataset'`

**Step 3: Implement get_gsm8k_test_dataset**

Add to `tuning/data/test_dataset.py`:

```python
from tuning.data.config import SYSTEM_MESSAGE_GSM8K, GSM8K_STRING

def get_gsm8k_test_dataset(num_prompts=None):
    """Load GSM8K test set with messages, prompt, and reference_answer columns."""
    from datasets import load_dataset
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")

    messages_list = []
    prompts = []
    reference_answers = []

    for row in gsm8k:
        question = row["question"]
        # Reference answer is the number after ####
        answer_text = row["answer"]
        # Extract just the number from "...#### <number>"
        ref_answer = answer_text.split("####")[-1].strip()

        prompt = GSM8K_STRING.format(question=question)
        messages_list.append([
            {"role": "system", "content": SYSTEM_MESSAGE_GSM8K},
            {"role": "user", "content": prompt},
        ])
        prompts.append(prompt)
        reference_answers.append(ref_answer)

    dataset = Dataset.from_dict({
        "messages": messages_list,
        "prompt": prompts,
        "reference_answer": reference_answers,
    })

    if num_prompts is not None:
        dataset = dataset.select(range(min(num_prompts, len(dataset))))

    return dataset
```

Note: The existing `from datasets import Dataset` import is already at the top of `test_dataset.py`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gsm8k_test_dataset.py -v`
Expected: All PASS (first run may take a moment to download the GSM8K dataset from HuggingFace)

**Step 5: Commit**

```bash
git add tuning/data/test_dataset.py tests/test_gsm8k_test_dataset.py
git commit -m "feat: add GSM8K test dataset loader"
```

---

### Task 3: GSM8KEvalStrategy, refactor strategy constructors, delete IFEvalConfig

Delete `IFEvalConfig` from `config_training.py` — eval config classes are unnecessary indirection. Strategy constructors take their params directly (`k_values`, `n_samples`, `num_prompts`, plus task-specific params like `strict` for IFEval). Remove `tokenizer` from constructors (only needed in `score_responses` where it's already a parameter).

Both strategies follow identical pass@k conventions: per-prompt boolean arrays fed into `pass_at_k()`, same multi-k constructor params.

**Files:**
- Modify: `tuning/training/config_training.py` (delete `IFEvalConfig`)
- Modify: `tuning/training/eval_strategy.py` (refactor `IFEvalStrategy`, add `GSM8KEvalStrategy`)
- Test: `tests/test_eval_strategy.py` (update IFEval tests, add GSM8K tests, delete obsolete config tests)

**Step 1: Write failing tests**

Update existing tests and add GSM8K tests in `tests/test_eval_strategy.py`:

- Remove the `from tuning.training.config_training import IFEvalConfig` import
- **Delete** `test_ifeval_config_exists` entirely (IFEvalConfig no longer exists)
- Update all `IFEvalStrategy(config=IFEvalConfig(...), tokenizer=MagicMock())` calls to `IFEvalStrategy(k_values=..., n_samples=..., num_prompts=..., strict=...)`
- **Rewrite** `test_ifeval_strategy_accepts_config` → `test_ifeval_strategy_accepts_direct_params`
- **Update** `test_passk_config_no_eval_fields` assertion messages (reference strategy constructors, not IFEvalConfig)

```python
# Rewritten test (replaces test_ifeval_strategy_accepts_config):
def test_ifeval_strategy_accepts_direct_params():
    """IFEvalStrategy should accept params directly (no config object)."""
    from tuning.training.eval_strategy import IFEvalStrategy
    with patch("tuning.training.eval_strategy.get_ifeval_test_dataset") as mock_dataset, \
         patch("tuning.training.eval_strategy.evaluation_lib") as mock_eval_lib:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
        })
        mock_eval_lib.read_prompt_list.return_value = []
        strategy = IFEvalStrategy(k_values=[1, 5], n_samples=5, num_prompts=1, strict=True)
        assert strategy.k_values == [1, 5]
        assert strategy.n_samples == 5
        assert strategy.strict is True
        assert strategy.stopping_metric() == "pass_at_1"


# Updated assertion messages:
def test_passk_config_no_eval_fields():
    """PassAtKConfig should not have eval-specific fields (those belong on strategy constructors)."""
    from tuning.training.config_training import PassAtKConfig
    config = PassAtKConfig()
    assert not hasattr(config, "strict"), "strict belongs on eval strategy constructor"
    assert not hasattr(config, "k_values"), "k_values belongs on eval strategy constructor"
    assert not hasattr(config, "n_samples"), "n_samples belongs on eval strategy constructor"
    assert not hasattr(config, "num_prompts"), "num_prompts belongs on eval strategy constructor"


# New GSM8K tests (same structure as IFEval tests):

def test_gsm8k_strategy_implements_interface():
    """GSM8KEvalStrategy must implement all EvalStrategy abstract methods."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy, EvalStrategy
    assert issubclass(GSM8KEvalStrategy, EvalStrategy)
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "reference_answer": ["42"],
        })
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert hasattr(strategy, "get_test_messages")
        assert hasattr(strategy, "score_responses")
        assert hasattr(strategy, "stopping_metric")
        assert hasattr(strategy, "wandb_metrics")
        assert hasattr(strategy, "label_prefix")


def test_gsm8k_stopping_metric():
    """GSM8K stopping metric should follow pass@k convention: 'pass_at_{k}'."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "reference_answer": ["42"],
        })
        strategy = GSM8KEvalStrategy(k_values=[1, 5], n_samples=5, num_prompts=1)
        assert strategy.stopping_metric() == "pass_at_1"


def test_gsm8k_label_prefix():
    """GSM8K label_prefix should be 'gsm8k-p@{stopping_k}'."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "reference_answer": ["42"],
        })
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert strategy.label_prefix == "gsm8k-p@1"


def test_gsm8k_score_responses_pass_at_k():
    """score_responses should compute pass@k per-prompt, mirroring IFEvalStrategy."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "Q1"}], [{"role": "user", "content": "Q2"}]],
            "prompt": ["Q1", "Q2"],
            "reference_answer": ["42", "100"],
        })
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda texts, **kw: {"input_ids": [[0] * 10] * len(texts)}
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=2)
        results = [
            {"prompt": "Q1", "responses": ["Step 1: ...\n#### 42"]},      # correct
            {"prompt": "Q2", "responses": ["Step 1: ...\n#### 99"]},      # incorrect
        ]
        scores = strategy.score_responses(results, tokenizer)
        assert scores["pass_at_1"] == 0.5
        assert scores["num_prompts_evaluated"] == 2


def test_gsm8k_score_responses_flexible_fallback():
    """score_responses should use flexible extraction when #### is missing."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "Q1"}]],
            "prompt": ["Q1"],
            "reference_answer": ["42"],
        })
        tokenizer = MagicMock()
        tokenizer.side_effect = lambda texts, **kw: {"input_ids": [[0] * 10] * len(texts)}
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=1)
        results = [
            {"prompt": "Q1", "responses": ["The answer is 42"]},  # no ####, flexible kicks in
        ]
        scores = strategy.score_responses(results, tokenizer)
        assert scores["pass_at_1"] == 1.0


def test_gsm8k_wandb_metrics():
    """wandb_metrics should prefix scores with eval/gsm8k_."""
    from tuning.training.eval_strategy import GSM8KEvalStrategy
    with patch("tuning.training.eval_strategy.get_gsm8k_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "reference_answer": ["42"],
        })
        strategy = GSM8KEvalStrategy(k_values=[1], n_samples=1, num_prompts=1)
        scores = {"pass_at_1": 0.75, "num_prompts_evaluated": 10, "avg_response_length_tokens": 50.0}
        wandb_dict = strategy.wandb_metrics(scores)
        assert "eval/gsm8k_pass_at_1" in wandb_dict
        assert "eval/avg_response_length_tokens" in wandb_dict
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_strategy.py -v`
Expected: FAIL — existing IFEval tests fail (tokenizer removal, config import removal), GSM8K tests fail (no GSM8KEvalStrategy yet)

**Step 3a: Delete IFEvalConfig from config_training.py**

Remove the entire `IFEvalConfig` class from `tuning/training/config_training.py`.

**Step 3b: Refactor IFEvalStrategy to take direct params**

In `tuning/training/eval_strategy.py`, change `IFEvalStrategy.__init__` from:
```python
    def __init__(self, config: IFEvalConfig, tokenizer):
        self.k_values = config.k_values
        self.stopping_k = config.k_values[0]
        self._n_samples = config.n_samples
        self.strict = config.strict
        ...
```
To:
```python
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
        ...
```

Remove the `IFEvalConfig` import from eval_strategy.py.

**Step 3c: Add GSM8KEvalStrategy**

Add imports and class after `IFEvalStrategy` in `tuning/training/eval_strategy.py`:

```python
from tuning.data.test_dataset import get_gsm8k_test_dataset
from tuning.evaluation.gsm8k_scoring import is_correct as gsm8k_is_correct


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

            # Per-prompt boolean array: is each sample correct?
            eval_results = [gsm8k_is_correct(r, ref) for r in responses]
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
        return f"gsm8k-p@{self.stopping_k}"

    def wandb_metrics(self, scores: Dict[str, float]) -> Dict[str, float]:
        metrics = {}
        for k in self.k_values:
            key = f"pass_at_{k}"
            if key in scores:
                metrics[f"eval/gsm8k_{key}"] = scores[key]
        metrics["eval/avg_response_length_tokens"] = scores.get("avg_response_length_tokens", 0.0)
        return metrics
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_strategy.py -v`
Expected: All PASS (both updated IFEval tests and new GSM8K tests)

**Step 5: Commit**

```bash
git add tuning/training/config_training.py tuning/training/eval_strategy.py tests/test_eval_strategy.py
git commit -m "feat: add GSM8KEvalStrategy, delete IFEvalConfig, strategies take direct params"
```

---

### Task 4: Trainers accept pre-built eval strategies directly

Replace `ifeval_config` param with `primary_eval` and `monitor_evals` in both `sft_training.py` and `dpo_training.py`. The trainers become fully task-agnostic — they never construct strategies, never import eval types, and never know about IFEval or GSM8K. Strategy construction moves entirely to the pipeline (Task 5).

**Files:**
- Modify: `tuning/training/sft_training.py` (replace `ifeval_config` with `primary_eval` + `monitor_evals`)
- Modify: `tuning/training/dpo_training.py` (same)
- Test: `tests/test_eval_strategy.py`

**Step 1: Write failing tests**

```python
def test_sft_training_accepts_primary_eval():
    """train_model_sft should accept primary_eval and monitor_evals, not ifeval_config."""
    import inspect
    from tuning.training.sft_training import train_model_sft
    sig = inspect.signature(train_model_sft)
    assert "primary_eval" in sig.parameters
    assert "monitor_evals" in sig.parameters
    assert "ifeval_config" not in sig.parameters


def test_dpo_training_accepts_primary_eval():
    """train_model_dpo should accept primary_eval and monitor_evals, not ifeval_config."""
    import inspect
    from tuning.training.dpo_training import train_model_dpo
    sig = inspect.signature(train_model_dpo)
    assert "primary_eval" in sig.parameters
    assert "monitor_evals" in sig.parameters
    assert "ifeval_config" not in sig.parameters
```

Also:
- **Delete** `test_sft_training_imports_eval_strategy` (sft_training no longer imports eval_strategy directly — it receives pre-built strategies)

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_eval_strategy.py::test_sft_training_accepts_primary_eval tests/test_eval_strategy.py::test_dpo_training_accepts_primary_eval -v`
Expected: FAIL — `ifeval_config` is still in params, `primary_eval` is not

**Step 3a: Modify sft_training.py**

Change signature from:
```python
def train_model_sft(
    run_config: SFTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: TrainingArgumentsConfig = None,
    perplexity_config = None,
    passk_config = None,
    ifeval_config = None,
):
```

To:
```python
def train_model_sft(
    run_config: SFTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: TrainingArgumentsConfig = None,
    perplexity_config = None,
    passk_config = None,
    primary_eval = None,
    monitor_evals = None,
):
```

Change callback construction from:
```python
    if passk_config is not None and passk_config.enabled:
        ifeval_strategy = IFEvalStrategy(
            config=ifeval_config or IFEvalConfig(),
            tokenizer=tokenizer,
        )
        passk_callback = PassAtKStoppingCallback(
            config=passk_config,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
            base_model_hf=run_config.model_name_hf,
            primary_eval=ifeval_strategy,
        )
```

To:
```python
    if passk_config is not None and passk_config.enabled:
        passk_callback = PassAtKStoppingCallback(
            config=passk_config,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
            base_model_hf=run_config.model_name_hf,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals or [],
        )
```

Remove the `IFEvalStrategy` and `IFEvalConfig` imports from `sft_training.py` (no longer needed).

**Step 3b: Apply the same changes to dpo_training.py**

Same pattern: replace `ifeval_config` with `primary_eval` + `monitor_evals`, remove strategy construction, remove IFEval imports.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_eval_strategy.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tuning/training/sft_training.py tuning/training/dpo_training.py tests/test_eval_strategy.py
git commit -m "refactor: trainers accept pre-built eval strategies, drop eval configs"
```

---

### Task 5: Pipeline composer function builds all eval components

Replace the 4 separate builder functions (`_sft_passk_config`, `_dpo_passk_config`, `_sft_ifeval_config`, `_dpo_ifeval_config`) with a single composer function `_build_eval_components` that returns a `(passk_config, primary_eval, monitor_evals)` tuple. Since strategy constructors no longer need a tokenizer (removed in Task 3), the pipeline can build strategies before calling trainers.

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py`
- Test: `tests/test_unified_early_pipeline.py`

**Step 1: Write failing tests**

Add to `tests/test_unified_early_pipeline.py`:

```python
class TestTaskNameDispatch:
    def test_gsm8k_task_name_accepted(self):
        args = _parse_args(REQUIRED + ["--task-name", "gsm8k"])
        assert args.task_name == "gsm8k"

    def test_default_task_name_is_ifeval(self):
        args = _parse_args(REQUIRED)
        assert args.task_name == "ifeval"
```

**Step 2: Run tests to verify current state**

Run: `pytest tests/test_unified_early_pipeline.py -v -k "task_name"`

**Step 3: Modify the pipeline**

1. **Add the composer function** (replaces `_sft_passk_config`, `_dpo_passk_config`, `_sft_ifeval_config`, `_dpo_ifeval_config`):

```python
def _build_eval_components(args, stage, gpu_util):
    """Build PassAtKConfig + eval strategies for the given task and stage.

    Returns (passk_config, primary_eval, monitor_evals).
    All three are None/[] if pass@k is disabled for this stage.
    """
    prefix = stage  # "sft" or "dpo"
    if not getattr(args, f"{prefix}_enable_passk", False):
        return None, None, []

    from tuning.training.config_training import PassAtKConfig
    passk_config = PassAtKConfig(
        target_pass_at_k=getattr(args, f"{prefix}_passk_targets"),
        early_tuples=getattr(args, f"{prefix}_passk_early") or None,
        temperature=getattr(args, f"{prefix}_passk_temperature"),
        enabled=True,
        num_inference_gpus=getattr(args, f"{prefix}_passk_num_inference_gpus"),
        use_persistent_vllm=getattr(args, f"{prefix}_passk_persistent_vllm"),
        vllm_gpu_memory_utilization=gpu_util,
    )

    k_values = getattr(args, f"{prefix}_passk_k_values", [1])
    n_samples = getattr(args, f"{prefix}_passk_n_samples", 1)
    num_prompts = getattr(args, f"{prefix}_passk_num_prompts", None)

    if args.task_name == "ifeval":
        from tuning.training.eval_strategy import IFEvalStrategy
        strict = getattr(args, f"{prefix}_passk_strict", True)
        primary_eval = IFEvalStrategy(
            k_values=k_values, n_samples=n_samples,
            num_prompts=num_prompts or 541, strict=strict,
        )
    elif args.task_name == "gsm8k":
        from tuning.training.eval_strategy import GSM8KEvalStrategy
        primary_eval = GSM8KEvalStrategy(
            k_values=k_values, n_samples=n_samples,
            num_prompts=num_prompts,
        )
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")

    return passk_config, primary_eval, []
```

2. **Delete** `_sft_passk_config`, `_dpo_passk_config`, `_sft_ifeval_config`, `_dpo_ifeval_config`.

3. **Update `_sft_tags`**: Change from taking `ifeval_config` to taking `primary_eval`. Get `k_val` from `primary_eval.stopping_k` instead of `ifeval_config.k_values[0]`:

```python
def _sft_tags(passk_config, ppl_config, primary_eval=None):
    tags = ["sft"]
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        tags.append(f"p{k_val}")
        ...
```

4. **Update `run_sft`**: Replace separate builder calls with composer:

```python
    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "sft", gpu_util)
    ppl_config = _sft_ppl_config(args)
    tags = _sft_tags(passk_config, ppl_config, primary_eval)
    ...
        model, tokenizer, trainer, callbacks = train_model_sft(
            ...,
            passk_config=passk_config,
            perplexity_config=ppl_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
        )
```

5. **Update `run_dpo`**: Same pattern — composer + pass to trainer:

```python
    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "dpo", gpu_util)
    ppl_config = _dpo_ppl_config(args)

    tags = ["dpo", str(checkpoint["threshold_value"]), str(checkpoint["data_points_seen"])]
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        tags.append(f"p{k_val}")
    ...
        train_model_dpo(
            ...,
            passk_config=passk_config,
            perplexity_config=ppl_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
        )
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "refactor: composer function builds eval components, replaces separate builders"
```

---

### Task 6: Pre-process sft-gsm8k dataset

The `sft-gsm8k` base dataset doesn't exist on disk yet. Run the existing processing script to create it.

**Files:**
- Run: `tuning/data/gsm8k_sft.py`

**Step 1: Verify the dataset doesn't exist**

```bash
ls tuning/data/datasets/sft-gsm8k* 2>/dev/null || echo "No sft-gsm8k dataset found"
```

**Step 2: Run the processing script**

```bash
cd /project/6105902/shougan/balance-budget
python -m tuning.data.gsm8k_sft
```

**Step 3: Verify the dataset was created**

```bash
ls tuning/data/datasets/sft-gsm8k/
```

Expected: `dataset_dict.json`, `train/`, `test/`

**Step 4: No commit needed** (data files are not tracked in git)

---

### Task 7: End-to-end smoke test (manual, requires GPU)

Verify the full pipeline works with GSM8K on a small model.

**No code changes — this is a validation step.**

```bash
# Allocate GPU
f1

# Run a small GSM8K training with pass@k callback
python tuning/training/unified_early_pipeline.py \
    --model llama3-1B \
    --wandb-project tuning \
    --dataset gsm8k \
    --task-name gsm8k \
    --train-size 256 \
    --sft-passk-targets 1.2 \
    --sft-passk-num-prompts 50 \
    --sft-passk-n-samples 1 \
    --run-sft
```

**Verify:**
- SFT trains on `sft-gsm8k-256` dataset
- PassAtKCallback uses GSM8KEvalStrategy
- `pass_at_k` metrics are logged to wandb (for each k in k_values)
- Checkpoints are saved with `threshold_type: "pass_at_{stopping_k}"`

---

## Summary of Changes

| File | Change |
|------|--------|
| `tuning/evaluation/gsm8k_scoring.py` | **NEW** — answer extraction and correctness checking |
| `tuning/data/test_dataset.py` | Add `get_gsm8k_test_dataset()` |
| `tuning/training/config_training.py` | Delete `IFEvalConfig` (no eval config classes) |
| `tuning/training/eval_strategy.py` | Refactor `IFEvalStrategy` to take direct params, add `GSM8KEvalStrategy` |
| `tuning/training/sft_training.py` | Replace `ifeval_config` with `primary_eval` + `monitor_evals`, remove IFEval imports |
| `tuning/training/dpo_training.py` | Same changes as sft_training |
| `tuning/training/unified_early_pipeline.py` | Single `_build_eval_components` composer replaces 4 builder functions |
| `tests/test_gsm8k_scoring.py` | **NEW** — scoring tests |
| `tests/test_gsm8k_test_dataset.py` | **NEW** — dataset loader tests |
| `tests/test_eval_strategy.py` | Update IFEval tests (direct params), add GSM8K tests, delete obsolete config tests |
| `tests/test_unified_early_pipeline.py` | Add task-name dispatch tests |

