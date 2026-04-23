# SimpleRL-Zoo RLVR Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `hkust-nlp/SimpleRL-Zoo-Data` dataset as four `--dataset` choices (`simplerl`, `simplerl-easy`, `simplerl-medium`, `simplerl-hard`) in the RLVR/GRPO pipeline.

**Architecture:** New data loader `simplerl_rlvr.py` downloads the three pre-split tiers from HuggingFace, extracts raw question + ground-truth, and wraps them with existing OpenMath chat formatting. The `simplerl` alias auto-resolves to a concrete tier based on `args.model` via a stub map in `unified_early_pipeline.py`. All tiers reuse `math500_reward_func`.

**Tech Stack:** HuggingFace datasets, `hf_hub_download`, pandas, pytest

**Spec:** `docs/superpowers/specs/2026-04-23-simplerl-rlvr-design.md`

---

### File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `tuning/data/simplerl_rlvr.py` | Dataset loader: downloads HF parquet, extracts questions, formats prompts, saves to disk |
| Modify | `tuning/training/unified_early_pipeline.py` | CLI choices, alias resolver, reward function mapping |
| Create | `tests/test_simplerl_rlvr.py` | Tests for dataset loader + alias resolver + CLI parsing |

---

### Task 1: Alias resolver and CLI wiring

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:39-65` (add `MODEL_TO_SIMPLERL_TIER` map after `MODEL_TO_GPU_3`)
- Modify: `tuning/training/unified_early_pipeline.py:130` (extend `--dataset` choices)
- Modify: `tuning/training/unified_early_pipeline.py:692-705` (`_build_reward_funcs`)
- Test: `tests/test_simplerl_rlvr.py`

- [ ] **Step 1: Write failing tests for alias resolver and CLI parsing**

Create `tests/test_simplerl_rlvr.py`:

```python
# ABOUTME: Tests for SimpleRL-Zoo RLVR dataset integration.
# ABOUTME: Covers alias resolution, CLI parsing of simplerl dataset choices, and reward function dispatch.

import argparse
import pytest

from tuning.training.unified_early_pipeline import (
    _parse_args,
    _resolve_simplerl_dataset,
    _build_reward_funcs,
    MODEL_TO_SIMPLERL_TIER,
)


REQUIRED = ["--model", "llama3-3B", "--wandb-project", "tuning"]


class TestResolveSimplerlDataset:
    def test_rewrites_simplerl_to_concrete_tier(self):
        args = argparse.Namespace(dataset="simplerl", model="llama3-8B")
        _resolve_simplerl_dataset(args)
        assert args.dataset == "simplerl-medium"

    def test_leaves_concrete_tier_unchanged(self):
        args = argparse.Namespace(dataset="simplerl-hard", model="llama3-8B")
        _resolve_simplerl_dataset(args)
        assert args.dataset == "simplerl-hard"

    def test_leaves_unrelated_dataset_unchanged(self):
        args = argparse.Namespace(dataset="gsm8k", model="llama3-8B")
        _resolve_simplerl_dataset(args)
        assert args.dataset == "gsm8k"

    def test_all_models_have_tier_mapping(self):
        expected_models = {"llama3-1B", "llama3-3B", "llama3-8B",
                           "qwen2-2B", "qwen2-3B", "qwen2-7B"}
        assert set(MODEL_TO_SIMPLERL_TIER.keys()) == expected_models


class TestParseArgsSimplerl:
    def test_simplerl_accepted(self):
        args = _parse_args(REQUIRED + ["--dataset", "simplerl"])
        assert args.dataset == "simplerl"

    def test_simplerl_easy_accepted(self):
        args = _parse_args(REQUIRED + ["--dataset", "simplerl-easy"])
        assert args.dataset == "simplerl-easy"

    def test_simplerl_medium_accepted(self):
        args = _parse_args(REQUIRED + ["--dataset", "simplerl-medium"])
        assert args.dataset == "simplerl-medium"

    def test_simplerl_hard_accepted(self):
        args = _parse_args(REQUIRED + ["--dataset", "simplerl-hard"])
        assert args.dataset == "simplerl-hard"


class TestBuildRewardFuncsSimplerl:
    def test_simplerl_easy_uses_math500_reward(self):
        from tuning.training.reward_functions import math500_reward_func
        args = argparse.Namespace(dataset="simplerl-easy")
        funcs = _build_reward_funcs(args)
        assert funcs == [math500_reward_func]

    def test_simplerl_medium_uses_math500_reward(self):
        from tuning.training.reward_functions import math500_reward_func
        args = argparse.Namespace(dataset="simplerl-medium")
        funcs = _build_reward_funcs(args)
        assert funcs == [math500_reward_func]

    def test_simplerl_hard_uses_math500_reward(self):
        from tuning.training.reward_functions import math500_reward_func
        args = argparse.Namespace(dataset="simplerl-hard")
        funcs = _build_reward_funcs(args)
        assert funcs == [math500_reward_func]

    def test_simplerl_alias_uses_math500_reward(self):
        from tuning.training.reward_functions import math500_reward_func
        args = argparse.Namespace(dataset="simplerl")
        funcs = _build_reward_funcs(args)
        assert funcs == [math500_reward_func]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simplerl_rlvr.py -v`

Expected: ImportError for `_resolve_simplerl_dataset` and `MODEL_TO_SIMPLERL_TIER` (not yet defined).

- [ ] **Step 3: Add `MODEL_TO_SIMPLERL_TIER` map to `unified_early_pipeline.py`**

After the `MODEL_TO_GPU_3` block (line ~64), add:

```python
MODEL_TO_SIMPLERL_TIER = {
    "llama3-1B": "medium",
    "llama3-3B": "medium",
    "llama3-8B": "medium",
    "qwen2-2B":  "medium",
    "qwen2-3B":  "medium",
    "qwen2-7B":  "medium",
}
```

- [ ] **Step 4: Add `_resolve_simplerl_dataset` helper to `unified_early_pipeline.py`**

After the `effective_eval_seed` function (line ~78), add:

```python
def _resolve_simplerl_dataset(args):
    """Rewrite args.dataset='simplerl' to a concrete tier based on model strength."""
    if args.dataset == "simplerl":
        tier = MODEL_TO_SIMPLERL_TIER[args.model]
        print(f"[simplerl] {args.model} -> simplerl-{tier}")
        args.dataset = f"simplerl-{tier}"
```

- [ ] **Step 5: Extend `--dataset` choices (line ~130)**

Change:

```python
parser.add_argument("--dataset", default="gsm8k", choices=["tuluif", "gsm8k", "openmath", "ifrlvr"],)
```

To:

```python
parser.add_argument("--dataset", default="gsm8k",
                    choices=["tuluif", "gsm8k", "openmath", "ifrlvr",
                             "simplerl", "simplerl-easy", "simplerl-medium", "simplerl-hard"],)
```

- [ ] **Step 6: Extend `_build_reward_funcs` (line ~692)**

After the `elif args.dataset == "ifrlvr":` block, add:

```python
    elif args.dataset in {"simplerl", "simplerl-easy", "simplerl-medium", "simplerl-hard"}:
        return [math500_reward_func]
```

- [ ] **Step 7: Wire `_resolve_simplerl_dataset` into `run_sft`, `run_dpo`, `run_grpo`**

Add `_resolve_simplerl_dataset(args)` as the first line inside each of these three functions, before `_init_seeds(args)`:

In `run_sft` (line ~391):
```python
def run_sft(args):
    _resolve_simplerl_dataset(args)
    ...
```

In `run_dpo` (line ~574):
```python
def run_dpo(args):
    ...
    _resolve_simplerl_dataset(args)
    ...
```
(After `claim_next_checkpoint` but before `DatasetConfig` construction.)

In `run_grpo` (line ~708):
```python
def run_grpo(args):
    ...
    _resolve_simplerl_dataset(args)
    ...
```
(After `claim_next_checkpoint` but before `DatasetConfig` construction.)

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_simplerl_rlvr.py -v`

Expected: All 12 tests PASS.

- [ ] **Step 9: Run existing pipeline tests to check for regressions**

Run: `python -m pytest tests/test_unified_early_pipeline.py -v`

Expected: All existing tests still PASS.

- [ ] **Step 10: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_simplerl_rlvr.py
git commit -m "feat: add simplerl dataset choices, alias resolver, and reward mapping to pipeline CLI"
```

---

### Task 2: Dataset loader

**Files:**
- Create: `tuning/data/simplerl_rlvr.py`
- Modify: `tests/test_simplerl_rlvr.py` (add dataset loader tests)

- [ ] **Step 1: Write failing tests for dataset loader**

Append to `tests/test_simplerl_rlvr.py`:

```python
from tuning.data.simplerl_rlvr import SimpleRLRLVR, SIMPLERL_TIERS


class TestSimpleRLRLVR:
    @pytest.fixture(params=["easy", "medium", "hard"])
    def loaded_dataset(self, request):
        difficulty = request.param
        ds = SimpleRLRLVR(difficulty)
        ds.load_from_huggingface("hkust-nlp/SimpleRL-Zoo-Data")
        ds.format_dataset()
        return ds.get_dataset(), difficulty

    def test_has_train_and_test_splits(self, loaded_dataset):
        dataset, _ = loaded_dataset
        assert "train" in dataset
        assert "test" in dataset

    def test_train_has_prompt_and_reference_answer(self, loaded_dataset):
        dataset, _ = loaded_dataset
        train = dataset["train"]
        assert "prompt" in train.column_names
        assert "reference_answer" in train.column_names

    def test_prompt_is_system_user_pair(self, loaded_dataset):
        dataset, _ = loaded_dataset
        row = dataset["train"][0]
        assert isinstance(row["prompt"], list)
        assert len(row["prompt"]) == 2
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1]["role"] == "user"

    def test_reference_answer_is_nonempty_string(self, loaded_dataset):
        dataset, _ = loaded_dataset
        row = dataset["train"][0]
        assert isinstance(row["reference_answer"], str)
        assert len(row["reference_answer"]) > 0

    def test_prompts_are_deduplicated(self, loaded_dataset):
        dataset, _ = loaded_dataset
        prompts = dataset["train"]["prompt"]
        user_texts = [p[1]["content"] for p in prompts]
        assert len(user_texts) == len(set(user_texts))

    def test_uses_openmath_system_message(self, loaded_dataset):
        from tuning.data.config import SYSTEM_MESSAGE_OPENMATH
        dataset, _ = loaded_dataset
        row = dataset["train"][0]
        assert row["prompt"][0]["content"] == SYSTEM_MESSAGE_OPENMATH

    def test_uses_compmath_format(self, loaded_dataset):
        dataset, _ = loaded_dataset
        row = dataset["train"][0]
        assert row["prompt"][1]["content"].startswith("Problem:")

    def test_train_size_approximately_8k(self, loaded_dataset):
        dataset, difficulty = loaded_dataset
        train_size = len(dataset["train"])
        assert 7000 <= train_size <= 9000, f"{difficulty} has {train_size} train rows, expected ~8k"

    def test_test_split_is_nonempty(self, loaded_dataset):
        dataset, _ = loaded_dataset
        assert len(dataset["test"]) > 0


class TestSimplerlTiers:
    def test_tiers_dict_has_three_entries(self):
        assert set(SIMPLERL_TIERS.keys()) == {"easy", "medium", "hard"}

    def test_tier_subsets_are_abel_variants(self):
        for tier, subset in SIMPLERL_TIERS.items():
            assert "abel" in subset, f"{tier} subset should use abel variant"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simplerl_rlvr.py::TestSimpleRLRLVR -v --no-header 2>&1 | head -20`

Expected: ImportError for `SimpleRLRLVR` and `SIMPLERL_TIERS`.

- [ ] **Step 3: Create `tuning/data/simplerl_rlvr.py`**

```python
# ABOUTME: SimpleRL-Zoo dataset loader for GRPO/RLVR training with three difficulty tiers.
# ABOUTME: Downloads pre-split parquet from hkust-nlp/SimpleRL-Zoo-Data, applies OpenMath chat formatting.

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
import pandas as pd

from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING

SIMPLERL_TIERS = {
    "easy":   "simplelr_abel_gsm8k_level1",
    "medium": "simplelr_abel_level1to4",
    "hard":   "simplelr_abel_level3to5",
}


class SimpleRLRLVR(HFDataset):
    def __init__(self, difficulty: str):
        if difficulty not in SIMPLERL_TIERS:
            raise ValueError(f"difficulty must be one of {set(SIMPLERL_TIERS)}, got {difficulty!r}")
        self._difficulty = difficulty
        self._subset = SIMPLERL_TIERS[difficulty]
        super().__init__(dataset_name=f"simplerl-{difficulty}")

    def load_from_huggingface(self, hf_path: str, *args, **kwargs):
        self.hf_path = hf_path
        train_path = hf_hub_download(
            hf_path, f"{self._subset}/train.parquet", repo_type="dataset"
        )
        test_path = hf_hub_download(
            hf_path, f"{self._subset}/test.parquet", repo_type="dataset"
        )
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        self._dataset = DatasetDict({
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        })
        self._raw_dataset = self._dataset

    def _get_rows(self, dataset):
        seen = set()
        rows = []
        for row in dataset:
            question = row["extra_info"]["question"]
            ground_truth = row["reward_model"]["ground_truth"]
            formatted = COMPMATH_STRING.format(problem=question)
            if formatted in seen:
                continue
            seen.add(formatted)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
                    {"role": "user", "content": formatted},
                ],
                "reference_answer": ground_truth,
            })
        return rows

    def format_dataset(self):
        train_rows = self._get_rows(self._dataset["train"])
        test_rows = self._get_rows(self._dataset["test"])
        self._dataset = DatasetDict({
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        })
        print(f"SimpleRL-Zoo ({self._difficulty}) Dataset - {self._dataset}")
        print(f"Example row - {self._dataset['train'][0]}")


if __name__ == "__main__":
    for difficulty in SIMPLERL_TIERS:
        ds = SimpleRLRLVR(difficulty)
        ds.load_from_huggingface("hkust-nlp/SimpleRL-Zoo-Data")
        ds.format_dataset()
        save_name = f"rlvr-simplerl-{difficulty}"
        ds.clear_old_datasets(prefix=save_name)
        ds.save_dataset_to_disk(save_name=save_name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simplerl_rlvr.py -v`

Expected: All tests PASS (both Task 1 and Task 2 tests). The `TestSimpleRLRLVR` fixture downloads from HF on first run — this will take a few seconds.

- [ ] **Step 5: Run all existing tests to check for regressions**

Run: `python -m pytest tests/test_rlvr_datasets.py tests/test_reward_functions.py tests/test_unified_early_pipeline.py -v`

Expected: All existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add tuning/data/simplerl_rlvr.py tests/test_simplerl_rlvr.py
git commit -m "feat: add SimpleRL-Zoo dataset loader with easy/medium/hard difficulty tiers"
```

---

### Task 3: Build disk artifacts and smoke-test end-to-end

**Files:**
- Run: `tuning/data/simplerl_rlvr.py` (`__main__`)

- [ ] **Step 1: Activate the venv and run the dataset builder**

```bash
source /project/6105902/shougan/balance-budget/venv/bin/activate
module load arrow/19.0.1
python -m tuning.data.simplerl_rlvr
```

Expected output (three tiers printed):
```
SimpleRL-Zoo (easy) Dataset - DatasetDict({train: Dataset(..., num_rows=8388), test: Dataset(...)})
Example row - {'prompt': [{'role': 'system', ...}, {'role': 'user', ...}], 'reference_answer': '...'}
...
SimpleRL-Zoo (medium) Dataset - DatasetDict({train: Dataset(..., num_rows=8139), test: Dataset(...)})
...
SimpleRL-Zoo (hard) Dataset - DatasetDict({train: Dataset(..., num_rows=8523), test: Dataset(...)})
```

- [ ] **Step 2: Verify disk artifacts exist with expected sizes**

```bash
python -c "
from datasets import load_from_disk
for tier in ['easy','medium','hard']:
    ds = load_from_disk(f'tuning/data/datasets/rlvr-simplerl-{tier}')
    print(f'rlvr-simplerl-{tier}: train={len(ds[\"train\"])}, test={len(ds[\"test\"])}')
"
```

Expected:
```
rlvr-simplerl-easy: train=8388, test=...
rlvr-simplerl-medium: train=8139, test=...
rlvr-simplerl-hard: train=8523, test=...
```

(Exact test counts depend on dedup in the test split; train counts should match HF.)

- [ ] **Step 3: Spot-check a row from each tier through `get_train_dataset`**

Verify the full loading pipeline works end-to-end (loads from disk artifact, subsets to `--train-size`):

```bash
python -c "
from tuning.training.config_training import DatasetConfig, PTRunConfig
from tuning.data.train_dataset import get_train_dataset
for tier in ['easy','medium','hard']:
    cfg = PTRunConfig(
        dataset_config=DatasetConfig(
            dataset=f'simplerl-{tier}',
            dataset_type='rlvr',
            train_size=100,
        ),
        model_name='llama3-3B',
        model_name_hf='meta-llama/Llama-3.2-3B',
        task_name='math500',
        pft_method='grpo',
        do_training=True,
    )
    ds = get_train_dataset(cfg)
    print(f'simplerl-{tier} subset: train={len(ds[\"train\"])}, test={len(ds[\"test\"])}')
    print(f'  prompt[0]: {ds[\"train\"][0][\"prompt\"][1][\"content\"][:80]}...')
    print(f'  ref_ans:   {ds[\"train\"][0][\"reference_answer\"]}')
"
```

Expected: Each tier loads 100 train rows from its disk artifact, prompt starts with "Problem:", reference_answer is a short string.

- [ ] **Step 4: Commit (no code change — record verification)**

```bash
git commit --allow-empty -m "verify: SimpleRL-Zoo disk artifacts built and end-to-end loading works"
```
