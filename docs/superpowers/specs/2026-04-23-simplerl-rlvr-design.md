# SimpleRL-Zoo RLVR Dataset Integration

**Date:** 2026-04-23  
**Status:** Approved

## Summary

Add the `hkust-nlp/SimpleRL-Zoo-Data` dataset to the RLVR/GRPO pipeline as four `--dataset` choices: `simplerl-easy`, `simplerl-medium`, `simplerl-hard`, and `simplerl` (auto-selects tier by base model strength). Reuses the existing `math500_reward_func` and `SYSTEM_MESSAGE_OPENMATH`/`COMPMATH_STRING` formatting — no new reward function or system prompt.

## Dataset Facts

| CLI id | HF subset | Train rows | Test rows | Source |
|---|---|---|---|---|
| `simplerl-easy` | `simplelr_abel_gsm8k_level1` | 8,388 | ~3k | GSM8K + MATH L1 |
| `simplerl-medium` | `simplelr_abel_level1to4` | 8,139 | ~3k | MATH L1–4 |
| `simplerl-hard` | `simplelr_abel_level3to5` | 8,523 | ~3k | MATH L3–5 |

- `reward_model.ground_truth` is the bare final answer in all three subsets — used as `reference_answer`.
- `extra_info.question` is the raw problem text — wrapped with `COMPMATH_STRING` + `SYSTEM_MESSAGE_OPENMATH`.
- `prompt` column is ignored; we rebuild with our own chat template.
- `extra_info.level` is unreliable (stale, often hardcoded to 1); tier is encoded by subdirectory.
- The `target` column is inconsistent across subsets (full solution in GSM8K, bare answer in MATH); `reward_model.ground_truth` is consistent and correct.

## Components

### New: `tuning/data/simplerl_rlvr.py`

Single class `SimpleRLRLVR(difficulty: str)` extending `HFDataset`.

```
SIMPLERL_TIERS = {
    "easy":   "simplelr_abel_gsm8k_level1",
    "medium": "simplelr_abel_level1to4",
    "hard":   "simplelr_abel_level3to5",
}
```

`load_from_huggingface(hf_path)` downloads both `{subset}/train.parquet` and `{subset}/test.parquet` via `hf_hub_download`, loads them with pandas, and assembles a `DatasetDict{train, test}` — same pattern as `ifrlvr_rlvr.py`.

`_get_rows(dataset)` iterates rows, extracts `extra_info["question"]` and `reward_model["ground_truth"]`, deduplicates by formatted user content (same as `openmath_rlvr.py`), and emits:

```python
{
  "prompt": [
    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
    {"role": "user",   "content": COMPMATH_STRING.format(problem=question)},
  ],
  "reference_answer": ground_truth,
}
```

`format_dataset()` calls `_get_rows` on both train and test independently, then reassembles `DatasetDict{train, test}` from the two row lists directly (no `train_test_split` — the HF repo ships its own test split, which we preserve).

`__main__` iterates all three tiers, saves each to disk:
- `rlvr-simplerl-easy`
- `rlvr-simplerl-medium`
- `rlvr-simplerl-hard`

No `rlvr-simplerl` artifact — the alias resolves before `get_train_dataset` runs.

### Changes to `tuning/training/unified_early_pipeline.py`

**1. `MODEL_TO_SIMPLERL_TIER` map** (near top, alongside `MODEL_TO_GPU_1`):

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

All default to `"medium"` until Shougan fills in the model-strength conditions.

**2. `_resolve_simplerl_dataset(args)` helper**:

```python
def _resolve_simplerl_dataset(args):
    if args.dataset == "simplerl":
        tier = MODEL_TO_SIMPLERL_TIER[args.model]
        print(f"[simplerl] {args.model} -> simplerl-{tier}")
        args.dataset = f"simplerl-{tier}"
```

Called at the top of `run_sft`, `run_dpo`, `run_grpo` (before `DatasetConfig` is built in each).

**3. `--dataset` choices** (line ~130):

```python
choices=["tuluif", "gsm8k", "openmath", "ifrlvr",
         "simplerl", "simplerl-easy", "simplerl-medium", "simplerl-hard"]
```

**4. `_build_reward_funcs`** (line ~692):

```python
elif args.dataset in {"simplerl", "simplerl-easy", "simplerl-medium", "simplerl-hard"}:
    return [math500_reward_func]
```

`"simplerl"` is in the set as a guard; in practice it is always resolved to a concrete tier before this runs.

## Data Flow

```
CLI --dataset simplerl
  └─ _resolve_simplerl_dataset(args)  ← maps model → tier, rewrites args.dataset
       └─ args.dataset = "simplerl-medium"  (default stub)

run_grpo / run_sft / run_dpo
  └─ DatasetConfig(dataset="simplerl-medium", dataset_type="rlvr", train_size=N)
       └─ get_train_dataset()
            └─ loads from disk: rlvr-simplerl-medium-N
                 (built by python -m tuning.data.simplerl_rlvr)

GRPO reward: math500_reward_func
  └─ math-verify + #### fallback (same as openmath)
```

## Testing

**Dataset build verification** (`__main__`):
- Run `python -m tuning.data.simplerl_rlvr`
- Assert three dirs exist: `rlvr-simplerl-{easy,medium,hard}`
- Confirm train row counts: 8388 / 8139 / 8523
- Confirm test row counts are nonzero
- Spot-check one example row from each tier (printed by existing pattern)

**Alias resolution unit test** (`tests/`):
- Construct `Namespace(dataset="simplerl", model="llama3-8B")`
- Call `_resolve_simplerl_dataset(args)`
- Assert `args.dataset == "simplerl-medium"`

## Out of Scope

- No new eval strategy or task-name entry (`--task-name math500` already covers this).
- No combined `simplerl` disk artifact.
- The `qwen`-prefix HF subsets are ignored — we rebuild the prompt ourselves.
- Model-strength mapping logic in `MODEL_TO_SIMPLERL_TIER` is a stub; Shougan fills it in.
