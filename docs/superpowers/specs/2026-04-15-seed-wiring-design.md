# Seed Wiring for Unified Early Pipeline

## Problem

The `--seed` CLI argument in `unified_early_pipeline.py` is defined but never read.
All training uses hardcoded `seed=42`, and pass@k evals have no seed control at all.
This prevents reproducibility studies and seed-based variance measurement.

## Goals

- **Reproducibility**: same `--seed` value produces identical training trajectories and eval results
- **Variance measurement**: varying `--seed` across runs measures training + eval variance

## CLI Interface

Two arguments on `unified_early_pipeline.py`:

- `--seed` (existing, default=42) — master seed driving training RNG, LoRA init, data sampling, and eval generation
- `--eval_seed` (new, default=None) — when set, overrides only the eval (pass@k) seed; when None, evals use `--seed`

Effective eval seed: `eval_seed if eval_seed is not None else seed`.

Default behavior is unchanged — runs without `--seed` still get 42 everywhere.

## Seed Propagation

### Training

- `config_training.py` `TrainingArgumentsConfig.to_hf_args()` — currently hardcodes `d["seed"] = 42`. Change to accept seed as a parameter and use that instead of the hardcoded value.
- `config_training.py` `LoraConfig.random_state` — currently hardcodes `42`. Accept seed from the pipeline.
- `unified_early_pipeline.py` — thread `args.seed` into `TrainingArgumentsConfig` and `LoraConfig` construction so it flows into SFT, DPO, and GRPO stages.

HuggingFace Trainer handles internal seeding (torch, numpy, python random) from the seed it receives.

### Data Sampling

Three files currently do module-level `random.seed(42)`:
- `data/utils.py:4`
- `data/hf_dataset.py:12`
- `data/test_dataset.py:7`

These run at import time, before `args.seed` is parsed.

Approach: Remove the module-level `random.seed(42)` calls. Have the pipeline call `random.seed(args.seed)` once early in `unified_early_pipeline.py`'s main function, before any data loading happens. Data loading is sequential and happens early, so global seed state is sufficient.

### Evals (Pass@K)

- `config_inference.py` — add `seed: Optional[int] = None` field to `VLLMSamplingParamsConfig`. When None, vLLM uses its default (stochastic). When set, passed through to `SamplingParams`.
- `passk_callback.py` — the callback receives the effective eval seed and injects it into the inference config before building `SamplingParams`. This affects both the data-parallel worker path (`_data_parallel_worker`) and the persistent vLLM path. All workers get the same seed.
- `unified_early_pipeline.py` — compute `effective_eval_seed = args.eval_seed if args.eval_seed is not None else args.seed` and pass it to the eval callback setup.

One seed value is used for all prompts within a single eval batch. vLLM handles per-request RNG isolation internally.

### W&B Logging

Log both `seed` and `eval_seed` (the effective value, not None) to W&B run config alongside existing config logging. Enables filtering/grouping runs by seed in the dashboard.

## Change Summary

| Component | Current | Proposed |
|---|---|---|
| CLI | `--seed` defined, ignored | `--seed` (default=42) drives everything; new `--eval_seed` optionally overrides eval RNG |
| Training RNG | Hardcoded 42 in `config_training.py` | Accepts seed from pipeline |
| LoRA init | Hardcoded 42 in `LoraConfig` | Accepts seed from pipeline |
| Data sampling | Module-level `random.seed(42)` in 3 files | Removed; pipeline sets `random.seed(args.seed)` once before data loading |
| Eval (pass@k) | No seed in `VLLMSamplingParamsConfig` | New `seed` field, populated with effective eval seed |
| W&B | Seed not logged | Both `seed` and `eval_seed` logged to run config |
