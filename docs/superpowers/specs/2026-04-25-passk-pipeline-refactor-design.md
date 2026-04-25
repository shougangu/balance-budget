# Pass@K Callback + Unified Early Pipeline Refactor — Design

**Date:** 2026-04-25
**Scope:** `tuning/training/passk_callback.py` (724 lines) and `tuning/training/unified_early_pipeline.py` (971 lines)
**Approach:** One spec, two implementation plans (passk_callback first, unified_early_pipeline second).

## Goals

1. Split two over-long files into responsibility-focused modules under two new subpackages (`passk/` and `pipeline/`).
2. Aggressive logic cleanup of three known-messy hot spots:
   - `_run_eval_with_results` 4-way conditional ladder.
   - `on_evaluate` (~115 lines mixing eval, logging, threshold, early-tuple, and gap-checkpoint logic).
   - `run_dpo` / `run_grpo` near-duplicate post-training scaffolding.
3. End state: each new file has a single clear responsibility; `on_evaluate` becomes a short orchestrator; the persistent→ephemeral fallback no longer duplicates the ephemeral path.

## Non-goals

- `passk_pipeline.py`, `passk_early_pipeline.py`, `ppl_pipeline.py`, `perplexity_pipeline.py` — separate scripts, not in scope.
- Internals of `sft_training.py`, `dpo_training.py`, `grpo_training.py`, `kto_training.py` — they consume the callback but are not refactored here.
- Switching from `print(...)` to Python's `logging` module — separate project. The cleanup goal is consolidating the `[PassAtKCallback]` prefix into one place, not changing the mechanism.
- `MODEL_TO_GPU_*` table contents — copied as-is into the new module.

## Final file layout

```
tuning/training/
├── passk_callback.py                      # one-line shim: re-exports PassAtKStoppingCallback
├── passk/
│   ├── __init__.py
│   ├── callback.py                        # PassAtKStoppingCallback — slim orchestrator
│   ├── runners.py                         # VLLMRunner strategy classes
│   ├── data_parallel.py                   # _data_parallel_worker + partition_prompts
│   ├── logging.py                         # _log_eval_metrics + _log_raw_generation_table
│   └── decisions.py                       # CheckpointDecisionEngine
├── unified_early_pipeline.py              # thin entry point (~15 lines, slurm calls this)
└── pipeline/
    ├── __init__.py
    ├── cli.py                             # _parse_args, parse_early_tuple, GPU/tier maps,
    │                                      # init_cuda_env, is_worker_mode, _init_seeds,
    │                                      # effective_eval_seed, _resolve_simplerl_dataset
    ├── eval_components.py                 # _build_eval_components, _build_monitor_evals,
    │                                      # _sft_ppl_config, _dpo_ppl_config, _sft_tags,
    │                                      # post_training_tags
    ├── checkpoint_metadata.py             # load_checkpoints, claim_next_checkpoint,
    │                                      # mark_completed, _update_row, etc.
    ├── stages.py                          # run_sft, run_post_training, _build_reward_funcs
    └── orchestrator.py                    # main, _build_base_cmd,
                                           # _submit_sbatch_worker, _dispatch_parallel_workers
```

## Public surface

**Stable (one re-export shim, intentional):**
- `from tuning.training.passk_callback import PassAtKStoppingCallback` — production code uses this heavily (`dpo_training.py`, `grpo_training.py`, `passk_early_pipeline.py`, `scripts/live_lm_comparison.py`). The shim keeps that path working.

**Changes (tests updated, no shims):**
- `_init_seeds`, `effective_eval_seed`, `_parse_args` → import from `tuning.training.pipeline.cli`.
- `partition_prompts`, `_data_parallel_worker` → import from `tuning.training.passk.data_parallel`.
- `_update_row`, `parse_metadata_from_output`, etc. → import from `tuning.training.pipeline.checkpoint_metadata`.

These were leading-underscore private helpers; we don't ship back-compat shims for them.

## Import-order carve-outs

Two documented exceptions to "imports at the top of the file":

1. **Entry point dynamic unsloth import.** `unified_early_pipeline.py` keeps `import unsloth` inside the worker-mode-and-not-grpo branch, before any transformers/peft is loaded. Comment in code documents this.
2. **Lazy `stages` import in orchestrator.** `orchestrator.main()` imports `stages` lazily inside the worker-mode branch. Without this, pure-orchestrator runs (which only dispatch sbatch and never train) would pull in transformers/peft via `stages.py`'s top-level imports, *and* those imports could fire before the entry point's unsloth gate. Comment in code documents this.

`pipeline/cli.py` must not transitively import unsloth/torch/transformers — verified during implementation.

## passk_callback logic decomposition

### VLLMRunner strategy (`passk/runners.py`)

Replaces the 4-way `if/elif` ladder in `_run_eval_with_results` (current passk_callback.py:553-594) with a runner selected once at callback init.

```python
class VLLMRunner:
    """Knows how to: (optionally) save adapter, (optionally) offload model,
    run inference, restore."""
    def run(self, model, eval_strategy, save_adapter_to: str | None) -> list[dict]: ...

class ExternalVLLMRunner(VLLMRunner):     # uses externally-provided LLM, no adapter save
class PersistentVLLMRunner(VLLMRunner):   # holds persistent LLM, swaps LoRA each call
class EphemeralVLLMRunner(VLLMRunner):    # creates+destroys LLM each call, offloads training model
class DataParallelVLLMRunner(VLLMRunner): # spawns N subprocess workers, offloads training model
```

The "offload training model to CPU → run → restore" pattern (currently duplicated in 3 branches) lives in a single `_with_model_offloaded(model)` context manager used by `Ephemeral` and `DataParallel`. `Persistent` doesn't offload (training model stays on GPU 0; vLLM uses its own GPU memory budget).

The persistent→ephemeral fallback no longer duplicates the ephemeral path. If `PersistentVLLMRunner.run` raises on first attempt, the callback logs a warning and replaces `self._runner` with an `EphemeralVLLMRunner` instance, then retries via the standard path. Same observable behavior, one less code path.

`_run_eval_with_results` collapses to:

```python
def _run_eval_with_results(self, model, eval_strategy):
    with tempfile.TemporaryDirectory() as adapter_dir:
        adapter_path = self._save_adapter_if_needed(model, adapter_dir)
        model_results = self._runner.run(model, eval_strategy, adapter_path)
    scores = eval_strategy.score_responses(model_results, self.tokenizer)
    return scores, model_results
```

### CheckpointDecisionEngine (`passk/decisions.py`)

Pure logic, no `model`/`wandb`/`print` dependencies. Owns the threshold list, early-tuple list, and gap-checkpoint counter.

```python
@dataclass
class CheckpointDecision:
    label: str             # e.g. "0.7", "2@0.02", "gap-12000-0.42"
    advances_state: bool   # whether this consumes the gap counter

class CheckpointDecisionEngine:
    def __init__(self, target_thresholds, early_tuples, max_checkpoint_gap): ...
    def decide(self, primary_metric, history, data_points_seen,
               last_checkpoint_data_points) -> list[CheckpointDecision]:
        """Returns decisions in order. Mutates internal state (trims thresholds,
        pops triggered early_tuples) so subsequent calls reflect what's left."""
```

The callback no longer holds `target_pass_at_k_thresholds` / `early_tuples` / `max_checkpoint_gap` directly — they become engine state.

### Logging consolidation (`passk/logging.py`)

One method `_log_eval_metrics(eval_strategy, scores, raw_results, state)` does the W&B `log` dict + raw-generations table call + score-summary print that `on_evaluate` currently repeats for primary and monitor evals. The `[PassAtKCallback]` prefix lives in one place. `print` stays as the mechanism.

### `on_evaluate` (`passk/callback.py`)

```python
def on_evaluate(self, args, state, control, model=None, **kwargs):
    model = model or kwargs.get("model")
    if model is None:
        print("[PassAtKCallback] Warning: model is None, skipping eval")
        return control

    data_points_seen = self._compute_data_points_seen(args, state)

    primary_scores, primary_metric = self._eval_and_log(model, self.primary_eval, state)
    self._primary_metric_history.append(primary_metric)

    for monitor in self.monitor_evals:
        self._eval_and_log(model, monitor, state)

    decisions = self._decision_engine.decide(
        primary_metric=primary_metric,
        history=self._primary_metric_history,
        data_points_seen=data_points_seen,
        last_checkpoint_data_points=self._last_checkpoint_data_points,
    )
    for decision in decisions:
        self._save_sweetspot_checkpoint(model, decision.label, state, args)
        if decision.advances_state:
            self._last_checkpoint_data_points = data_points_seen

    self._last_eval_step = state.global_step
    return control
```

`_eval_and_log` runs `_run_eval_with_results`, calls the logging helper, prints the score summary — eliminating the cut-and-paste between primary and monitor branches.

### Renames

- `self.prevResults` → `self._primary_metric_history` (private, PEP8). Mechanical.

## unified_early_pipeline logic decomposition

### Shared post-training helper (`pipeline/stages.py`)

`run_dpo` and `run_grpo` collapse into one `run_post_training(args, method)` that owns the shared shape:

```python
def run_post_training(args, method: Literal["dpo", "grpo"]):
    metadata_file = args.metadata_file[0]
    checkpoint = claim_next_checkpoint(metadata_file)
    if checkpoint is None:
        print("No checkpoints available to claim, nothing to do.")
        sys.exit(42)

    train_size = _resolve_remaining_budget(args, method, checkpoint)
    if train_size <= 0:
        print(f"Skipping {checkpoint['checkpoint_path']}: no data budget remaining")
        mark_completed(metadata_file, checkpoint["checkpoint_path"])
        return

    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)

    configs = _build_post_training_configs(args, method, checkpoint, train_size)
    passk_config, primary_eval, monitor_evals = _build_eval_components(
        args, method, configs.gpu_util
    )
    ppl_config = _dpo_ppl_config(args) if method == "dpo" else None
    tags = post_training_tags(method, checkpoint, primary_eval, passk_config, ppl_config)

    with wandb.init(name=configs.run_config.model_name, project=args.wandb_project,
                    job_type=method, tags=tags, config=...):
        _train_dispatch(method, configs, passk_config, primary_eval,
                        monitor_evals, ppl_config, checkpoint)

    mark_completed(metadata_file, checkpoint["checkpoint_path"])
```

Method-specific bits as small builders:

- `_resolve_remaining_budget(args, method, checkpoint)` — fixed-split vs dynamic budget logic, parameterized by method (current `dpo_size` / `grpo_size` paths).
- `_build_post_training_configs(args, method, checkpoint, train_size)` — returns a small dataclass with `dataset_config`, `sft_run_config`, `run_config`, `lora_config`, `model_load_config`, `training_args`, `gpu_util`. DPO and GRPO branches diverge here on training-config type (`DPOTrainingConfig` vs `GRPOTrainingConfig` with extra GRPO fields).
- `_train_dispatch(method, ...)` — calls `train_model_dpo` or `train_model_grpo` with method-specific kwargs (~10 lines each branch). DPO-only `perplexity_test_dataset` setup stays inline in this dispatcher.

`run_sft` stays as its own function — it doesn't claim/mark checkpoints, doesn't build sft-from-checkpoint configs, has its own tag rules. Forcing it into the same shape would over-abstract.

### Tag building consolidation (`pipeline/eval_components.py`)

```python
def post_training_tags(method, checkpoint, primary_eval, passk_config,
                       ppl_config=None) -> list[str]: ...
```

Single definition replaces copy-pasted tag blocks in `run_dpo` / `run_grpo`. `_sft_tags` stays in this module — its rules differ (no checkpoint values).

### Bug fix: `SBATCH_WORKER_SCRIPT` mutation

Current `unified_early_pipeline.py:276` does `SBATCH_WORKER_SCRIPT = "..."` inside `_parse_args` without `global`, so `--short` silently does not switch sbatch scripts. Fix during the move into `cli.py`: resolve and attach as `args.sbatch_script`; the orchestrator reads from there.

## Tests

### Existing tests kept green

| Test file | What changes |
|---|---|
| `test_passk_callback_wandb_tables.py` | Import path stays (re-exported) |
| `test_callback_step_bridging.py` | Import path stays |
| `test_external_vllm_reuse.py` | Import path stays |
| `test_eval_strategy.py` | Import path stays |
| `test_multi_gpu_inference.py` | `partition_prompts` → `tuning.training.passk.data_parallel` |
| `test_seed_wiring.py` | `_data_parallel_worker` → `tuning.training.passk.data_parallel` |
| `test_passk_early_data_chat_templating.py` | Verify and update if needed |
| `test_unified_early_pipeline.py` | Many imports → `tuning.training.pipeline.cli` and `.checkpoint_metadata` |
| `test_simplerl_rlvr.py` | Imports → `pipeline.cli` |
| `test_grpo_config.py` | `_parse_args` → `pipeline.cli` |

### New unit tests

- **`tests/test_checkpoint_decision_engine.py`** — pure-logic tests for `CheckpointDecisionEngine`. Threshold crossing (descending sort, sweep-down), early-tuple triggering (patience window, min-increase), gap-checkpoint (only when no other decision fired), state advancement. ~6-8 small tests covering cases currently buried inside `on_evaluate`.
- **`tests/test_vllm_runners.py`** — behavior tests for the runner strategy seam (no actual vLLM init). Verifies: `_run_eval_with_results` selects the right runner from config; persistent→ephemeral fallback swaps the runner without re-running inference twice. Heavier vLLM init is left to existing `test_external_vllm_reuse.py` and `test_multi_gpu_inference.py`.
- **`tests/test_post_training_runner.py`** — small test that `run_post_training(args, "dpo")` and `(args, "grpo")` produce the right `wandb.init` kwargs and `train_model_*` call args. Heavy mocking is fine here — goal is verifying dispatch shape.

## Implementation plans

### Plan 1 — passk_callback refactor (lands first)

1. Create `passk/` subpackage skeleton.
2. Move `partition_prompts` + `_data_parallel_worker` to `data_parallel.py`. Update test imports. Run.
3. Extract `VLLMRunner` strategy into `runners.py` with all four runners. Replace ladder. Run.
4. Extract `CheckpointDecisionEngine` into `decisions.py`. Replace inline threshold/early-tuple/gap logic. Add unit tests. Run.
5. Extract logging helpers into `logging.py`. Slim `on_evaluate`. Run.
6. Convert `passk_callback.py` to one-line shim. Run full test suite + smoke pipeline run.

### Plan 2 — unified_early_pipeline refactor (after Plan 1 merged)

1. Create `pipeline/` subpackage skeleton.
2. Move `cli.py` (parsing, seeds, GPU maps, simplerl resolver, CUDA env init). Update test imports. Run.
3. Move `checkpoint_metadata.py`. Update test imports. Run.
4. Move `eval_components.py` (incl. new `post_training_tags`). Run.
5. Extract `run_post_training` shared helper, port `run_dpo` and `run_grpo` to it. Run + new test.
6. Move `stages.py` (incl. `run_sft`, `run_post_training`, `_build_reward_funcs`). Run.
7. Move `orchestrator.py` (incl. `main`, lazy `stages` import). Convert `unified_early_pipeline.py` to thin entry point. Fix the `SBATCH_WORKER_SCRIPT` mutation bug. Run + smoke pipeline run.

Each step ends green before the next starts. Each plan ends with a real smoke pipeline run, not just unit tests, since some import-order issues (the unsloth gate, the lazy stages import) only surface in real worker mode.

## Risk and mitigation

- **Import-order regression** — the unsloth gate is load-bearing; pure-orchestrator runs must not transitively pull transformers/peft. Mitigation: smoke run at the end of each plan; explicit assertion in implementation that `pipeline.cli` does not transitively import torch/unsloth.
- **Persistent vLLM fallback behavior change** — the new path swaps the runner instance and retries; current path runs the ephemeral code inline. Observable behavior is identical (warning printed, ephemeral inference happens). Verified via the new `test_vllm_runners.py` fallback test.
- **Hidden test imports** — searches above caught the explicit imports, but there may be additional reach-ins. Each plan's first step runs the full test suite to surface any miss.
