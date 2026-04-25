# DDP for GRPO RLVR Training and Evaluation

**Date:** 2026-04-25
**Status:** Approved

## Summary

Add single-node DDP (DistributedDataParallel) support to GRPO RLVR training and to `PassAtKCallback` evaluation, so training and pass@k inference both use every GPU in the SLURM allocation. Launch GRPO via `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE`. SFT and DPO are out of scope and remain unchanged.

The eval callback uses **cooperative DDP**: each rank serves a slice of the eval prompt set through its local colocated vLLM engine (`trainer.vllm_generation.llm`), results are gathered to rank 0, and rank 0 scores, logs, and saves checkpoints. `world_size=1` is the degenerate case of the same code path — single-GPU GRPO behavior is preserved.

## Decisions Locked

| Decision | Choice | Rationale |
|---|---|---|
| Scope | GRPO training + GRPO eval (`PassAtKCallback`) only | SFT/DPO have a different optimal shape: they fit single-GPU at full batch and benefit from spare-GPU eval workers. DDP buys ~0 there until they're memory-bound. |
| Topology | Single-node, configurable GPU count via `$SLURM_GPUS_ON_NODE` (typically 4 or 8) | User constraint — no cross-node fabric. |
| vLLM mode | Colocate (existing `vllm_mode="colocate"`) | Closer delta from current single-GPU; uses every GPU for training. Server mode deferred until/unless throughput plateaus. |
| Tensor parallelism | `vllm_tensor_parallel_size=1` (per-rank independent vLLM) | Each rank's vLLM is fully self-contained; no TP subgroup coordination needed. |
| Eval pattern | Cooperative DDP (Q4-B): rank-partitioned generation, gather to rank 0 | Explicit user goal: "higher batch sizes with all the gpus together". Linear eval throughput scaling. |
| Launcher | `torchrun` | Open TRL bug ([#2262](https://github.com/huggingface/trl/issues/2262)) reports `accelerate launch` produces NaN gradients vs torchrun's correct training. No accelerate-specific features (FSDP, DeepSpeed, multi-node) are needed here. |
| Compatibility | DDP is the only GRPO path; world_size=1 is the degenerate case | Cleaner long-term — no dual code path. SFT/DPO untouched, so their `_init_cuda_env()`-based spare-GPU eval optimization is preserved. |

## TRL Source-of-Truth References

The Compute Canada-patched TRL 0.29.0+computecanada in `.venv` wraps vLLM in a `VLLMGeneration` class:

| Path | What it does |
|---|---|
| `.venv/.../trl/trainer/grpo_trainer.py:67` | `from ..generation.vllm_generation import VLLMGeneration` |
| `.venv/.../trl/trainer/grpo_trainer.py:713` | `self.vllm_generation = VLLMGeneration(...)` constructed during init |
| `.venv/.../trl/trainer/grpo_trainer.py:1228` | `self.vllm_generation.sync_weights()` — called per training step (gated by `_last_loaded_step`) |
| `.venv/.../trl/generation/vllm_generation.py:370` | `self.llm = LLM(...)` — the raw vLLM engine, accessible as `trainer.vllm_generation.llm` |
| `.venv/.../trl/generation/vllm_generation.py:486-541` | PEFT-aware sync: `merge_adapter()` → `llm_model.load_weights(...)` → `unmerge_adapter()` |

Implication: by the time `on_evaluate` fires, `trainer.vllm_generation.llm` already holds the **merged trained weights** as its in-memory model. We can call `trainer.vllm_generation.llm.chat(local_messages, sampling_params, chat_template=...)` directly per rank — **no LoRA adapter save/load needed for inference**. The training model itself is back to LoRA-on-base after `unmerge_adapter()`.

`vllm_enable_sleep_mode` defaults to `False` (`.venv/.../trl/trainer/grpo_config.py:491`); we keep it off, so no `wake_up()` is needed before generation.

## Architecture

Three layers change. Everything else stays put.

### Layer A — Launch and orchestration

**`tuning/training/unified_early_pipeline.py`**

1. New CLI flag: `--grpo-num-gpus N` (default `1`).
2. `_init_cuda_env()` becomes a no-op when `LOCAL_RANK in os.environ` (set by torchrun). SFT/DPO worker invocation does not set `LOCAL_RANK`, so their existing pin-to-GPU-0 + `CUDA_VISIBLE_DEVICES_ALL` behavior is preserved exactly.
3. When orchestrator dispatches a GRPO worker job:
   - Submits sbatch with `--gres=gpu:N` where `N = args.grpo_num_gpus`.
   - Sbatch script invokes `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE python -m tuning.training.unified_early_pipeline ...` for the GRPO subprocess.
4. SFT and DPO worker dispatch unchanged — they still go through bare `python tuning/training/unified_early_pipeline.py ...`.

**`tuning/slurm/unified_early_pipeline.sh`**

The script branches on whether the worker invocation is a GRPO step. Cleanest implementation: detect `--run-grpo` in `$@` and switch to `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE python -m tuning.training.unified_early_pipeline "$@"` only in that case. SFT and DPO continue to run via plain `python tuning/training/unified_early_pipeline.py "$@"`.

The orchestrator-mode invocation (no `--run-*` flag) continues to use plain `python` since it spawns subprocesses with the right launcher.

### Layer B — GRPO trainer wiring

**`tuning/training/grpo_training.py`**

Minimal change. `GRPOTrainer` reads `RANK`/`LOCAL_RANK`/`WORLD_SIZE` from environment automatically via accelerate's auto-detect, and TRL's colocate-vLLM path natively supports per-rank vLLM construction (see source references above).

Specific edits:
- The existing `set_trainer_vllm` hook at lines 88-89 stays as-is — `hasattr(trainer, 'vllm_generation')` is `True` on this TRL build, and the hook fires per-rank under DDP since `train_model_grpo` runs on every rank.
- Pass `trainer.accelerator` into the callback after construction so it can later call `accelerator.unwrap_model(...)` for adapter saving. Single hook line: `cb.set_accelerator(trainer.accelerator)` for any `PassAtKStoppingCallback` in `callbacks`.

Reward functions, model loading, and tokenizer setup do not change. The `train_model_grpo` function continues to run end-to-end on every rank; HF Trainer's accelerate integration handles DDP wrapping internally.

### Layer C — Eval callback (`PassAtKCallback`)

**`tuning/training/passk_callback.py`**

Two new methods plus rank gating in existing logic.

#### New: `_run_eval_with_results_ddp(model, eval_strategy)`

Activated when `dist.is_initialized() and dist.get_world_size() > 1`. All ranks gather the full response set and score it — `score_responses` is deterministic given the same inputs, so every rank arrives at the same `scores` dict. This avoids any post-eval broadcast: state mutations (threshold trimming, `prevResults` append, etc.) happen identically on every rank without explicit synchronization. Only the I/O operations (`wandb.log`, `model.save_pretrained`) are gated on rank 0.

```python
def _run_eval_with_results_ddp(self, model, eval_strategy):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    all_messages = eval_strategy.get_test_messages()
    local_indices = list(range(rank, len(all_messages), world_size))
    local_messages = [all_messages[i] for i in local_indices]

    sampling_params = SamplingParams(
        n=eval_strategy.n_samples,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
    )

    if local_messages:
        outputs = self._external_vllm.chat(
            local_messages,
            sampling_params,
            chat_template=self._chat_template,
        )
        local_pairs = [
            (idx, [r.text for r in out.outputs])
            for idx, out in zip(local_indices, outputs)
        ]
    else:
        local_pairs = []

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_pairs)

    # All ranks reorder, group, and score — deterministic given gathered responses.
    flat = sorted(
        ((idx, texts) for chunk in gathered for idx, texts in chunk),
        key=lambda t: t[0],
    )
    responses_per_index = [texts for _, texts in flat]
    test_prompts = eval_strategy.get_test_prompts()
    grouped = defaultdict(list)
    for prompt, resp in zip(test_prompts, responses_per_index):
        grouped[prompt].extend(resp)
    model_results = [{"prompt": p, "responses": r} for p, r in grouped.items()]
    scores = eval_strategy.score_responses(model_results, self.tokenizer)

    return scores, model_results
```

`self._external_vllm` is already plumbed via the existing `set_trainer_vllm(trainer.vllm_generation.llm)` hook. No new attributes.

#### Modified: `_run_eval_with_results(model, eval_strategy)`

Add a single new branch at the top:

```python
if dist.is_initialized() and dist.get_world_size() > 1:
    return self._run_eval_with_results_ddp(model, eval_strategy)
# else: existing single-GPU paths (external vLLM / persistent / data-parallel / ephemeral)
```

The four existing single-GPU paths are preserved unchanged for SFT, DPO, and `world_size=1` GRPO smoke runs (when `dist` is not initialized).

#### Modified: `on_evaluate`

Add a rank-0 helper and gate I/O operations on it. State mutations stay un-gated and run on every rank with identical inputs.

```python
def _is_rank_zero(self):
    return not dist.is_initialized() or dist.get_rank() == 0
```

Concretely, in `on_evaluate`:

1. Run primary eval via `_run_eval_with_results` (DDP-aware). All ranks receive the same `scores` and `raw_results`.
2. `self.prevResults.append(stopping_value)` — runs on every rank (deterministic).
3. `_log_raw_generation_table` and `wandb.log` — gated on `self._is_rank_zero()`.
4. Threshold detection (which threshold was reached, which `early_tuple` triggered) — runs on every rank deterministically.
5. `_save_sweetspot_checkpoint` — gated on `self._is_rank_zero()`. The threshold list trim (`self.target_pass_at_k_thresholds = ...[:reached_index]`) and `early_tuples.pop(idx)` run on every rank, in lockstep.
6. Gap-checkpoint logic — same pattern: file write gated on rank 0, the `self._last_checkpoint_data_points = data_points_seen` update runs on every rank.
7. `self._last_eval_step = state.global_step` — runs on every rank (`state.global_step` is already DDP-synced by HF Trainer).
8. Run monitor evals the same way (loop the DDP flow per monitor eval).

No `broadcast_object_list` is needed because every rank ends `on_evaluate` with the same internal state.

#### Sweetspot adapter save under DDP

`save_sweetspot_checkpoint` (in `tuning/training/callback_utils.py`) calls `model.save_pretrained(...)` directly. Under DDP the `model` passed in is wrapped by accelerate. Add an `accelerator: Accelerator | None = None` keyword argument; when provided, unwrap before save:

```python
def save_sweetspot_checkpoint(model, tokenizer, ..., accelerator=None):
    target = accelerator.unwrap_model(model) if accelerator is not None else model
    target.save_pretrained(...)
    tokenizer.save_pretrained(...)
```

Plumb the accelerator into `PassAtKCallback` via a new setter (`set_accelerator`) called from `train_model_grpo` in the same post-construction loop that calls `set_trainer_vllm` (`tuning/training/grpo_training.py:87-90`). The setter is a no-op for SFT/DPO since their training functions don't call it.

### Retired flags

- `--grpo-passk-num-inference-gpus` — under DDP, all GPUs are training+eval. Emit deprecation warning when this flag is passed with `--grpo-num-gpus > 1`. The flag's value is silently ignored in that case. SFT/DPO still honor the equivalent flags.

## Edge Cases

| Case | Handling |
|---|---|
| `world_size = 1` (torchrun `--nproc_per_node=1`) | DDP path degenerates: `prompts[0::1] == prompts`, `all_gather_object` is a single-element copy. Rank 0 gating becomes always-true. Identical results to non-DDP single-process runs. |
| `len(prompts) < world_size` | `prompts[rank::world_size]` returns empty list on the trailing ranks; the empty-list early-return in the DDP method skips the vLLM call. `dist.all_gather_object` works with empty lists. |
| Uneven splits (e.g. 1501 / 4 = 376/375/375/375) | Round-robin slicing handles automatically; reorder by `global_idx` after gather restores original order. |
| Multiple `monitor_evals` | Loop the DDP flow per eval. Each eval's gather is independent. |
| `n_samples > 1` per prompt | `SamplingParams(n=N)` handled inside vLLM per prompt; unaffected by DDP partitioning. |
| `vllm_enable_sleep_mode = True` (future) | Would require `trainer.vllm_generation.llm.wake_up([...])` before `.chat()`. Not enabled in this design. |
| Trainer ends mid-eval (interrupt, OOM) | Existing try/except in `train_model_grpo` (lines 99-109) catches; a `dist.destroy_process_group()` call in a finally block ensures NCCL cleanup. |
| W&B logging on non-zero ranks | `wandb.run` is None on non-zero ranks via HF Trainer's default rank-0-only init. We additionally gate `_log_raw_generation_table` on `_is_rank_zero()` to avoid wasted table-shaping work. |
| `args.world_size` in `data_points_seen` calc (`passk_callback.py:614`) | Already uses `getattr(args, "world_size", 1)`; HF Trainer sets `args.world_size` correctly under DDP. |
| LoRA adapter merge state during eval | `trainer.model` has unmerged LoRA after the last training step; `trainer.vllm_generation.llm` has merged trained weights. Two consistent copies for two distinct uses. |
| Persistent vLLM under DDP | The persistent-vLLM path is for SFT/DPO only and is incompatible with DDP. The DDP branch fires before any persistent-vLLM logic, so the persistent code is skipped under `world_size > 1`. |

## File-Level Change List

| File | Change | Approx size |
|---|---|---|
| `tuning/training/unified_early_pipeline.py` | Add `--grpo-num-gpus` flag; make `_init_cuda_env()` no-op when `LOCAL_RANK in os.environ`; orchestrator passes `--gres=gpu:N` for GRPO sbatch when `N > 1`. | ~30 lines |
| `tuning/slurm/unified_early_pipeline.sh` | Branch on `--run-grpo` in argv: invoke via `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE python -m tuning.training.unified_early_pipeline ...` only for GRPO worker mode. | ~15 lines |
| `tuning/training/grpo_training.py` | Add `cb.set_accelerator(trainer.accelerator)` plumbing for `PassAtKStoppingCallback`. Existing `set_trainer_vllm` hook unchanged. | ~5 lines |
| `tuning/training/passk_callback.py` | New `_run_eval_with_results_ddp`; new branch in `_run_eval_with_results`; `_is_rank_zero()` helper; rank-0 gating around `wandb.log`, `_log_raw_generation_table`, and `_save_sweetspot_checkpoint` in `on_evaluate`; new `set_accelerator` setter. | ~100 added, ~10 modified |
| `tuning/training/callback_utils.py` | `save_sweetspot_checkpoint`: optional `accelerator` kwarg; unwrap model before `save_pretrained`. | ~10 lines |
| `tuning/training/config_training.py` | No structural change. May tighten `vllm_gpu_memory_utilization` defaults during testing. | 0–5 lines |
| `tests/test_grpo_ddp_eval.py` (new) | Mock `dist.is_initialized` / `dist.get_rank` / `dist.get_world_size` / `dist.all_gather_object`; verify per-rank partitioning, gather merge, and rank-0 I/O gating. CPU-only. | ~150 lines |
| `tests/test_unified_pipeline_ddp.py` (new) | Argparse handling for `--grpo-num-gpus`; sbatch command construction; `_init_cuda_env` short-circuit when `LOCAL_RANK` is set. | ~80 lines |

**Files explicitly NOT touched:**

- `tuning/training/sft_training.py`, `tuning/training/dpo_training.py` (out of scope)
- All other `*_pipeline.py` files
- `tuning/inference/*`, `tuning/data/*`, `tuning/evaluation/*`, eval strategies
- `tuning/training/perplexity_callback.py` (perplexity is logits-based; falls back to single-rank path. If used with `--grpo-num-gpus > 1`, fails loud; will fix in a separate PR if needed.)

## Verification Plan

1. CPU-only unit tests: `pytest tests/test_grpo_config.py tests/test_grpo_ddp_eval.py tests/test_unified_pipeline_ddp.py`
2. Single-GPU GRPO smoke run end-to-end (verifies no regression at `world_size=1`).
3. 4-GPU GRPO smoke run end-to-end (verifies DDP path: training proceeds, eval fires, sweetspot save works, W&B run looks correct, no NCCL hangs).
4. `nvidia-smi` during 4-GPU run: all 4 GPUs at >50% util during training, all 4 active during `on_evaluate`.
5. Sweetspot artifact check: trained adapter loads correctly into vLLM in a downstream non-DDP eval — proves the unwrapped save path is correct.

## Out of Scope for This Spec

- DDP for SFT or DPO (separate work; different architectural fit).
- Multi-node DDP (would require a different launcher path and network fabric considerations).
- vLLM server mode (deferred until/unless colocate throughput plateaus on 8-GPU runs).
- vLLM tensor parallelism > 1 (defer; per-rank independent engines are sufficient at single-node scale).
- Sleep mode optimization (`vllm_enable_sleep_mode=True`) — possible future enhancement.
- Perplexity callback under DDP — fix in a separate PR if needed.

## Sources

- [TRL #2262: torchrun vs accelerate launch divergence](https://github.com/huggingface/trl/issues/2262)
- [HuggingFace blog: Co-located vLLM in TRL](https://huggingface.co/blog/vllm-colocate)
- [TRL GRPO Trainer source](https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py)
- [PEFT save_pretrained with DDP/Accelerator](https://huggingface.co/docs/peft/main/en/developer_guides/lora)
- Local: `.venv/lib/python3.11/site-packages/trl/trainer/grpo_trainer.py:67,713,1228`
- Local: `.venv/lib/python3.11/site-packages/trl/generation/vllm_generation.py:115,370,460`
