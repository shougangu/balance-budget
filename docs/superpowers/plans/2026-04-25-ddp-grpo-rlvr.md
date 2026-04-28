# DDP for GRPO RLVR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add single-node DDP for GRPO training and PassAtK eval. Each rank runs its colocated vLLM via `torchrun`; eval prompts are partitioned across ranks, gathered, and scored deterministically.

**Architecture:** GRPO worker mode launches via `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE`. Each rank runs `train_model_grpo` end-to-end. `PassAtKStoppingCallback` adds a DDP branch that partitions prompts, generates per rank via the colocated `ExternalVLLMRunner._llm.chat(...)`, gathers responses with `dist.all_gather_object`, and lets every rank score deterministically; only rank 0 calls `wandb.log` (via `passk/logging.py`) and `model.save_pretrained`. SFT and DPO are out of scope and unchanged.

**Tech Stack:** PyTorch DDP via `torchrun`, TRL 0.29.0+computecanada (with `VLLMGeneration` wrapper), HuggingFace Trainer/PEFT/Accelerate, vLLM colocate mode, pytest with mocked `vllm`/`unsloth`.

**Spec:** `docs/superpowers/specs/2026-04-25-ddp-grpo-rlvr-design.md`

---

## Refactor context (2026-04-28)

Since the original plan was written, the pipeline and passk callback were split into subpackages. References in this plan target the new layout:

| Old path | New path |
|---|---|
| `tuning/training/unified_early_pipeline.py` (CLI parsing, `_init_cuda_env`) | `tuning/training/pipeline/cli.py` |
| `tuning/training/unified_early_pipeline.py` (`main`, `_submit_sbatch_worker`, `_dispatch_parallel_workers`) | `tuning/training/pipeline/orchestrator.py` |
| `tuning/training/unified_early_pipeline.py` (stages: `run_sft` / `run_dpo` / `run_grpo`) | `tuning/training/pipeline/stages.py` |
| `tuning/training/passk_callback.py` (callback class) | `tuning/training/passk/callback.py` (the old path is a re-export shim) |

Symbol renames / removals to be aware of:

- `_init_cuda_env` → `init_cuda_env` (no leading underscore; lives in `pipeline.cli`).
- `self._external_vllm` no longer exists on the callback. Its replacement is `self._runner = ExternalVLLMRunner(self._runner_config, llm=...)`, set by `set_trainer_vllm`. The colocated vLLM is reachable as `self._runner._llm` (when the runner is `ExternalVLLMRunner`).
- `wandb.log` is no longer called inline in `on_evaluate`; it's encapsulated by `passk/logging.py:log_eval_metrics`, invoked through `self._eval_and_log(...)`.
- The threshold / early / gap branches collapsed into `CheckpointDecisionEngine.decide(...)`, which returns a list of `Decision(label, advances_state)` objects. The callback loops over decisions and calls `self._save_sweetspot_checkpoint(...)`.
- `save_sweetspot_checkpoint(...)` in `tuning/training/callback_utils.py` calls `model.save_pretrained_merged(checkpoint_path, tokenizer, save_method="merged_16bit")` — that's an unsloth API. GRPO uses `use_unsloth=False`, so the DDP path must save the unwrapped PEFT model via `target.save_pretrained(checkpoint_path)` instead. The patch in Task 5 handles this branching.
- `_dispatch_parallel_workers` already accepts a `sbatch_script` parameter (added by the parallel-post-training change). Task 11 layers `--gres=gpu:N` injection on top of it.

---

## File Structure

| File | Role |
|---|---|
| `tuning/training/pipeline/cli.py` | Add `--grpo-num-gpus` flag; make `init_cuda_env` no-op under torchrun |
| `tuning/training/pipeline/orchestrator.py` | `_dispatch_parallel_workers` injects `--gres=gpu:N` for GRPO worker submissions when `grpo_num_gpus>1` |
| `tuning/slurm/unified_early_pipeline.sh` | Branch on `--run-grpo` in argv; invoke `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE python -m tuning.training.unified_early_pipeline ...` for GRPO worker |
| `tuning/training/grpo_training.py` | Plumb `trainer.accelerator` into `PassAtKStoppingCallback` after trainer construction |
| `tuning/training/passk/callback.py` | New `_is_rank_zero` helper + `_accelerator` attribute; rank-0 gating in `_eval_and_log` and `on_evaluate`; new `_run_eval_with_results_ddp`; DDP branch in `_run_eval_with_results` |
| `tuning/training/callback_utils.py` | `save_sweetspot_checkpoint` accepts optional `accelerator` kwarg; when set, unwraps the model and calls `save_pretrained` (PEFT) instead of `save_pretrained_merged` (unsloth) |
| `tests/test_grpo_ddp_eval.py` (new) | Mock `torch.distributed`; verify partitioning, gather merge, rank-0 I/O gating |
| `tests/test_unified_pipeline_ddp.py` (new) | Argparse `--grpo-num-gpus`, `init_cuda_env` short-circuit when `LOCAL_RANK` set, sbatch dispatch with gres |

---

## Task 1: `init_cuda_env()` no-op when running under torchrun

`init_cuda_env` (formerly `_init_cuda_env`) pins training to GPU 0 and saves the rest as `CUDA_VISIBLE_DEVICES_ALL` for the spare-GPU eval workers. Under torchrun, every rank already has its `CUDA_VISIBLE_DEVICES` correctly pinned by `LOCAL_RANK`, so this function must short-circuit.

**Files:**
- Create: `tests/test_unified_pipeline_ddp.py`
- Modify: `tuning/training/pipeline/cli.py` (the `init_cuda_env` function around line 53)

- [ ] **Step 1: Write the failing test**

Create `tests/test_unified_pipeline_ddp.py`:

```python
# ABOUTME: Tests for DDP-related changes in pipeline.cli and pipeline.orchestrator.
# ABOUTME: CPU-only; mocks heavy imports.

import os

from tuning.training.pipeline.cli import init_cuda_env


def test_init_cuda_env_noop_when_local_rank_set(monkeypatch):
    """Under torchrun (LOCAL_RANK set), init_cuda_env must not mutate CUDA_VISIBLE_DEVICES."""
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES_ALL", raising=False)

    init_cuda_env()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "CUDA_VISIBLE_DEVICES_ALL" not in os.environ


def test_init_cuda_env_pins_gpu0_without_local_rank(monkeypatch):
    """Without torchrun (no LOCAL_RANK), legacy behavior: pin GPU 0, save the rest."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES_ALL", raising=False)

    init_cuda_env()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES_ALL"] == "0,1,2,3"
```

- [ ] **Step 2: Run test to verify the LOCAL_RANK case fails**

Run: `cd /project/6105902/shougan/balance-budget && source .venv/bin/activate && pytest tests/test_unified_pipeline_ddp.py::test_init_cuda_env_noop_when_local_rank_set -v`

Expected: FAIL — current code overwrites `CUDA_VISIBLE_DEVICES` to `"0"` regardless of LOCAL_RANK.

- [ ] **Step 3: Make `init_cuda_env` short-circuit under torchrun**

In `tuning/training/pipeline/cli.py`, replace the `init_cuda_env` function (around line 53):

```python
def init_cuda_env():
    """Restrict training to GPU 0 and save full GPU list for inference workers.

    No-op under torchrun (LOCAL_RANK is set) because each rank's CUDA_VISIBLE_DEVICES
    is already pinned per-rank by the launcher.
    """
    if "LOCAL_RANK" in os.environ:
        return
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if all_gpus:
        os.environ["CUDA_VISIBLE_DEVICES_ALL"] = all_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus.split(",")[0]
```

- [ ] **Step 4: Run both tests to verify they pass**

Run: `pytest tests/test_unified_pipeline_ddp.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_unified_pipeline_ddp.py tuning/training/pipeline/cli.py
git commit -m "feat: skip init_cuda_env when LOCAL_RANK is set (torchrun)"
```

---

## Task 2: `--grpo-num-gpus` CLI flag

Add the orchestrator-only flag that controls how many GPUs the GRPO sbatch worker gets. Default 1 keeps current single-GPU behavior.

**Files:**
- Modify: `tuning/training/pipeline/cli.py` (after `--grpo-num-epochs`, around line 156)
- Modify: `tests/test_unified_pipeline_ddp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_unified_pipeline_ddp.py`:

```python
from tuning.training.pipeline.cli import _parse_args


def test_grpo_num_gpus_default():
    args = _parse_args(["--model", "qwen2-2B", "--wandb-project", "test"])
    assert args.grpo_num_gpus == 1


def test_grpo_num_gpus_override():
    args = _parse_args([
        "--model", "qwen2-2B",
        "--wandb-project", "test",
        "--grpo-num-gpus", "4",
    ])
    assert args.grpo_num_gpus == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_unified_pipeline_ddp.py::test_grpo_num_gpus_default -v`

Expected: FAIL — `argparse.ArgumentError: unrecognized arguments` or `AttributeError: 'Namespace' object has no attribute 'grpo_num_gpus'`.

- [ ] **Step 3: Add the flag**

In `tuning/training/pipeline/cli.py`, add immediately after the `--grpo-num-epochs` argument (around line 156):

```python
    parser.add_argument("--grpo-num-gpus", type=int, default=1,
                        help="Number of GPUs for GRPO DDP training. >1 launches GRPO via torchrun.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_unified_pipeline_ddp.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/pipeline/cli.py tests/test_unified_pipeline_ddp.py
git commit -m "feat: add --grpo-num-gpus CLI flag (default 1)"
```

---

## Task 3: `_is_rank_zero()` helper on `PassAtKStoppingCallback`

Foundation for rank-aware logic. Returns `True` when not under DDP or when rank == 0.

**Files:**
- Create: `tests/test_grpo_ddp_eval.py`
- Modify: `tuning/training/passk/callback.py` (helper near top, after `__init__`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_grpo_ddp_eval.py`:

```python
# ABOUTME: Tests for DDP eval support in PassAtKStoppingCallback.
# ABOUTME: CPU-only; mocks vllm, unsloth, and torch.distributed.

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from tuning.training.config_training import PassAtKConfig
from tuning.training.passk.callback import PassAtKStoppingCallback


class _FakeEval:
    """Minimal EvalStrategy stand-in."""
    def __init__(self):
        self._n_samples = 1
        self.stopping_k = 1

    @property
    def id(self): return "test"
    @property
    def n_samples(self): return self._n_samples
    @property
    def label_prefix(self): return "p@1"
    def get_test_messages(self):
        return [[{"role": "user", "content": f"Prompt {i}"}] for i in range(8)]
    def get_test_prompts(self):
        return [f"Prompt {i}" for i in range(8)]
    def score_responses(self, results, tokenizer):
        return {"pass_at_1": 0.5}
    def stopping_metric(self):
        return "pass_at_1"
    def wandb_metrics(self, scores):
        return {"eval/pass_at_1": scores["pass_at_1"]}


def _make_callback(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    config = PassAtKConfig(
        target_pass_at_k=[0.5],
        temperature=0.5,
        max_tokens=128,
        enabled=True,
        use_persistent_vllm=False,
        vllm_gpu_memory_utilization=0.4,
        num_inference_gpus=1,
    )
    tokenizer = SimpleNamespace(chat_template="dummy",
                                apply_chat_template=lambda *a, **kw: "Prompt 0")
    return PassAtKStoppingCallback(
        config=config,
        tokenizer=tokenizer,
        model_name="qwen2-2B",
        base_model_hf="Qwen/Qwen2-2B",
        primary_eval=_FakeEval(),
        monitor_evals=[],
    )


def test_is_rank_zero_no_dist(monkeypatch):
    """Without torch.distributed initialized, every process is rank 0."""
    cb = _make_callback(monkeypatch)
    with patch("torch.distributed.is_initialized", return_value=False):
        assert cb._is_rank_zero() is True


def test_is_rank_zero_under_ddp(monkeypatch):
    """Under DDP, only rank 0 returns True."""
    cb = _make_callback(monkeypatch)
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=0):
        assert cb._is_rank_zero() is True
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1):
        assert cb._is_rank_zero() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grpo_ddp_eval.py::test_is_rank_zero_no_dist -v`

Expected: FAIL — `AttributeError: 'PassAtKStoppingCallback' object has no attribute '_is_rank_zero'`.

- [ ] **Step 3: Add the helper**

In `tuning/training/passk/callback.py`, add to the imports near the top:

```python
import torch.distributed as dist
```

Then add this method to `PassAtKStoppingCallback`, right after `__init__` and before `on_train_begin` (around line 132):

```python
    def _is_rank_zero(self) -> bool:
        """True when not under DDP, or when this is rank 0 under DDP."""
        return not dist.is_initialized() or dist.get_rank() == 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: add _is_rank_zero helper to PassAtKStoppingCallback"
```

---

## Task 4: `_accelerator` attribute on `PassAtKStoppingCallback`

Initialize the attribute to `None` in `__init__` so `train_model_grpo` can assign `trainer.accelerator` to it directly. No setter method — direct attribute access is enough since the callback is module-internal.

**Files:**
- Modify: `tuning/training/passk/callback.py` (inside `__init__`, near other state attrs)
- Modify: `tests/test_grpo_ddp_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grpo_ddp_eval.py`:

```python
def test_default_accelerator_is_none(monkeypatch):
    cb = _make_callback(monkeypatch)
    assert cb._accelerator is None


def test_accelerator_can_be_assigned_directly(monkeypatch):
    """train_model_grpo assigns trainer.accelerator to cb._accelerator directly."""
    cb = _make_callback(monkeypatch)
    fake_accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    cb._accelerator = fake_accelerator
    assert cb._accelerator is fake_accelerator
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grpo_ddp_eval.py::test_default_accelerator_is_none -v`

Expected: FAIL — `AttributeError: ... no attribute '_accelerator'`.

- [ ] **Step 3: Add the default attribute**

In `tuning/training/passk/callback.py` `__init__`, add next to `self._last_checkpoint_data_points = 0` (around line 63):

```python
        self._accelerator = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: add _accelerator attribute to PassAtKStoppingCallback"
```

---

## Task 5: `save_sweetspot_checkpoint` accepts optional `accelerator`

Current `save_sweetspot_checkpoint` calls `model.save_pretrained_merged(checkpoint_path, tokenizer, save_method="merged_16bit")` (an unsloth API). GRPO loads with `use_unsloth=False`, so the DDP path must use the standard PEFT `save_pretrained` on the unwrapped underlying model. SFT/DPO callers (no accelerator passed) keep the unsloth merged-save behavior unchanged.

**Files:**
- Modify: `tuning/training/callback_utils.py` (`save_sweetspot_checkpoint`, around line 53)
- Modify: `tests/test_grpo_ddp_eval.py`

- [ ] **Step 1: Confirm current `save_sweetspot_checkpoint` signature**

Run: `grep -n "def save_sweetspot_checkpoint\|save_pretrained_merged" /project/6105902/shougan/balance-budget/tuning/training/callback_utils.py`

Confirm signature ends after `extra_metadata: dict = None` and that line 84 is `model.save_pretrained_merged(checkpoint_path, tokenizer, save_method="merged_16bit")`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_grpo_ddp_eval.py`:

```python
def test_save_sweetspot_unwraps_when_accelerator_provided(tmp_path, monkeypatch):
    """With accelerator, save_pretrained is called on the unwrapped PEFT model (no unsloth merge)."""
    from tuning.training.callback_utils import save_sweetspot_checkpoint

    underlying = MagicMock(name="peft_model")
    wrapped = MagicMock(name="ddp_model")
    accelerator = SimpleNamespace(unwrap_model=MagicMock(return_value=underlying))
    tokenizer = MagicMock()

    state = SimpleNamespace(global_step=42, log_history=[])
    args = SimpleNamespace(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        world_size=4,
        output_dir=str(tmp_path),
        to_dict=lambda: {"per_device_train_batch_size": 8},
    )

    save_sweetspot_checkpoint(
        model=wrapped,
        tokenizer=tokenizer,
        model_name="qwen2-2B",
        threshold_label="p@1-0.5",
        state=state,
        args=args,
        metadata_path=str(tmp_path / "meta.jsonl"),
        extra_metadata={"threshold_type": "pass_at_1", "threshold_value": 0.5},
        accelerator=accelerator,
    )

    accelerator.unwrap_model.assert_called_once_with(wrapped)
    underlying.save_pretrained.assert_called_once()
    underlying.save_pretrained_merged.assert_not_called()
    wrapped.save_pretrained.assert_not_called()


def test_save_sweetspot_no_unwrap_when_accelerator_none(tmp_path):
    """Without accelerator (SFT/DPO callers), legacy merged save is used."""
    from tuning.training.callback_utils import save_sweetspot_checkpoint

    model = MagicMock(name="unsloth_model")
    tokenizer = MagicMock()

    state = SimpleNamespace(global_step=10, log_history=[])
    args = SimpleNamespace(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        world_size=1,
        output_dir=str(tmp_path),
        to_dict=lambda: {"per_device_train_batch_size": 8},
    )

    save_sweetspot_checkpoint(
        model=model,
        tokenizer=tokenizer,
        model_name="qwen2-2B",
        threshold_label="p@1-0.5",
        state=state,
        args=args,
        metadata_path=str(tmp_path / "meta.jsonl"),
        extra_metadata={"threshold_type": "pass_at_1", "threshold_value": 0.5},
    )

    model.save_pretrained_merged.assert_called_once()
    model.save_pretrained.assert_not_called()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_grpo_ddp_eval.py::test_save_sweetspot_unwraps_when_accelerator_provided -v`

Expected: FAIL — `TypeError: save_sweetspot_checkpoint() got an unexpected keyword argument 'accelerator'`.

- [ ] **Step 4: Add the kwarg and unwrap+PEFT-save branch**

Edit `tuning/training/callback_utils.py` `save_sweetspot_checkpoint`:

a. Add `accelerator=None` to the signature as the last keyword arg.

b. Replace the line `model.save_pretrained_merged(checkpoint_path, tokenizer, save_method="merged_16bit")` (around line 84) with:

```python
    if accelerator is not None:
        target = accelerator.unwrap_model(model)
        target.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
    else:
        model.save_pretrained_merged(checkpoint_path, tokenizer, save_method="merged_16bit")
```

(The DDP-PEFT branch saves the tokenizer separately because `save_pretrained` on a PEFT adapter doesn't include it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add tuning/training/callback_utils.py tests/test_grpo_ddp_eval.py
git commit -m "feat: save_sweetspot_checkpoint unwraps DDP model and uses PEFT save when accelerator passed"
```

---

## Task 6: Assign `trainer.accelerator` on the callback in `train_model_grpo`

Direct attribute assignment in the same loop as the existing `set_trainer_vllm` hook. No setter ceremony — the callback is module-internal and `_accelerator` was initialized in Task 4.

**Files:**
- Modify: `tuning/training/grpo_training.py:87-90`

- [ ] **Step 1: Read the existing hook**

Run: `sed -n '85,95p' /project/6105902/shougan/balance-budget/tuning/training/grpo_training.py`

Confirm the loop that calls `cb.set_trainer_vllm(trainer.vllm_generation.llm)`.

- [ ] **Step 2: Add the accelerator assignment**

Edit `tuning/training/grpo_training.py:87-90`. Replace the existing loop:

```python
    for cb in callbacks or []:
        if isinstance(cb, PassAtKStoppingCallback) and hasattr(trainer, 'vllm_generation'):
            cb.set_trainer_vllm(trainer.vllm_generation.llm)
            print(f"[GRPO] PassAtK callback will reuse GRPOTrainer's vLLM engine")
```

with:

```python
    for cb in callbacks or []:
        if isinstance(cb, PassAtKStoppingCallback):
            if hasattr(trainer, 'vllm_generation'):
                cb.set_trainer_vllm(trainer.vllm_generation.llm)
                print(f"[GRPO] PassAtK callback will reuse GRPOTrainer's vLLM engine")
            cb._accelerator = trainer.accelerator
```

- [ ] **Step 3: Run existing tests to make sure nothing regressed**

Run: `pytest tests/test_grpo_config.py tests/test_grpo_ddp_eval.py tests/test_unified_pipeline_ddp.py -v`

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add tuning/training/grpo_training.py
git commit -m "feat: assign trainer.accelerator on PassAtKStoppingCallback"
```

---

## Task 7: `_run_eval_with_results_ddp` method

Main DDP eval logic. Each rank generates its slice via the colocated vLLM (reachable as `self._runner._llm` after `set_trainer_vllm` swaps the runner to `ExternalVLLMRunner`), all ranks gather, all ranks score deterministically.

**Files:**
- Modify: `tuning/training/passk/callback.py` (add new method, near `_run_eval_with_results` around line 204)
- Modify: `tests/test_grpo_ddp_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grpo_ddp_eval.py`:

```python
def test_run_eval_ddp_partitions_and_merges(monkeypatch):
    """Rank 1 of 2 generates indices 1,3,5,7; rank 0 generates 0,2,4,6.
    After all_gather, every rank reconstructs the full ordered response set and scores."""
    cb = _make_callback(monkeypatch)

    fake_llm = MagicMock()
    def fake_chat(messages, sampling_params, chat_template):
        return [SimpleNamespace(outputs=[SimpleNamespace(text=f"resp_for_{m[0]['content']}")])
                for m in messages]
    fake_llm.chat.side_effect = fake_chat
    cb.set_trainer_vllm(fake_llm)

    eval_strategy = _FakeEval()

    # Simulate rank 0 of 2
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.all_gather_object") as mock_gather:
        rank1_pairs = [(i, [f"resp_for_Prompt {i}"]) for i in [1, 3, 5, 7]]
        def fake_gather(out_list, local):
            out_list[0] = local
            out_list[1] = rank1_pairs
        mock_gather.side_effect = fake_gather

        scores, model_results = cb._run_eval_with_results_ddp(model=MagicMock(), eval_strategy=eval_strategy)

    assert len(model_results) == 8
    assert scores == {"pass_at_1": 0.5}


def test_run_eval_ddp_handles_empty_local_slice(monkeypatch):
    """Rank with empty slice (more ranks than prompts) skips chat and still gathers."""
    cb = _make_callback(monkeypatch)

    fake_llm = MagicMock()
    fake_llm.chat.side_effect = AssertionError("chat should not be called on empty slice")
    cb.set_trainer_vllm(fake_llm)

    class _FewPromptsEval(_FakeEval):
        def get_test_messages(self):
            return [[{"role": "user", "content": "P0"}]]
        def get_test_prompts(self):
            return ["P0"]

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.all_gather_object") as mock_gather:
        def fake_gather(out_list, local):
            out_list[0] = [(0, ["resp_for_P0"])]
            out_list[1] = local
        mock_gather.side_effect = fake_gather

        scores, model_results = cb._run_eval_with_results_ddp(
            model=MagicMock(), eval_strategy=_FewPromptsEval()
        )

    assert len(model_results) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grpo_ddp_eval.py::test_run_eval_ddp_partitions_and_merges -v`

Expected: FAIL — `AttributeError: ... has no attribute '_run_eval_with_results_ddp'`.

- [ ] **Step 3: Implement the method**

In `tuning/training/passk/callback.py`, near `_run_eval_with_results` (around line 204), add:

```python
    def _run_eval_with_results_ddp(self, model, eval_strategy: EvalStrategy):
        """DDP eval path: each rank generates its slice via the colocated vLLM,
        all ranks gather responses, all ranks score deterministically.

        Returns (scores, model_results) on every rank. Requires the runner to be
        ExternalVLLMRunner (set by set_trainer_vllm in train_model_grpo).
        """
        from collections import defaultdict
        from vllm import SamplingParams
        from tuning.inference.config_inference import VLLMSamplingParamsConfig
        from tuning.training.passk.runners import ExternalVLLMRunner

        if not isinstance(self._runner, ExternalVLLMRunner):
            raise RuntimeError(
                "DDP eval path requires the colocated trainer vLLM "
                "(set via set_trainer_vllm)."
            )
        llm = self._runner._llm

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        all_messages = eval_strategy.get_test_messages()
        local_indices = list(range(rank, len(all_messages), world_size))
        local_messages = [all_messages[i] for i in local_indices]

        sampling_params = SamplingParams(**VLLMSamplingParamsConfig(
            n=eval_strategy.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).model_dump())

        if local_messages:
            outputs = llm.chat(
                local_messages,
                sampling_params,
                chat_template=self._runner.config.chat_template,
            )
            local_pairs = [
                (idx, [r.text for r in out.outputs])
                for idx, out in zip(local_indices, outputs)
            ]
        else:
            local_pairs = []

        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_pairs)

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

        print(f"[PassAtKCallback] DDP eval rank={rank}/{world_size}: "
              f"local={len(local_messages)} prompts, gathered={len(model_results)} unique")
        scores = eval_strategy.score_responses(model_results, self.tokenizer)
        return scores, model_results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: add _run_eval_with_results_ddp for cooperative DDP eval"
```

---

## Task 8: DDP branch in `_run_eval_with_results`

Dispatcher: when DDP is active and `world_size > 1`, route to the new method. Otherwise fall through to the existing `self._runner.run(...)` path.

**Files:**
- Modify: `tuning/training/passk/callback.py` (top of `_run_eval_with_results`, around line 204)
- Modify: `tests/test_grpo_ddp_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grpo_ddp_eval.py`:

```python
def test_dispatch_ddp_when_world_size_gt_1(monkeypatch):
    """When dist is initialized with world_size > 1, _run_eval_with_results dispatches to DDP path."""
    cb = _make_callback(monkeypatch)
    cb.set_trainer_vllm(MagicMock())

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_world_size", return_value=4), \
         patch.object(cb, "_run_eval_with_results_ddp",
                      return_value=({"pass_at_1": 0.42}, [{"prompt": "p", "responses": ["r"]}])) as ddp_mock:
        scores, results = cb._run_eval_with_results(model=MagicMock(), eval_strategy=_FakeEval())

    ddp_mock.assert_called_once()
    assert scores == {"pass_at_1": 0.42}


def test_dispatch_single_gpu_when_world_size_1(monkeypatch):
    """When world_size == 1, falls through to runner.run path (DDP method NOT called)."""
    cb = _make_callback(monkeypatch)
    cb.set_trainer_vllm(MagicMock())
    # Stub the runner so we don't hit real vLLM.
    cb._runner.run = MagicMock(return_value=[{"prompt": "p", "responses": ["r"]}])

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_world_size", return_value=1), \
         patch.object(cb, "_run_eval_with_results_ddp") as ddp_mock:
        cb._run_eval_with_results(model=MagicMock(), eval_strategy=_FakeEval())

    ddp_mock.assert_not_called()
    cb._runner.run.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grpo_ddp_eval.py::test_dispatch_ddp_when_world_size_gt_1 -v`

Expected: FAIL — DDP method is never called by current dispatcher.

- [ ] **Step 3: Add the branch**

In `tuning/training/passk/callback.py`, find `def _run_eval_with_results(self, model, eval_strategy)` (around line 204). Add at the very top of the method body, before the `with tempfile.TemporaryDirectory() as adapter_dir:` block:

```python
        if dist.is_initialized() and dist.get_world_size() > 1:
            return self._run_eval_with_results_ddp(model, eval_strategy)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: dispatch DDP eval path when world_size > 1"
```

---

## Task 9: Rank-0 gating for I/O in `_eval_and_log` and `on_evaluate`

Gate the I/O operations (W&B logging, raw-generation table, sweetspot save) on `_is_rank_zero()`. State mutations stay un-gated and run identically on every rank — every rank scores, every rank advances `_primary_metric_history`, every rank advances the `CheckpointDecisionEngine`, every rank advances `_last_checkpoint_data_points`. Only rank 0 actually calls `wandb.log` (via `log_eval_metrics`) and `_save_sweetspot_checkpoint`.

**Files:**
- Modify: `tuning/training/passk/callback.py` (`_eval_and_log` around line 236, `on_evaluate` around line 249, and `_save_sweetspot_checkpoint` around line 182 to thread `accelerator`)
- Modify: `tests/test_grpo_ddp_eval.py`

- [ ] **Step 1: Read the current implementations**

Run:
```
sed -n '180,200p' /project/6105902/shougan/balance-budget/tuning/training/passk/callback.py
sed -n '236,282p' /project/6105902/shougan/balance-budget/tuning/training/passk/callback.py
```

Confirm:
- `_save_sweetspot_checkpoint` (line ~182) calls `save_sweetspot_checkpoint(...)` — needs to pass `accelerator=self._accelerator`.
- `_eval_and_log` (line ~236) calls `log_eval_metrics(...)` — wrap in rank-0 check.
- `on_evaluate` decision loop (line ~274) calls `self._save_sweetspot_checkpoint(...)` — wrap call site in rank-0 check while leaving `_last_checkpoint_data_points` advancement un-gated.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_grpo_ddp_eval.py`:

```python
def test_on_evaluate_gates_wandb_on_rank_nonzero(monkeypatch):
    """Under DDP rank 1, log_eval_metrics and _save_sweetspot_checkpoint must NOT fire."""
    cb = _make_callback(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.chat.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="r")])
                                  for _ in range(4)]
    cb.set_trainer_vllm(fake_llm)
    cb.metadata_path = "/tmp/dummy_metadata.json"

    state = SimpleNamespace(global_step=10, log_history=[])
    args = SimpleNamespace(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        world_size=2,
        output_dir="/tmp",
    )
    control = SimpleNamespace(should_training_stop=False)

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.all_gather_object") as mock_gather, \
         patch("tuning.training.passk.callback.log_eval_metrics") as mock_log, \
         patch.object(cb, "_save_sweetspot_checkpoint") as mock_save:
        def fake_gather(out_list, local):
            for i in range(len(out_list)):
                out_list[i] = local if i == 1 else [(j, ["r"]) for j in range(0, 8, 2)]
        mock_gather.side_effect = fake_gather

        cb.on_evaluate(args, state, control, model=MagicMock())

    mock_log.assert_not_called()
    mock_save.assert_not_called()


def test_on_evaluate_runs_wandb_on_rank0(monkeypatch):
    """Under DDP rank 0, log_eval_metrics fires (existing behavior)."""
    cb = _make_callback(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.chat.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="r")])
                                  for _ in range(4)]
    cb.set_trainer_vllm(fake_llm)
    cb.metadata_path = "/tmp/dummy_metadata.json"

    state = SimpleNamespace(global_step=10, log_history=[])
    args = SimpleNamespace(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        world_size=2,
        output_dir="/tmp",
    )
    control = SimpleNamespace(should_training_stop=False)

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=0), \
         patch("torch.distributed.get_world_size", return_value=2), \
         patch("torch.distributed.all_gather_object") as mock_gather, \
         patch("tuning.training.passk.callback.log_eval_metrics") as mock_log:
        def fake_gather(out_list, local):
            for i in range(len(out_list)):
                out_list[i] = local if i == 0 else [(j, ["r"]) for j in range(1, 8, 2)]
        mock_gather.side_effect = fake_gather

        cb.on_evaluate(args, state, control, model=MagicMock())

    assert mock_log.called
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_grpo_ddp_eval.py::test_on_evaluate_gates_wandb_on_rank_nonzero -v`

Expected: FAIL — `log_eval_metrics` is called on every rank in current code.

- [ ] **Step 4: Add rank-0 gating and thread `accelerator`**

In `tuning/training/passk/callback.py`:

a. Update `_save_sweetspot_checkpoint` (around line 182) to forward `accelerator`:

```python
    def _save_sweetspot_checkpoint(self, model, threshold, state: TrainerState, args: TrainingArguments):
        """Save a checkpoint when a sweetspot threshold is reached."""
        return save_sweetspot_checkpoint(
            model=model,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            threshold_label=f"{self.primary_eval.label_prefix}-{threshold}",
            state=state,
            args=args,
            metadata_path=self.metadata_path,
            extra_metadata={
                "threshold_type": self.primary_eval.stopping_metric(),
                "threshold_value": threshold,
            },
            accelerator=self._accelerator,
        )
```

b. Update `_eval_and_log` (around line 236) to gate `log_eval_metrics`:

```python
    def _eval_and_log(self, model, eval_strategy, state, *, is_primary: bool):
        scores, raw_results = self._run_eval_with_results(model, eval_strategy)
        if self._is_rank_zero():
            log_eval_metrics(
                eval_strategy=eval_strategy,
                scores=scores,
                raw_results=raw_results,
                global_step=state.global_step,
                step_offset=self._step_offset,
                thresholds_remaining=self._decision_engine.target_thresholds,
                is_primary=is_primary,
            )
        return scores
```

c. Update the decision loop in `on_evaluate` (around line 274) to gate the save while leaving state advancement on every rank:

```python
        for decision in decisions:
            if self._is_rank_zero():
                self._save_sweetspot_checkpoint(model, decision.label, state, args)
            if decision.advances_state:
                self._last_checkpoint_data_points = data_points_seen
            if self._is_rank_zero():
                print(f"[PassAtKCallback] Saved checkpoint: {decision.label}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 12 passed.

- [ ] **Step 6: Run the full callback test suite to make sure nothing regressed**

Run: `pytest tests/test_external_vllm_reuse.py tests/test_callback_step_bridging.py tests/test_passk_callback_wandb_tables.py -v`

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add tuning/training/passk/callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: rank-0 gate I/O in PassAtKCallback eval and decision loops"
```

---

## Task 10: sbatch script branches on `--run-grpo` to use torchrun

The sbatch script invokes the unified pipeline. For GRPO worker mode, switch to `torchrun`. Other invocations (orchestrator, SFT, DPO) keep the bare `python` call.

**Files:**
- Modify: `tuning/slurm/unified_early_pipeline.sh` (the `python tuning/training/unified_early_pipeline.py "$@"` line, around line 35)

- [ ] **Step 1: Read the current dispatch line**

Run: `sed -n '30,45p' /project/6105902/shougan/balance-budget/tuning/slurm/unified_early_pipeline.sh`

- [ ] **Step 2: Add the GRPO branch**

Replace the line `python tuning/training/unified_early_pipeline.py "$@"` with:

```bash
# GRPO worker mode launches under torchrun for DDP across all node GPUs.
# Other modes (orchestrator, --run-sft, --run-dpo) use plain python.
IS_GRPO_WORKER=0
for _arg in "$@"; do
    if [[ "$_arg" == "--run-grpo" ]]; then
        IS_GRPO_WORKER=1
        break
    fi
done

NPROC="${SLURM_GPUS_ON_NODE:-1}"

if [[ "$IS_GRPO_WORKER" == "1" ]]; then
    echo "[unified_early_pipeline.sh] GRPO worker mode: torchrun --nproc_per_node=${NPROC}"
    torchrun --nproc_per_node="${NPROC}" -m tuning.training.unified_early_pipeline "$@"
else
    python tuning/training/unified_early_pipeline.py "$@"
fi
```

- [ ] **Step 3: Lint the script**

Run: `bash -n tuning/slurm/unified_early_pipeline.sh`

Expected: no syntax errors (no output).

- [ ] **Step 4: Verify `python -m tuning.training.unified_early_pipeline` works**

Run: `cd /project/6105902/shougan/balance-budget && source .venv/bin/activate && python -m tuning.training.unified_early_pipeline --help 2>&1 | head -3`

Expected: argparse usage line printed (not a `ModuleNotFoundError`).

- [ ] **Step 5: Commit**

```bash
git add tuning/slurm/unified_early_pipeline.sh
git commit -m "feat: dispatch GRPO worker via torchrun for DDP"
```

---

## Task 11: Orchestrator submits GRPO sbatch with `--gres=gpu:N`

When `pt_method == "grpo"` and `args.grpo_num_gpus > 1`, inject `--gres=gpu:{args.grpo_num_gpus}` at the sbatch dispatch site. `_dispatch_parallel_workers` already takes `sbatch_script`; we extend it with `args` so it can compute the gres flag inline.

**Files:**
- Modify: `tuning/training/pipeline/orchestrator.py` (`_submit_sbatch_worker` around line 17, `_dispatch_parallel_workers` around line 33, and its caller in `main` around line 102)
- Modify: `tests/test_unified_pipeline_ddp.py`

- [ ] **Step 1: Read current sbatch submission code**

Run: `grep -n "_submit_sbatch_worker\|_dispatch_parallel_workers" /project/6105902/shougan/balance-budget/tuning/training/pipeline/orchestrator.py`

Confirm the `cmd = ["sbatch", sbatch_script, *worker_args]` line in `_submit_sbatch_worker` and the call site in `main`.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_unified_pipeline_ddp.py`:

```python
from unittest.mock import patch, MagicMock
from types import SimpleNamespace


def test_dispatch_grpo_passes_gres_to_sbatch(tmp_path):
    """When pt_method='grpo' and grpo_num_gpus=4, dispatch injects --gres=gpu:4."""
    from tuning.training.pipeline.orchestrator import _dispatch_parallel_workers

    metadata_file = tmp_path / "meta.jsonl"
    metadata_file.write_text("")

    args = SimpleNamespace(
        post_training_method="grpo",
        grpo_num_gpus=4,
        parallel=2,
    )

    fake_result = MagicMock(returncode=0,
                            stdout="Submitted batch job 12345\n",
                            stderr="")
    with patch("tuning.training.pipeline.orchestrator.subprocess.run",
               return_value=fake_result) as mock_run:
        _dispatch_parallel_workers(
            parallel=args.parallel,
            base_cmd=["python", "tuning/training/unified_early_pipeline.py", "--model", "qwen2-2B"],
            pt_flag="--run-grpo",
            metadata_files=[str(metadata_file)],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )

    cmd = mock_run.call_args[0][0]
    assert "--gres=gpu:4" in cmd
    assert cmd[0] == "sbatch"


def test_dispatch_dpo_does_not_pass_gres(tmp_path):
    """For DPO (or grpo_num_gpus=1), no --gres flag is injected (sbatch script default applies)."""
    from tuning.training.pipeline.orchestrator import _dispatch_parallel_workers

    metadata_file = tmp_path / "meta.jsonl"
    metadata_file.write_text("")

    args = SimpleNamespace(
        post_training_method="dpo",
        grpo_num_gpus=1,
        parallel=2,
    )

    fake_result = MagicMock(returncode=0,
                            stdout="Submitted batch job 12345\n",
                            stderr="")
    with patch("tuning.training.pipeline.orchestrator.subprocess.run",
               return_value=fake_result) as mock_run:
        _dispatch_parallel_workers(
            parallel=args.parallel,
            base_cmd=["python", "tuning/training/unified_early_pipeline.py", "--model", "qwen2-2B"],
            pt_flag="--run-dpo",
            metadata_files=[str(metadata_file)],
            sbatch_script="tuning/slurm/unified_early_pipeline.sh",
            args=args,
        )

    cmd = mock_run.call_args[0][0]
    assert not any(a.startswith("--gres") for a in cmd)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_unified_pipeline_ddp.py::test_dispatch_grpo_passes_gres_to_sbatch -v`

Expected: FAIL — `_dispatch_parallel_workers` doesn't accept `args` kwarg.

- [ ] **Step 4: Plumb `args` and inject `--gres` inline**

In `tuning/training/pipeline/orchestrator.py`:

a. Modify `_submit_sbatch_worker` (around line 17) to accept extra pre-script flags:

```python
def _submit_sbatch_worker(sbatch_script, worker_args, sbatch_flags=()):
    """Submit an sbatch worker job, return the Slurm job ID as a string.

    sbatch_flags go between 'sbatch' and the script path. Exits the
    orchestrator on sbatch error or unparseable output.
    """
    cmd = ["sbatch", *sbatch_flags, sbatch_script, *worker_args]
    print(f"[orchestrator] Submitting sbatch worker: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"sbatch failed (code {result.returncode}): {result.stderr.strip()}")
    tokens = result.stdout.strip().split()
    if len(tokens) < 4 or tokens[0] != "Submitted":
        sys.exit(f"Unexpected sbatch stdout: {result.stdout!r}")
    return tokens[-1]
```

b. Modify `_dispatch_parallel_workers` (around line 33) to accept `args` and compute the gres flag inline:

```python
def _dispatch_parallel_workers(parallel, base_cmd, pt_flag, metadata_files,
                                sbatch_script, args):
    """Submit parallel-1 sbatch workers for post-training.

    Injects --gres=gpu:N when pt_method=='grpo' and grpo_num_gpus>1. No-op when
    parallel <= 1. Strips --parallel from worker args so workers don't recursively
    dispatch.
    """
    if parallel <= 1:
        return

    sbatch_flags = []
    if args.post_training_method == "grpo" and args.grpo_num_gpus > 1:
        sbatch_flags.append(f"--gres=gpu:{args.grpo_num_gpus}")

    worker_argv = []
    skip_next = False
    for tok in base_cmd[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok == "--parallel":
            skip_next = True
            continue
        worker_argv.append(tok)
    worker_argv += [pt_flag, "--run-all"]
    for mf in metadata_files:
        if Path(mf).is_file():
            worker_argv += ["--metadata-file", mf]

    for i in range(parallel - 1):
        job_id = _submit_sbatch_worker(sbatch_script, worker_argv,
                                        sbatch_flags=sbatch_flags)
        print(f"[orchestrator] Submitted worker {i+1}/{parallel-1}: job {job_id}")
```

c. Update the single call site in `main()` (around line 102) to pass `args`:

```python
    _dispatch_parallel_workers(
        parallel=args.parallel,
        base_cmd=base_cmd,
        pt_flag=pt_flag,
        metadata_files=all_files,
        sbatch_script=args.sbatch_script,
        args=args,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_unified_pipeline_ddp.py -v`

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add tuning/training/pipeline/orchestrator.py tests/test_unified_pipeline_ddp.py
git commit -m "feat: inject --gres=gpu:N for GRPO sbatch when grpo_num_gpus>1"
```

---

## Task 12: Manual smoke verification

CPU-only unit tests are not enough. Verify the DDP path actually trains and evaluates correctly on real GPUs. Run as fire-and-forget sbatch (matches Shougan's preference).

**Files:** none (one-off verification)

- [ ] **Step 1: Run the full CPU test suite**

Run: `pytest tests/test_grpo_config.py tests/test_grpo_ddp_eval.py tests/test_unified_pipeline_ddp.py tests/test_external_vllm_reuse.py tests/test_callback_step_bridging.py tests/test_passk_callback_wandb_tables.py -v`

Expected: all green.

- [ ] **Step 2: Single-GPU GRPO smoke run (regression check)**

Submit a 1-GPU GRPO run with the existing CLI (no `--grpo-num-gpus`). Verify training proceeds, eval fires, sweetspot save works, W&B log looks correct. Pick the smallest model (qwen2-2B), small `--train-size`, e.g.:

```bash
sbatch tuning/slurm/unified_early_pipeline.sh \
  --model qwen2-2B --wandb-project ddp_smoke \
  --task-name gsm8k --dataset openmath \
  --post-training-method grpo --run-grpo --run-all \
  --train-size 64 --grpo-num-epochs 1 --grpo-eval-steps 8
```

Watch `<JOBID>_ddp_smoke.out` until it finishes. Confirm:
- `[unified_early_pipeline.sh] GRPO worker mode` line **is** present (the `--run-grpo` argv triggers torchrun even with NPROC=1).
- Training step logs appear.
- `[PassAtKCallback]` log shows eval running once.
- W&B run created with `grpo` tag.

- [ ] **Step 3: 4-GPU GRPO smoke run (DDP path)**

```bash
sbatch tuning/slurm/unified_early_pipeline.sh \
  --model qwen2-2B --wandb-project ddp_smoke \
  --task-name gsm8k --dataset openmath \
  --post-training-method grpo --grpo-num-gpus 4 \
  --run-grpo --run-all \
  --train-size 64 --grpo-num-epochs 1 --grpo-eval-steps 8
```

Watch the output file. Confirm:
- `GRPO worker mode: torchrun --nproc_per_node=4` printed by sbatch.
- Four ranks start (look for matching `[PassAtKCallback]` lines from each rank).
- Training proceeds without NCCL hangs.
- Exactly one W&B run is created (rank 0 only).
- `[PassAtKCallback] DDP eval rank=0/4: ...` log line appears at eval time.
- One sweetspot checkpoint saved if threshold reached, in expected `tuning/models/...` location.

- [ ] **Step 4: Verify sweetspot artifact loads correctly downstream**

Pick the saved checkpoint path from Step 3, run a non-DDP eval against it (any existing single-GPU eval entry point). The adapter must load into vLLM without error — proves the unwrapped `save_pretrained` path in Task 5 produces a valid PEFT adapter.

- [ ] **Step 5: Update `MEMORY.md` with run identifiers**

If the smoke runs revealed any unexpected `vllm_gpu_memory_utilization` adjustments (because DDP shares vLLM mem with grad mem), record the working values in `MEMORY.md` for future runs.

---

## Self-Review

**Spec coverage:**

- Decisions table: each row maps to a task — Layer A → Tasks 1, 2, 10, 11; Layer B → Task 6; Layer C → Tasks 3, 4, 5, 7, 8, 9; Verification → Task 12.
- TRL source-of-truth references: cited in spec; tasks reference `trainer.vllm_generation.llm` (Task 6 + 7) and `trainer.accelerator` (Task 6) consistently with the spec.
- Edge cases (`world_size=1` degenerate, empty local slice, monitor evals): covered by tests in Tasks 7, 8, 9 plus Task 12 smoke runs.
- Retired flag (`--grpo-passk-num-inference-gpus`): not actively retired in this plan — the spec calls it "deprecation warning" but YAGNI applies; the existing flag continues to work for SFT/DPO and is silently bypassed for the DDP eval path because the DDP branch in `_run_eval_with_results` runs first. No code changes needed; revisit if confusion arises in practice.

**Refactor alignment:**

- All paths target the new `tuning/training/pipeline/` and `tuning/training/passk/` subpackages.
- Tests import from new locations (`tuning.training.passk.callback`, `tuning.training.pipeline.cli`, `tuning.training.pipeline.orchestrator`). The `passk_callback.py` shim still works, so production imports there are untouched.
- `init_cuda_env` (was `_init_cuda_env`) referenced consistently.
- `set_trainer_vllm` swaps the runner to `ExternalVLLMRunner`; DDP path reaches `self._runner._llm` rather than the removed `self._external_vllm`.
- `wandb.log` gating happens in `_eval_and_log`, since the actual `wandb.log` call lives inside `passk/logging.py:log_eval_metrics` rather than the callback.
- `save_sweetspot_checkpoint` accepts `accelerator`; DDP path uses PEFT `save_pretrained` because GRPO is `use_unsloth=False`, while SFT/DPO callers keep the unsloth `save_pretrained_merged` path.

**Placeholder scan:** no "TBD" / "TODO" / vague handlers. Each step has exact code or exact commands.

**Type/name consistency:**
- `_is_rank_zero()` defined in Task 3, used in Task 9. ✓
- `_accelerator` defined in Task 4, set in Task 6, consumed in Task 9 via `_save_sweetspot_checkpoint` → `save_sweetspot_checkpoint(... accelerator=...)`. ✓
- `_run_eval_with_results_ddp` defined in Task 7, dispatched in Task 8. ✓
- `--grpo-num-gpus` defined in Task 2 (in `pipeline.cli`), consumed in Task 11 (`args.grpo_num_gpus` in `pipeline.orchestrator`). ✓
- sbatch script reads `$SLURM_GPUS_ON_NODE` (Task 10); orchestrator passes `--gres=gpu:N` (Task 11) which sets that var. ✓
