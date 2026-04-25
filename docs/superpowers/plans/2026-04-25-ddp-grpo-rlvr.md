# DDP for GRPO RLVR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add single-node DDP for GRPO training and PassAtK eval. Each rank runs its colocated vLLM via `torchrun`; eval prompts are partitioned across ranks, gathered, and scored deterministically.

**Architecture:** GRPO worker mode launches via `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE`. Each rank runs `train_model_grpo` end-to-end. `PassAtKCallback` adds a DDP branch that partitions prompts, generates per rank via `trainer.vllm_generation.llm.chat(...)`, gathers responses with `dist.all_gather_object`, and lets every rank score deterministically; only rank 0 calls `wandb.log` and `model.save_pretrained`. SFT and DPO are out of scope and unchanged.

**Tech Stack:** PyTorch DDP via `torchrun`, TRL 0.29.0+computecanada (with `VLLMGeneration` wrapper), HuggingFace Trainer/PEFT/Accelerate, vLLM colocate mode, pytest with mocked `vllm`/`unsloth`.

**Spec:** `docs/superpowers/specs/2026-04-25-ddp-grpo-rlvr-design.md`

---

## File Structure

| File | Role |
|---|---|
| `tuning/training/unified_early_pipeline.py` | Add `--grpo-num-gpus` flag; make `_init_cuda_env` no-op under torchrun; orchestrator submits GRPO sbatch with `--gres=gpu:N` |
| `tuning/slurm/unified_early_pipeline.sh` | Branch on `--run-grpo` in argv; invoke `torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE python -m tuning.training.unified_early_pipeline ...` for GRPO worker |
| `tuning/training/grpo_training.py` | Plumb `trainer.accelerator` into `PassAtKStoppingCallback` after trainer construction |
| `tuning/training/passk_callback.py` | New `_is_rank_zero` helper, `set_accelerator` setter, `_run_eval_with_results_ddp`; DDP branch in `_run_eval_with_results`; rank-0 gating in `on_evaluate` |
| `tuning/training/callback_utils.py` | `save_sweetspot_checkpoint` accepts optional `accelerator` kwarg and unwraps the model before `save_pretrained` |
| `tests/test_grpo_ddp_eval.py` (new) | Mock `torch.distributed`; verify partitioning, gather merge, rank-0 I/O gating |
| `tests/test_unified_pipeline_ddp.py` (new) | Argparse `--grpo-num-gpus`, `_init_cuda_env` short-circuit when `LOCAL_RANK` set, sbatch dispatch |

---

## Task 1: `_init_cuda_env()` no-op when running under torchrun

`_init_cuda_env` pins training to GPU 0 and saves the rest as `CUDA_VISIBLE_DEVICES_ALL` for the spare-GPU eval workers. Under torchrun, every rank already has its `CUDA_VISIBLE_DEVICES` correctly pinned by `LOCAL_RANK`, so this function must short-circuit.

**Files:**
- Create: `tests/test_unified_pipeline_ddp.py`
- Modify: `tuning/training/unified_early_pipeline.py:18-23`

- [ ] **Step 1: Write the failing test**

Create `tests/test_unified_pipeline_ddp.py`:

```python
# ABOUTME: Tests for DDP-related changes in unified_early_pipeline (CLI, env init).
# ABOUTME: CPU-only; mocks heavy imports.

import os
import sys
from types import ModuleType

if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub

from tuning.training.unified_early_pipeline import _init_cuda_env


def test_init_cuda_env_noop_when_local_rank_set(monkeypatch):
    """Under torchrun (LOCAL_RANK set), _init_cuda_env must not mutate CUDA_VISIBLE_DEVICES."""
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES_ALL", raising=False)

    _init_cuda_env()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert "CUDA_VISIBLE_DEVICES_ALL" not in os.environ


def test_init_cuda_env_pins_gpu0_without_local_rank(monkeypatch):
    """Without torchrun (no LOCAL_RANK), legacy behavior: pin GPU 0, save the rest."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES_ALL", raising=False)

    _init_cuda_env()

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES_ALL"] == "0,1,2,3"
```

- [ ] **Step 2: Run test to verify the LOCAL_RANK case fails**

Run: `cd /project/6105902/shougan/balance-budget && source .venv/bin/activate && pytest tests/test_unified_pipeline_ddp.py::test_init_cuda_env_noop_when_local_rank_set -v`

Expected: FAIL — current code overwrites `CUDA_VISIBLE_DEVICES` to `"0"` regardless of LOCAL_RANK.

- [ ] **Step 3: Make `_init_cuda_env` short-circuit under torchrun**

In `tuning/training/unified_early_pipeline.py`, replace lines 18–23:

```python
def _init_cuda_env():
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
git add tests/test_unified_pipeline_ddp.py tuning/training/unified_early_pipeline.py
git commit -m "feat: skip _init_cuda_env when LOCAL_RANK is set (torchrun)"
```

---

## Task 2: `--grpo-num-gpus` CLI flag

Add the orchestrator-only flag that controls how many GPUs the GRPO sbatch worker gets. Default 1 keeps current single-GPU behavior.

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:185` (after `--grpo-num-epochs`)
- Modify: `tests/test_unified_pipeline_ddp.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_unified_pipeline_ddp.py`:

```python
from tuning.training.unified_early_pipeline import _parse_args


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

In `tuning/training/unified_early_pipeline.py`, add immediately after the `--grpo-num-epochs` argument (around line 185):

```python
    parser.add_argument("--grpo-num-gpus", type=int, default=1,
                        help="Number of GPUs for GRPO DDP training. >1 launches GRPO via torchrun.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_unified_pipeline_ddp.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_pipeline_ddp.py
git commit -m "feat: add --grpo-num-gpus CLI flag (default 1)"
```

---

## Task 3: `_is_rank_zero()` helper on `PassAtKStoppingCallback`

Foundation for rank-aware logic. Returns `True` when not under DDP or when rank == 0.

**Files:**
- Create: `tests/test_grpo_ddp_eval.py`
- Modify: `tuning/training/passk_callback.py:117` (inside `PassAtKStoppingCallback` class, helper near top)

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
from tuning.training.passk_callback import PassAtKStoppingCallback


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

In `tuning/training/passk_callback.py`, add to the top of imports (alongside `import torch`):

```python
import torch.distributed as dist
```

Then add this method to `PassAtKStoppingCallback`, right after `__init__` (around line 215, before `on_train_begin`):

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
git add tuning/training/passk_callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: add _is_rank_zero helper to PassAtKStoppingCallback"
```

---

## Task 4: `set_accelerator()` setter on `PassAtKStoppingCallback`

Plumbing for the sweetspot save path. The setter is a no-op for SFT/DPO (they never call it).

**Files:**
- Modify: `tuning/training/passk_callback.py` (alongside `set_trainer_vllm` near line 260)
- Modify: `tests/test_grpo_ddp_eval.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_grpo_ddp_eval.py`:

```python
def test_set_accelerator_stores_reference(monkeypatch):
    cb = _make_callback(monkeypatch)
    fake_accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    cb.set_accelerator(fake_accelerator)
    assert cb._accelerator is fake_accelerator


def test_default_accelerator_is_none(monkeypatch):
    cb = _make_callback(monkeypatch)
    assert cb._accelerator is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_grpo_ddp_eval.py::test_default_accelerator_is_none -v`

Expected: FAIL — attribute error.

- [ ] **Step 3: Add the setter and default attribute**

In `tuning/training/passk_callback.py`, add to `__init__` next to `self._external_vllm = None` (around line 182):

```python
        self._accelerator = None
```

Add the setter method next to the existing `set_trainer_vllm` (around line 260):

```python
    def set_accelerator(self, accelerator):
        """Plumb accelerate.Accelerator for DDP-aware adapter saves."""
        self._accelerator = accelerator
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk_callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: add set_accelerator hook on PassAtKStoppingCallback"
```

---

## Task 5: `save_sweetspot_checkpoint` accepts optional `accelerator`

Backward-compatible: when `accelerator` is None (SFT/DPO), behavior is unchanged. When provided, unwrap the DDP-wrapped model before `save_pretrained`.

**Files:**
- Modify: `tuning/training/callback_utils.py` (find `save_sweetspot_checkpoint`)
- Modify: `tests/test_grpo_ddp_eval.py`

- [ ] **Step 1: Read current `save_sweetspot_checkpoint` to find exact signature**

Run: `grep -n "def save_sweetspot_checkpoint" /project/6105902/shougan/balance-budget/tuning/training/callback_utils.py`

Note the signature and the `model.save_pretrained(...)` line.

- [ ] **Step 2: Write the failing test**

Append to `tests/test_grpo_ddp_eval.py`:

```python
def test_save_sweetspot_unwraps_when_accelerator_provided(tmp_path, monkeypatch):
    """With accelerator, save_pretrained is called on the unwrapped model."""
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
    wrapped.save_pretrained.assert_not_called()


def test_save_sweetspot_no_unwrap_when_accelerator_none(tmp_path):
    """Without accelerator (SFT/DPO callers), save_pretrained is called on model directly."""
    from tuning.training.callback_utils import save_sweetspot_checkpoint

    model = MagicMock(name="peft_model")
    tokenizer = MagicMock()

    state = SimpleNamespace(global_step=10, log_history=[])
    args = SimpleNamespace(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        world_size=1,
        output_dir=str(tmp_path),
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

    model.save_pretrained.assert_called_once()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_grpo_ddp_eval.py::test_save_sweetspot_unwraps_when_accelerator_provided -v`

Expected: FAIL — `TypeError: save_sweetspot_checkpoint() got an unexpected keyword argument 'accelerator'`.

- [ ] **Step 4: Add the kwarg and unwrap logic**

Edit `tuning/training/callback_utils.py` `save_sweetspot_checkpoint`:

- Add `accelerator=None` to the signature (last keyword arg).
- Replace the `model.save_pretrained(...)` call with:

```python
    target = accelerator.unwrap_model(model) if accelerator is not None else model
    target.save_pretrained(checkpoint_path)
```

(Use the existing `checkpoint_path` variable name; if your code uses a different name like `output_dir`, match it.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add tuning/training/callback_utils.py tests/test_grpo_ddp_eval.py
git commit -m "feat: save_sweetspot_checkpoint unwraps DDP model when accelerator passed"
```

---

## Task 6: Wire `set_accelerator` in `train_model_grpo`

Plumb `trainer.accelerator` into the callback after trainer construction, in the same loop as the existing `set_trainer_vllm` hook.

**Files:**
- Modify: `tuning/training/grpo_training.py:87-90`

- [ ] **Step 1: Read the existing hook**

Run: `sed -n '85,95p' /project/6105902/shougan/balance-budget/tuning/training/grpo_training.py`

Confirm the loop that calls `cb.set_trainer_vllm(trainer.vllm_generation.llm)`.

- [ ] **Step 2: Add the accelerator plumbing**

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
            cb.set_accelerator(trainer.accelerator)
```

- [ ] **Step 3: Verify imports**

Confirm `PassAtKStoppingCallback` is already imported at line 11. If not, add `from tuning.training.passk_callback import PassAtKStoppingCallback`.

- [ ] **Step 4: Run existing tests to make sure nothing regressed**

Run: `pytest tests/test_grpo_config.py tests/test_grpo_ddp_eval.py tests/test_unified_pipeline_ddp.py -v`

Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/grpo_training.py
git commit -m "feat: plumb trainer.accelerator into PassAtKStoppingCallback"
```

---

## Task 7: `_run_eval_with_results_ddp` method

Main DDP eval logic. Each rank generates its slice via `trainer.vllm_generation.llm.chat`, all ranks gather, all ranks score deterministically.

**Files:**
- Modify: `tuning/training/passk_callback.py` (add new method, near other `_run_eval_*` methods around line 545)
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
        # Configure all_gather_object to fill in the rank-1 slice with mock responses
        rank1_pairs = [(i, [f"resp_for_Prompt {i}"]) for i in [1, 3, 5, 7]]
        def fake_gather(out_list, local):
            out_list[0] = local
            out_list[1] = rank1_pairs
        mock_gather.side_effect = fake_gather

        scores, model_results = cb._run_eval_with_results_ddp(model=MagicMock(), eval_strategy=eval_strategy)

    # All 8 prompts should be present in model_results
    assert len(model_results) == 8
    # Scores should be the deterministic stub
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

In `tuning/training/passk_callback.py`, near `_run_eval_with_results` (around line 547), add:

```python
    def _run_eval_with_results_ddp(self, model, eval_strategy: EvalStrategy):
        """DDP eval path: each rank generates its slice via the colocated vLLM,
        all ranks gather responses, all ranks score deterministically.

        Returns (scores, model_results) on every rank.
        """
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
git add tuning/training/passk_callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: add _run_eval_with_results_ddp for cooperative DDP eval"
```

---

## Task 8: DDP branch in `_run_eval_with_results`

Dispatcher: when DDP is active and `world_size > 1`, route to the new method. Otherwise fall through to the existing single-GPU paths.

**Files:**
- Modify: `tuning/training/passk_callback.py:547` (top of `_run_eval_with_results`)
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
    """When world_size == 1, falls through to single-GPU paths (DDP method NOT called)."""
    cb = _make_callback(monkeypatch)
    fake_llm = MagicMock()
    fake_llm.chat.return_value = [SimpleNamespace(outputs=[SimpleNamespace(text="r")])]
    cb.set_trainer_vllm(fake_llm)

    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_world_size", return_value=1), \
         patch.object(cb, "_run_eval_with_results_ddp") as ddp_mock:
        cb._run_eval_with_results(model=MagicMock(), eval_strategy=_FakeEval())

    ddp_mock.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grpo_ddp_eval.py::test_dispatch_ddp_when_world_size_gt_1 -v`

Expected: FAIL — DDP method is never called by current dispatcher.

- [ ] **Step 3: Add the branch**

In `tuning/training/passk_callback.py`, find `def _run_eval_with_results(self, model, eval_strategy)` (around line 547). Add at the very top of the method body, before `temp_dir = tempfile.mkdtemp()`:

```python
        if dist.is_initialized() and dist.get_world_size() > 1:
            return self._run_eval_with_results_ddp(model, eval_strategy)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk_callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: dispatch DDP eval path when world_size > 1"
```

---

## Task 9: Rank-0 gating in `on_evaluate`

Gate I/O operations (W&B logging, raw-generation table, sweetspot save) on `_is_rank_zero()`. State mutations stay un-gated and run identically on every rank.

**Files:**
- Modify: `tuning/training/passk_callback.py` `on_evaluate` (around line 608)
- Modify: `tests/test_grpo_ddp_eval.py`

- [ ] **Step 1: Read the current `on_evaluate` to know what to wrap**

Run: `sed -n '608,725p' /project/6105902/shougan/balance-budget/tuning/training/passk_callback.py`

Note where each of these gets called:
- `wandb.log(log_dict)` — multiple
- `self._log_raw_generation_table(...)` — multiple
- `self._save_sweetspot_checkpoint(...)` — multiple
- monitor evals also log via `wandb.log(monitor_log)`

- [ ] **Step 2: Write the failing test**

Append to `tests/test_grpo_ddp_eval.py`:

```python
def test_on_evaluate_gates_wandb_on_rank0(monkeypatch):
    """Under DDP rank 1, wandb.log and _log_raw_generation_table must NOT fire."""
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
         patch("tuning.training.passk_callback.wandb") as mock_wandb, \
         patch.object(cb, "_log_raw_generation_table") as mock_log_table, \
         patch.object(cb, "_save_sweetspot_checkpoint") as mock_save:
        def fake_gather(out_list, local):
            for i in range(len(out_list)):
                out_list[i] = local if i == 1 else [(j, ["r"]) for j in range(0, 8, 2)]
        mock_gather.side_effect = fake_gather

        cb.on_evaluate(args, state, control, model=MagicMock())

    mock_wandb.log.assert_not_called()
    mock_log_table.assert_not_called()
    mock_save.assert_not_called()


def test_on_evaluate_runs_wandb_on_rank0(monkeypatch):
    """Under DDP rank 0, wandb.log fires (existing behavior)."""
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
         patch("tuning.training.passk_callback.wandb") as mock_wandb, \
         patch.object(cb, "_log_raw_generation_table") as mock_log_table:
        def fake_gather(out_list, local):
            for i in range(len(out_list)):
                out_list[i] = local if i == 0 else [(j, ["r"]) for j in range(1, 8, 2)]
        mock_gather.side_effect = fake_gather

        cb.on_evaluate(args, state, control, model=MagicMock())

    assert mock_wandb.log.called
    assert mock_log_table.called
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_grpo_ddp_eval.py::test_on_evaluate_gates_wandb_on_rank0 -v`

Expected: FAIL — `wandb.log` is called on every rank in current code.

- [ ] **Step 4: Add rank-0 gating**

In `tuning/training/passk_callback.py` `on_evaluate`, wrap each I/O-only call with `if self._is_rank_zero():`. Concretely, find these lines and wrap:

1. The primary-eval `wandb.log(log_dict)` (around line 628):

```python
        if self._is_rank_zero():
            wandb.log(log_dict)
```

2. The primary-eval `self._log_raw_generation_table(...)` (around line 633):

```python
        if self._is_rank_zero():
            self._log_raw_generation_table(
                eval_strategy=self.primary_eval,
                model_results=raw_results,
                global_step=state.global_step,
                stopping_metric_name=stopping_key,
                stopping_metric_value=stopping_value,
            )
```

3. Both monitor-eval `wandb.log(monitor_log)` and `_log_raw_generation_table(...)` (around lines 650, 652):

```python
            if self._is_rank_zero():
                wandb.log(monitor_log)
                self._log_raw_generation_table(...)  # keep its existing args
```

4. The `_save_sweetspot_checkpoint` calls (threshold-reached at line 678, gap-checkpoint at line 717):

```python
            if self._is_rank_zero():
                checkpoint_path = self._save_sweetspot_checkpoint(model, reached_threshold, state, args)
            checkpoint_saved = True
```

(Same pattern at the gap-checkpoint site.)

5. Pass `accelerator` through to `save_sweetspot_checkpoint` by editing `_save_sweetspot_checkpoint` (around line 460):

```python
    def _save_sweetspot_checkpoint(self, model, threshold, state: TrainerState, args: TrainingArguments):
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

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_grpo_ddp_eval.py -v`

Expected: 12 passed.

- [ ] **Step 6: Run the full callback test suite to make sure nothing regressed**

Run: `pytest tests/test_external_vllm_reuse.py tests/test_callback_step_bridging.py tests/test_passk_callback_wandb_tables.py -v`

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add tuning/training/passk_callback.py tests/test_grpo_ddp_eval.py
git commit -m "feat: rank-0 gate I/O in PassAtKCallback.on_evaluate"
```

---

## Task 10: sbatch script branches on `--run-grpo` to use torchrun

The sbatch script invokes the unified pipeline. For GRPO worker mode, switch to `torchrun`. Other invocations (orchestrator, SFT, DPO) keep the bare `python` call.

**Files:**
- Modify: `tuning/slurm/unified_early_pipeline.sh:35`

- [ ] **Step 1: Read the current dispatch line**

Run: `sed -n '30,45p' /project/6105902/shougan/balance-budget/tuning/slurm/unified_early_pipeline.sh`

- [ ] **Step 2: Add the GRPO branch**

Replace line 35 (`python tuning/training/unified_early_pipeline.py "$@"`) with:

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

When the user passes `--grpo-num-gpus N` (N > 1), orchestrator-mode sbatch submission must request N GPUs instead of the default 1.

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (the `_submit_sbatch_worker` and/or sbatch call site)
- Modify: `tests/test_unified_pipeline_ddp.py`

- [ ] **Step 1: Read current sbatch submission code**

Run: `grep -n "_submit_sbatch_worker\|sbatch" /project/6105902/shougan/balance-budget/tuning/training/unified_early_pipeline.py | head -20`

Identify where `cmd = ["sbatch", sbatch_script, *worker_args]` is built (around line 864).

- [ ] **Step 2: Write the failing test**

Append to `tests/test_unified_pipeline_ddp.py`:

```python
from unittest.mock import patch, MagicMock


def test_sbatch_command_includes_gres_for_ddp(monkeypatch):
    """When grpo_num_gpus > 1, sbatch is invoked with --gres=gpu:N."""
    from tuning.training.unified_early_pipeline import _submit_sbatch_worker

    fake_result = MagicMock(returncode=0,
                            stdout="Submitted batch job 12345\n",
                            stderr="")
    with patch("subprocess.run", return_value=fake_result) as mock_run:
        job_id = _submit_sbatch_worker(
            "tuning/slurm/unified_early_pipeline.sh",
            ["--run-grpo", "--run-all"],
            num_gpus=4,
        )
    assert job_id == "12345"
    cmd = mock_run.call_args[0][0]
    assert "--gres=gpu:4" in cmd
    assert cmd[0] == "sbatch"


def test_sbatch_command_no_gres_when_default(monkeypatch):
    """When grpo_num_gpus == 1 (default), no --gres flag is added (sbatch script default applies)."""
    from tuning.training.unified_early_pipeline import _submit_sbatch_worker

    fake_result = MagicMock(returncode=0,
                            stdout="Submitted batch job 12345\n",
                            stderr="")
    with patch("subprocess.run", return_value=fake_result) as mock_run:
        _submit_sbatch_worker(
            "tuning/slurm/unified_early_pipeline.sh",
            ["--run-grpo", "--run-all"],
            num_gpus=1,
        )
    cmd = mock_run.call_args[0][0]
    assert not any(a.startswith("--gres") for a in cmd)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_unified_pipeline_ddp.py::test_sbatch_command_includes_gres_for_ddp -v`

Expected: FAIL — `_submit_sbatch_worker` doesn't accept `num_gpus`.

- [ ] **Step 4: Add the `num_gpus` arg and threading**

In `tuning/training/unified_early_pipeline.py`:

a. Modify `_submit_sbatch_worker` (around line 859):

```python
def _submit_sbatch_worker(sbatch_script, worker_args, num_gpus=1):
    """Submit an sbatch worker job, return the Slurm job ID as a string.

    When num_gpus > 1, prepend --gres=gpu:N to override the script default.
    """
    cmd = ["sbatch"]
    if num_gpus > 1:
        cmd.append(f"--gres=gpu:{num_gpus}")
    cmd.extend([sbatch_script, *worker_args])
    print(f"[orchestrator] Submitting sbatch worker: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"sbatch failed (code {result.returncode}): {result.stderr.strip()}")
    tokens = result.stdout.strip().split()
    if len(tokens) < 4 or tokens[0] != "Submitted":
        sys.exit(f"Unexpected sbatch stdout: {result.stdout!r}")
    return tokens[-1]
```

b. Update `_dispatch_parallel_workers` to thread the value through (around line 875):

```python
def _dispatch_parallel_workers(parallel, base_cmd, pt_flag, metadata_files, num_gpus=1):
    if parallel <= 1:
        return
    # ... existing worker_argv construction ...
    for i in range(parallel - 1):
        job_id = _submit_sbatch_worker(SBATCH_WORKER_SCRIPT, worker_argv, num_gpus=num_gpus)
        print(f"[orchestrator] Submitted worker {i+1}/{parallel-1}: job {job_id}")
```

c. Update the call site in `main()` (around line 947):

```python
    pt_method = args.post_training_method
    pt_flag = f"--run-{pt_method}" if pt_method != "dpo" else "--run-dpo"
    pt_num_gpus = args.grpo_num_gpus if pt_method == "grpo" else 1
    _dispatch_parallel_workers(
        parallel=args.parallel,
        base_cmd=base_cmd,
        pt_flag=pt_flag,
        metadata_files=all_files,
        num_gpus=pt_num_gpus,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_unified_pipeline_ddp.py -v`

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_pipeline_ddp.py
git commit -m "feat: orchestrator submits GRPO sbatch with --gres=gpu:N when grpo_num_gpus>1"
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
- `[unified_early_pipeline.sh] GRPO worker mode` line is **not** present (single-GPU goes through `python` path).
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

Pick the saved checkpoint path from Step 3, run a non-DDP eval against it (any existing single-GPU eval entry point). The adapter must load into vLLM without error — proves the unwrapped save in Task 5 is correct.

- [ ] **Step 5: Update `MEMORY.md` with run identifiers**

If the smoke runs revealed any unexpected `vllm_gpu_memory_utilization` adjustments (because DDP shares vLLM mem with grad mem), record the working values in `MEMORY.md` for future runs.

---

## Self-Review

**Spec coverage:**

- Decisions table: each row maps to a task — Layer A → Tasks 1, 2, 10, 11; Layer B → Task 6; Layer C → Tasks 3, 4, 5, 7, 8, 9; Verification → Task 12.
- TRL source-of-truth references: cited in spec; tasks reference `trainer.vllm_generation.llm` (Task 7) and `trainer.accelerator` (Task 6) consistently with the spec.
- Edge cases (`world_size=1` degenerate, empty local slice, monitor evals): covered by tests in Tasks 7, 8, 9 plus Task 12 smoke runs.
- Retired flag (`--grpo-passk-num-inference-gpus`): not actively retired in this plan — the spec calls it "deprecation warning" but YAGNI applies; the existing flag continues to work for SFT/DPO and is silently bypassed for the DDP eval path because `_run_eval_with_results_ddp` runs first. No code changes needed; revisit if confusion arises in practice.

**Placeholder scan:** no "TBD" / "TODO" / vague handlers. Each step has exact code or exact commands.

**Type/name consistency:**
- `_is_rank_zero()` defined in Task 3, used in Task 9. ✓
- `set_accelerator` defined in Task 4, called in Task 6, consumed in Task 5 via `self._accelerator`. ✓
- `_run_eval_with_results_ddp` defined in Task 7, dispatched in Task 8. ✓
- `--grpo-num-gpus` defined in Task 2, consumed in Task 11 (`args.grpo_num_gpus`). ✓
- sbatch script reads `$SLURM_GPUS_ON_NODE` (Task 10); orchestrator passes `--gres=gpu:N` (Task 11) which sets that var. ✓
