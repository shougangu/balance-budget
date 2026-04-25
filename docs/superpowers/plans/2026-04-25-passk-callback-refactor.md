# Pass@K Callback Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `tuning/training/passk_callback.py` (724 lines) into a `passk/` subpackage of focused modules, replace the 4-way vLLM-mode ladder with a `VLLMRunner` strategy, extract checkpoint decision logic into a pure-function engine, and slim `on_evaluate` to a short orchestrator.

**Architecture:** New subpackage `tuning/training/passk/` containing `callback.py`, `runners.py`, `data_parallel.py`, `logging.py`, `decisions.py`. `tuning/training/passk_callback.py` collapses to a one-line shim that re-exports `PassAtKStoppingCallback` (production code uses this import path heavily). Behavior preserved end-to-end; the persistent→ephemeral fallback now swaps the runner instance instead of duplicating the ephemeral code path.

**Tech Stack:** Python 3, `transformers` `TrainerCallback`, `vllm` (mocked in unit tests), `wandb`, `pytest`. No new dependencies.

**Reference spec:** `docs/superpowers/specs/2026-04-25-passk-pipeline-refactor-design.md`

**Working directory:** Implement directly on `main` per project memory (`feedback_implement_on_main.md`).

---

## Task 1: Create passk subpackage skeleton

**Files:**
- Create: `tuning/training/passk/__init__.py`

- [ ] **Step 1: Create empty package init**

```python
# ABOUTME: Pass@K callback subpackage — split from the previous monolithic passk_callback.py.
# ABOUTME: Public surface re-exported below; internals live in sibling modules.
```

Write to `tuning/training/passk/__init__.py`. Leave the body bare (no imports yet) — we'll add the `PassAtKStoppingCallback` re-export in Task 12 once it has moved.

- [ ] **Step 2: Verify package imports**

Run: `python -c "import tuning.training.passk"`
Expected: exits 0 with no output.

- [ ] **Step 3: Commit**

```bash
git add tuning/training/passk/__init__.py
git commit -m "refactor(passk): scaffold passk subpackage"
```

---

## Task 2: Move data_parallel helpers into new module

**Files:**
- Create: `tuning/training/passk/data_parallel.py`
- Modify: `tuning/training/passk_callback.py` (remove `partition_prompts`, `_data_parallel_worker`; import from new module)
- Modify: `tests/test_multi_gpu_inference.py` (update `partition_prompts` import path)
- Modify: `tests/test_seed_wiring.py` (update `_data_parallel_worker` import path)

The `partition_prompts` function (passk_callback.py:28-43) and `_data_parallel_worker` subprocess entry point (passk_callback.py:46-115) move verbatim. No logic changes here — this is a pure relocation that confirms the test-import update workflow before bigger surgery.

- [ ] **Step 1: Create new module with the moved code**

Create `tuning/training/passk/data_parallel.py` with this exact content:

```python
# ABOUTME: Helpers for data-parallel vLLM inference across multiple GPUs.
# ABOUTME: _data_parallel_worker is a subprocess entry point — keep top-level (no closures).

import os
from typing import List

import torch


def partition_prompts(messages: List, num_chunks: int) -> List[List]:
    """Split a list of messages into num_chunks roughly-equal chunks.

    If num_chunks > len(messages), only len(messages) chunks are returned (1 item each).
    """
    n = len(messages)
    num_chunks = min(num_chunks, n)
    chunks = []
    base_size = n // num_chunks
    remainder = n % num_chunks
    start = 0
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        chunks.append(messages[start:start + size])
        start += size
    return chunks


def _data_parallel_worker(worker_id, cuda_device, messages_chunk, base_model_hf, adapter_path,
                          n_samples, temperature, max_tokens, chat_template,
                          lora_max_rank, gpu_memory_utilization, result_queue,
                          stop_tokens=None, seed=None):
    """Worker function for data-parallel vLLM inference. Runs in a subprocess.

    Each worker pins itself to a single GPU, creates an ephemeral vLLM engine,
    runs inference on its chunk of prompts, and returns serialized outputs.

    Args:
        worker_id: Logical worker index (0, 1, 2...) used for result ordering.
        cuda_device: The actual CUDA device string (e.g. "3") from SLURM allocation.
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=base_model_hf,
            enable_lora=True,
            max_lora_rank=lora_max_rank,
            max_loras=1,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
            enforce_eager=True,
        )

        from tuning.inference.config_inference import VLLMSamplingParamsConfig
        inference_config = VLLMSamplingParamsConfig(
            n=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_tokens or [],
            seed=seed,
        )
        sampling_params = SamplingParams(**inference_config.model_dump())

        lora_request = LoRARequest(
            lora_name=f"adapter_worker{worker_id}",
            lora_int_id=1,
            lora_path=adapter_path,
        )

        outputs = llm.chat(
            messages_chunk,
            sampling_params,
            chat_template=chat_template,
            lora_request=lora_request,
        )

        serialized = []
        for output in outputs:
            texts = [resp.text for resp in output.outputs]
            serialized.append(texts)

        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        result_queue.put((worker_id, serialized, None))
    except Exception:
        import traceback
        result_queue.put((worker_id, None, traceback.format_exc()))
```

- [ ] **Step 2: Remove the moved code from `passk_callback.py` and import from the new module**

In `tuning/training/passk_callback.py`:

1. Delete lines 28-115 (the `partition_prompts` and `_data_parallel_worker` definitions).
2. Add this import near the existing imports:

```python
from tuning.training.passk.data_parallel import partition_prompts, _data_parallel_worker
```

(Keep the import — `_run_data_parallel_inference` inside the class still references `_data_parallel_worker` and `partition_prompts`. They're now imported names.)

- [ ] **Step 3: Update test imports**

In `tests/test_multi_gpu_inference.py:15`, change:
```python
from tuning.training.passk_callback import partition_prompts
```
to:
```python
from tuning.training.passk.data_parallel import partition_prompts
```

In `tests/test_seed_wiring.py:60`, change:
```python
from tuning.training.passk_callback import _data_parallel_worker
```
to:
```python
from tuning.training.passk.data_parallel import _data_parallel_worker
```

- [ ] **Step 4: Run affected tests**

Run: `pytest tests/test_multi_gpu_inference.py tests/test_seed_wiring.py -v`
Expected: all tests pass.

- [ ] **Step 5: Run callback tests for regression check**

Run: `pytest tests/test_callback_step_bridging.py tests/test_external_vllm_reuse.py tests/test_passk_callback_wandb_tables.py -v`
Expected: all tests pass (callback class still works because it imports from the new module).

- [ ] **Step 6: Commit**

```bash
git add tuning/training/passk/data_parallel.py tuning/training/passk_callback.py \
        tests/test_multi_gpu_inference.py tests/test_seed_wiring.py
git commit -m "refactor(passk): move data-parallel helpers to passk.data_parallel"
```

---

## Task 3: Add CheckpointDecisionEngine module (TDD, no integration yet)

**Files:**
- Create: `tuning/training/passk/decisions.py`
- Create: `tests/test_checkpoint_decision_engine.py`

Pure-logic module. Threshold sweep, early-tuple triggering, gap-checkpoint timing. Owns the lists/counters previously held directly on the callback. No `model`/`wandb`/`print` dependencies.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_checkpoint_decision_engine.py`:

```python
# ABOUTME: Unit tests for CheckpointDecisionEngine — pure threshold/early-tuple/gap logic.
# ABOUTME: No vllm or wandb mocks needed (engine has zero such deps).

from tuning.training.passk.decisions import CheckpointDecision, CheckpointDecisionEngine


def _engine(thresholds=None, early_tuples=None, max_gap=None):
    return CheckpointDecisionEngine(
        target_thresholds=thresholds or [],
        early_tuples=early_tuples or None,
        max_checkpoint_gap=max_gap,
    )


class TestThresholdSweep:
    def test_sorted_descending_on_init(self):
        eng = _engine(thresholds=[0.3, 0.7, 0.5])
        assert eng.target_thresholds == [0.7, 0.5, 0.3]

    def test_no_decision_when_below_all(self):
        eng = _engine(thresholds=[0.7, 0.5, 0.3])
        decisions = eng.decide(primary_metric=0.2, history=[0.2],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == []

    def test_picks_hardest_reached_and_trims(self):
        eng = _engine(thresholds=[0.7, 0.5, 0.3])
        decisions = eng.decide(primary_metric=0.55, history=[0.55],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="0.5", advances_state=True)]
        assert eng.target_thresholds == [0.7]

    def test_subsequent_call_after_threshold_consumed(self):
        eng = _engine(thresholds=[0.7, 0.5, 0.3])
        eng.decide(primary_metric=0.55, history=[0.55],
                   data_points_seen=100, last_checkpoint_data_points=0)
        decisions = eng.decide(primary_metric=0.55, history=[0.55, 0.55],
                               data_points_seen=200, last_checkpoint_data_points=100)
        assert decisions == []


class TestEarlyTuples:
    def test_no_trigger_when_history_too_short(self):
        eng = _engine(early_tuples=[(2, 0.05)])
        decisions = eng.decide(primary_metric=0.5, history=[0.5, 0.5],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == []

    def test_triggers_when_no_increase_over_window(self):
        eng = _engine(early_tuples=[(2, 0.05)])
        decisions = eng.decide(primary_metric=0.5,
                               history=[0.5, 0.5, 0.5],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="2@0.05", advances_state=True)]
        assert eng.early_tuples == []

    def test_does_not_trigger_when_increase_seen(self):
        eng = _engine(early_tuples=[(2, 0.05)])
        decisions = eng.decide(primary_metric=0.6,
                               history=[0.4, 0.5, 0.6],
                               data_points_seen=100, last_checkpoint_data_points=0)
        assert decisions == []
        assert eng.early_tuples == [(2, 0.05)]


class TestGapCheckpoint:
    def test_no_gap_when_disabled(self):
        eng = _engine(max_gap=None)
        decisions = eng.decide(primary_metric=0.0, history=[0.0],
                               data_points_seen=10000, last_checkpoint_data_points=0)
        assert decisions == []

    def test_gap_triggers_when_distance_exceeds_max(self):
        eng = _engine(max_gap=5000)
        decisions = eng.decide(primary_metric=0.42, history=[0.42],
                               data_points_seen=6000, last_checkpoint_data_points=0)
        assert decisions == [CheckpointDecision(label="gap-6000-0.42", advances_state=True)]

    def test_gap_skipped_when_threshold_already_fired(self):
        eng = _engine(thresholds=[0.5], max_gap=1000)
        decisions = eng.decide(primary_metric=0.6,
                               history=[0.6],
                               data_points_seen=2000, last_checkpoint_data_points=0)
        assert len(decisions) == 1
        assert decisions[0].label == "0.5"
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/test_checkpoint_decision_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuning.training.passk.decisions'`.

- [ ] **Step 3: Implement the engine**

Create `tuning/training/passk/decisions.py`:

```python
# ABOUTME: Pure-logic engine that decides when to save sweetspot checkpoints.
# ABOUTME: Owns threshold list, early-tuple list, max-gap counter — no W&B / model deps.

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CheckpointDecision:
    label: str
    advances_state: bool


class CheckpointDecisionEngine:
    def __init__(
        self,
        target_thresholds: List[float],
        early_tuples: Optional[List[Tuple[int, float]]],
        max_checkpoint_gap: Optional[int],
    ):
        self.target_thresholds = sorted(target_thresholds, reverse=True)
        self.early_tuples = list(early_tuples) if early_tuples else None
        self.max_checkpoint_gap = max_checkpoint_gap

    def decide(
        self,
        primary_metric: float,
        history: List[float],
        data_points_seen: int,
        last_checkpoint_data_points: int,
    ) -> List[CheckpointDecision]:
        decisions: List[CheckpointDecision] = []

        # Threshold sweep: pick the hardest threshold reached, trim the rest below it.
        if self.target_thresholds:
            reached_index = None
            reached_threshold = None
            for i, threshold in enumerate(self.target_thresholds):
                if primary_metric >= threshold:
                    reached_index = i
                    reached_threshold = threshold
                    break
            if reached_threshold is not None:
                decisions.append(CheckpointDecision(
                    label=str(reached_threshold), advances_state=True
                ))
                self.target_thresholds = self.target_thresholds[:reached_index]

        # Early tuples: trigger when last `patience` deltas all stayed under min_increase.
        if self.early_tuples is not None:
            triggered_idx = []
            for idx, (patience, min_increase) in enumerate(self.early_tuples):
                if len(history) > patience:
                    early_stopping = True
                    for old, new in zip(history[-patience-1:], history[-patience:]):
                        if new - old >= min_increase:
                            early_stopping = False
                            break
                    if early_stopping:
                        decisions.append(CheckpointDecision(
                            label=f"{patience}@{min_increase}",
                            advances_state=True,
                        ))
                        triggered_idx.append(idx)
            for idx in reversed(triggered_idx):
                self.early_tuples.pop(idx)

        # Gap checkpoint: only when no other decision fired this call.
        if (self.max_checkpoint_gap is not None
                and data_points_seen > 0
                and not decisions):
            gap = data_points_seen - last_checkpoint_data_points
            if gap >= self.max_checkpoint_gap:
                decisions.append(CheckpointDecision(
                    label=f"gap-{data_points_seen}-{primary_metric}",
                    advances_state=True,
                ))

        return decisions
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_checkpoint_decision_engine.py -v`
Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/decisions.py tests/test_checkpoint_decision_engine.py
git commit -m "refactor(passk): add CheckpointDecisionEngine module with unit tests"
```

---

## Task 4: Integrate CheckpointDecisionEngine into the callback

**Files:**
- Modify: `tuning/training/passk_callback.py` (replace inline threshold/early-tuple/gap logic in `on_evaluate`)

This is a behavior-preserving swap. The existing tests `test_callback_step_bridging.py` and `test_passk_callback_wandb_tables.py` already exercise the eval path and must keep passing.

- [ ] **Step 1: Add the engine import and construct it in `__init__`**

In `tuning/training/passk_callback.py`, after the existing imports add:

```python
from tuning.training.passk.decisions import CheckpointDecisionEngine
```

In `PassAtKStoppingCallback.__init__`, replace the threshold/early-tuples/max-gap storage with engine construction. Concretely, replace line 144 (`self.target_pass_at_k_thresholds = sorted(...)`), line 145 (`self.early_tuples = list(...)`), and line 154 (`self.max_checkpoint_gap = getattr(...)`) with:

```python
        # Replace direct fields — the engine owns this state from now on.
        self._decision_engine = CheckpointDecisionEngine(
            target_thresholds=config.target_pass_at_k,
            early_tuples=config.early_tuples,
            max_checkpoint_gap=getattr(config, "max_checkpoint_gap", None),
        )
        # Public-ish field kept for the existing init-time print (line ~191).
        self.target_pass_at_k_thresholds = self._decision_engine.target_thresholds
        self.early_tuples = self._decision_engine.early_tuples
```

(Keep `self.target_pass_at_k_thresholds` and `self.early_tuples` as read-only mirrors used only by the `__init__` print statements. They no longer drive logic. The mirrors are removed in Task 11.)

Remove the now-redundant `self.max_checkpoint_gap = getattr(...)` and `self._last_checkpoint_data_points = 0` is unchanged.

- [ ] **Step 2: Replace the threshold/early-tuple/gap block in `on_evaluate`**

Locate `on_evaluate` (around line 608) and delete the block from `# Check each threshold ...` (line 662) through the end of the gap-checkpoint block (line 721). Replace with:

```python
        decisions = self._decision_engine.decide(
            primary_metric=stopping_value,
            history=self.prevResults,
            data_points_seen=data_points_seen,
            last_checkpoint_data_points=self._last_checkpoint_data_points,
        )
        for decision in decisions:
            self._save_sweetspot_checkpoint(model, decision.label, state, args)
            if decision.advances_state:
                self._last_checkpoint_data_points = data_points_seen
            print(f"[PassAtKCallback] Saved checkpoint: {decision.label}")
```

(The init-time print mirrors `self.target_pass_at_k_thresholds` and `self.early_tuples` set in step 1 are stale aliases by design — they're only read once at construction time before any `decide()` call mutates engine state. No need to refresh them inside `on_evaluate`. They're removed in Task 11.)

The `_save_sweetspot_checkpoint` second arg currently takes either a float (threshold) or a string (early-tuple label / gap label). Verify in `tuning/training/callback_utils.py:save_sweetspot_checkpoint` that the threshold_label arg accepts both — it does (it's wrapped in an f-string). `decision.label` is always a string, so the call is consistent.

- [ ] **Step 3: Run callback tests**

Run: `pytest tests/test_callback_step_bridging.py tests/test_passk_callback_wandb_tables.py tests/test_checkpoint_decision_engine.py -v`
Expected: all tests pass.

- [ ] **Step 4: Run remaining passk-callback tests**

Run: `pytest tests/test_external_vllm_reuse.py tests/test_eval_strategy.py tests/test_multi_gpu_inference.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk_callback.py
git commit -m "refactor(passk): replace inline threshold/early/gap logic with decision engine"
```

---

## Task 5: Add VLLMRunner base class + RunnerConfig

**Files:**
- Create: `tuning/training/passk/runners.py`
- Create: `tests/test_vllm_runners.py`

A small base class with the shared inference machinery (sampling params, `llm.chat`, output formatting, model offload context manager). Subclasses come in Tasks 6-8.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vllm_runners.py`:

```python
# ABOUTME: Tests for VLLMRunner strategy — selection, fallback, and per-runner behavior.
# ABOUTME: vLLM is mocked; we test the dispatch shape, not real generation.

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from tuning.training.passk.runners import RunnerConfig, VLLMRunner


def test_runner_config_defaults_are_explicit():
    cfg = RunnerConfig(
        base_model_hf="m",
        vllm_gpu_memory_utilization=0.6,
        lora_max_rank=32,
        chat_template="t",
        temperature=0.5,
        max_tokens=256,
        available_gpus=["0"],
        num_inference_gpus=1,
    )
    assert cfg.base_model_hf == "m"
    assert cfg.vllm_gpu_memory_utilization == 0.6


def test_base_runner_is_abstract():
    cfg = RunnerConfig(
        base_model_hf="m", vllm_gpu_memory_utilization=0.6, lora_max_rank=32,
        chat_template="t", temperature=0.5, max_tokens=256,
        available_gpus=["0"], num_inference_gpus=1,
    )
    with __import__("pytest").raises(NotImplementedError):
        VLLMRunner(cfg).run(model=None, eval_strategy=None, adapter_path=None)
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/test_vllm_runners.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tuning.training.passk.runners'`.

- [ ] **Step 3: Implement `RunnerConfig` and `VLLMRunner` base**

Create `tuning/training/passk/runners.py`:

```python
# ABOUTME: VLLMRunner strategy — one runner per inference mode (External / Persistent /
# ABOUTME: Ephemeral / DataParallel). Shared offload context + inference shape live here.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from collections import defaultdict


@dataclass
class RunnerConfig:
    base_model_hf: str
    vllm_gpu_memory_utilization: float
    lora_max_rank: int
    chat_template: str
    temperature: float
    max_tokens: int
    available_gpus: List[str]
    num_inference_gpus: int


class VLLMRunner:
    """Base class. Subclasses override `run`; `cleanup` is optional."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._lora_request_id = 0

    def run(self, model, eval_strategy, adapter_path: Optional[str]) -> List[Dict]:
        raise NotImplementedError

    def cleanup(self) -> None:
        return None

    def _next_lora_request(self, adapter_path: Optional[str]):
        if adapter_path is None:
            return None
        from vllm.lora.request import LoRARequest
        self._lora_request_id += 1
        return LoRARequest(
            lora_name=f"adapter_{self._lora_request_id}",
            lora_int_id=self._lora_request_id,
            lora_path=adapter_path,
        )

    def _run_inference(self, llm, eval_strategy, adapter_path: Optional[str]) -> List[Dict]:
        from vllm import SamplingParams
        from tuning.inference.config_inference import VLLMSamplingParamsConfig

        inference_config = VLLMSamplingParamsConfig(
            n=eval_strategy.n_samples,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        sampling_params = SamplingParams(**inference_config.model_dump())
        lora_request = self._next_lora_request(adapter_path)

        test_messages = eval_strategy.get_test_messages()
        outputs = llm.chat(
            test_messages,
            sampling_params,
            chat_template=self.config.chat_template,
            lora_request=lora_request,
        )
        return self._format_outputs(outputs, eval_strategy)

    @staticmethod
    def _format_outputs(outputs, eval_strategy) -> List[Dict]:
        n_samples = eval_strategy.n_samples
        if n_samples == 1:
            responses = [output.outputs[0].text for output in outputs]
        else:
            responses = [[r.text for r in output.outputs] for output in outputs]
        test_prompts = eval_strategy.get_test_prompts()
        grouped = defaultdict(list)
        for prompt, resp in zip(test_prompts, responses):
            if isinstance(resp, list):
                grouped[prompt].extend(resp)
            else:
                grouped[prompt].append(resp)
        return [{"prompt": p, "responses": resps} for p, resps in grouped.items()]

    @contextmanager
    def _with_model_offloaded(self, model):
        original_device = next(model.parameters()).device
        model.cpu()
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            model.to(original_device)
            model.train()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vllm_runners.py -v`
Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/runners.py tests/test_vllm_runners.py
git commit -m "refactor(passk): add VLLMRunner base class + RunnerConfig"
```

---

## Task 6: Add `ExternalVLLMRunner`

**Files:**
- Modify: `tuning/training/passk/runners.py` (append class)
- Modify: `tests/test_vllm_runners.py` (append test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vllm_runners.py`:

```python
def _make_eval_strategy(n=1):
    es = MagicMock()
    es.n_samples = n
    es.get_test_messages.return_value = [[{"role": "user", "content": "hi"}]]
    es.get_test_prompts.return_value = ["hi"]
    return es


def _make_config():
    return RunnerConfig(
        base_model_hf="m", vllm_gpu_memory_utilization=0.6, lora_max_rank=32,
        chat_template="t", temperature=0.5, max_tokens=256,
        available_gpus=["0"], num_inference_gpus=1,
    )


def test_external_runner_uses_provided_llm_and_skips_lora():
    from tuning.training.passk.runners import ExternalVLLMRunner

    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    llm = MagicMock()
    llm.chat.return_value = [fake_output]

    runner = ExternalVLLMRunner(_make_config(), llm=llm)
    out = runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
                     adapter_path=None)

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    # External path always passes lora_request=None (no adapter swap on the trainer's vLLM).
    args, kwargs = llm.chat.call_args
    assert kwargs["lora_request"] is None
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/test_vllm_runners.py::test_external_runner_uses_provided_llm_and_skips_lora -v`
Expected: FAIL with `ImportError: cannot import name 'ExternalVLLMRunner'`.

- [ ] **Step 3: Implement `ExternalVLLMRunner`**

Append to `tuning/training/passk/runners.py`:

```python
class ExternalVLLMRunner(VLLMRunner):
    """Uses an externally-provided LLM (e.g. the trainer's own vLLM). No adapter save."""

    def __init__(self, config: RunnerConfig, llm):
        super().__init__(config)
        self._llm = llm

    def run(self, model, eval_strategy, adapter_path):
        # External LLM owns its own adapter handling; we never pass a LoRARequest.
        return self._run_inference(self._llm, eval_strategy, adapter_path=None)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vllm_runners.py -v`
Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/runners.py tests/test_vllm_runners.py
git commit -m "refactor(passk): add ExternalVLLMRunner"
```

---

## Task 7: Add `EphemeralVLLMRunner` and `PersistentVLLMRunner`

**Files:**
- Modify: `tuning/training/passk/runners.py`
- Modify: `tests/test_vllm_runners.py`

These both need to construct vLLM engines, so they're added together. Tests mock `vllm.LLM` so no real init happens.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vllm_runners.py`:

```python
def test_ephemeral_runner_creates_and_destroys_llm(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    fake_llm.chat.return_value = [fake_output]

    monkeypatch.setattr(runners_mod, "_make_llm", lambda cfg: fake_llm)
    cleanup_calls = []
    monkeypatch.setattr(runners_mod, "_cleanup_llm",
                        lambda llm: cleanup_calls.append(llm))

    model = MagicMock()
    model.parameters.return_value = iter([MagicMock(device="cuda:0")])

    runner = runners_mod.EphemeralVLLMRunner(_make_config())
    out = runner.run(model=model, eval_strategy=_make_eval_strategy(),
                     adapter_path="/tmp/adapter")

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    assert cleanup_calls == [fake_llm]
    # Model offload + restore happened.
    model.cpu.assert_called_once()
    model.to.assert_called_once()
    model.train.assert_called_once()


def test_persistent_runner_keeps_llm_alive(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    fake_llm.chat.return_value = [fake_output]

    make_calls = []
    monkeypatch.setattr(runners_mod, "_make_llm",
                        lambda cfg: (make_calls.append(cfg), fake_llm)[1])

    runner = runners_mod.PersistentVLLMRunner(_make_config())
    runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
               adapter_path="/tmp/a")
    runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
               adapter_path="/tmp/a")

    # LLM constructed exactly once across two run() calls.
    assert len(make_calls) == 1
```

- [ ] **Step 2: Run the failing tests**

Run: `pytest tests/test_vllm_runners.py::test_ephemeral_runner_creates_and_destroys_llm tests/test_vllm_runners.py::test_persistent_runner_keeps_llm_alive -v`
Expected: FAIL with `AttributeError` / missing class.

- [ ] **Step 3: Implement the runners and the helpers**

Append to `tuning/training/passk/runners.py`:

```python
def _make_llm(config: RunnerConfig):
    """Construct a vLLM LLM with our standard LoRA settings.

    enforce_eager=True is required: CUDA-graph capture is incompatible with dynamic
    LoRA adapter swapping.
    """
    from vllm import LLM
    return LLM(
        model=config.base_model_hf,
        enable_lora=True,
        max_lora_rank=config.lora_max_rank,
        max_loras=1,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )


def _cleanup_llm(llm):
    """Tear down an ephemeral LLM and free GPU memory."""
    from vllm.distributed.parallel_state import destroy_model_parallel
    from tuning.utils.gpu import cleanup_gpu

    llm.llm_engine.engine_core.shutdown()
    destroy_model_parallel()
    del llm
    cleanup_gpu()


class EphemeralVLLMRunner(VLLMRunner):
    """Creates a fresh vLLM engine per call; offloads training model to CPU."""

    def run(self, model, eval_strategy, adapter_path):
        with self._with_model_offloaded(model):
            llm = _make_llm(self.config)
            try:
                return self._run_inference(llm, eval_strategy, adapter_path)
            finally:
                _cleanup_llm(llm)


class PersistentVLLMRunner(VLLMRunner):
    """Keeps a persistent vLLM engine across calls; swaps LoRA adapters."""

    def __init__(self, config: RunnerConfig):
        super().__init__(config)
        self._llm = None

    def run(self, model, eval_strategy, adapter_path):
        if self._llm is None:
            self._llm = _make_llm(self.config)
        return self._run_inference(self._llm, eval_strategy, adapter_path)

    def cleanup(self):
        if self._llm is None:
            return
        try:
            llm_engine = getattr(self._llm, "llm_engine", None)
            if llm_engine is not None:
                executor = getattr(llm_engine, "model_executor", None)
                if executor is not None:
                    executor.shutdown()
        finally:
            self._llm = None
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            from tuning.utils.gpu import cleanup_gpu
            cleanup_dist_env_and_memory(shutdown_ray=False)
            cleanup_gpu()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vllm_runners.py -v`
Expected: 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/runners.py tests/test_vllm_runners.py
git commit -m "refactor(passk): add Ephemeral and Persistent VLLMRunners"
```

---

## Task 8: Add `DataParallelVLLMRunner`

**Files:**
- Modify: `tuning/training/passk/runners.py`
- Modify: `tests/test_vllm_runners.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vllm_runners.py`:

```python
def test_data_parallel_runner_offloads_and_dispatches(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    captured = {}

    def fake_dp(eval_strategy, adapter_path, config):
        captured["eval"] = eval_strategy
        captured["adapter"] = adapter_path
        captured["num_gpus"] = config.num_inference_gpus
        return [{"prompt": "hi", "responses": ["ok"]}]

    monkeypatch.setattr(runners_mod, "_run_data_parallel", fake_dp)

    cfg = RunnerConfig(
        base_model_hf="m", vllm_gpu_memory_utilization=0.6, lora_max_rank=32,
        chat_template="t", temperature=0.5, max_tokens=256,
        available_gpus=["0", "1"], num_inference_gpus=2,
    )
    model = MagicMock()
    model.parameters.return_value = iter([MagicMock(device="cuda:0")])

    runner = runners_mod.DataParallelVLLMRunner(cfg)
    out = runner.run(model=model, eval_strategy=_make_eval_strategy(),
                     adapter_path="/tmp/a")

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    assert captured["adapter"] == "/tmp/a"
    assert captured["num_gpus"] == 2
    model.cpu.assert_called_once()
    model.to.assert_called_once()
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/test_vllm_runners.py::test_data_parallel_runner_offloads_and_dispatches -v`
Expected: FAIL with missing `DataParallelVLLMRunner`.

- [ ] **Step 3: Implement `DataParallelVLLMRunner` and `_run_data_parallel`**

Append to `tuning/training/passk/runners.py`:

```python
def _run_data_parallel(eval_strategy, adapter_path: str, config: RunnerConfig) -> List[Dict]:
    """Spawn N subprocess workers, partition prompts, merge results.

    Lives at module level so closures over `self` aren't accidentally captured.
    """
    import multiprocessing as mp

    from tuning.training.passk.data_parallel import (
        partition_prompts, _data_parallel_worker,
    )
    from tuning.utils.utils import get_stop_tokens
    import tuning.config as tuning_config

    all_messages = eval_strategy.get_test_messages()
    all_prompts = eval_strategy.get_test_prompts()

    available_gpus = config.available_gpus
    num_gpus = config.num_inference_gpus
    if len(available_gpus) < num_gpus:
        print(f"[VLLMRunner] WARNING: requested {num_gpus} inference GPUs but only "
              f"{len(available_gpus)} available ({available_gpus}). "
              f"Using {len(available_gpus)}.")
        num_gpus = len(available_gpus)

    message_chunks = partition_prompts(all_messages, num_gpus)
    prompt_chunks = partition_prompts(all_prompts, num_gpus)
    actual_num_workers = len(message_chunks)

    print(f"[VLLMRunner] Data-parallel: {len(all_messages)} prompts across "
          f"{actual_num_workers} GPUs")
    for i, chunk in enumerate(message_chunks):
        print(f"[VLLMRunner]   Worker {i} → CUDA device {available_gpus[i]}: "
              f"{len(chunk)} prompts")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    stop_tokens = get_stop_tokens()
    eval_seed = tuning_config.get_eval_seed()

    processes = []
    for i in range(actual_num_workers):
        p = ctx.Process(
            target=_data_parallel_worker,
            args=(
                i, available_gpus[i], message_chunks[i], config.base_model_hf,
                adapter_path, eval_strategy.n_samples, config.temperature,
                config.max_tokens, config.chat_template, config.lora_max_rank,
                config.vllm_gpu_memory_utilization, result_queue,
                stop_tokens, eval_seed,
            ),
        )
        p.start()
        processes.append(p)

    results_by_worker = {}
    for _ in range(actual_num_workers):
        worker_id, serialized, error = result_queue.get()
        if error is not None:
            for p in processes:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(f"[VLLMRunner] Worker {worker_id} failed:\n{error}")
        results_by_worker[worker_id] = serialized

    for p in processes:
        p.join(timeout=30)

    merged = []
    for worker_id in range(actual_num_workers):
        chunk_texts = results_by_worker[worker_id]
        chunk_prompts = prompt_chunks[worker_id]
        for prompt, response_texts in zip(chunk_prompts, chunk_texts):
            merged.append({"prompt": prompt, "responses": response_texts})

    grouped = defaultdict(list)
    for item in merged:
        grouped[item["prompt"]].extend(item["responses"])
    return [{"prompt": p, "responses": resps} for p, resps in grouped.items()]


class DataParallelVLLMRunner(VLLMRunner):
    """Spawns N subprocess vLLM workers; offloads training model to CPU."""

    def run(self, model, eval_strategy, adapter_path):
        if adapter_path is None:
            raise ValueError("DataParallelVLLMRunner requires an adapter_path "
                             "(no External-mode equivalent).")
        with self._with_model_offloaded(model):
            return _run_data_parallel(eval_strategy, adapter_path, self.config)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vllm_runners.py -v`
Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/passk/runners.py tests/test_vllm_runners.py
git commit -m "refactor(passk): add DataParallelVLLMRunner"
```

---

## Task 9: Replace the `_run_eval_with_results` ladder with VLLMRunner dispatch

**Files:**
- Modify: `tuning/training/passk_callback.py`
- Modify: `tests/test_vllm_runners.py` (add fallback test)

This is the integration step. After this, the `if/elif` ladder is gone, the persistent→ephemeral fallback is one code path, and the redundant helpers (`_init_persistent_vllm`, `_create_ephemeral_vllm`, `_cleanup_ephemeral_vllm`, `_cleanup_vllm`, `_run_vllm_inference`, `_run_data_parallel_inference`, `_format_outputs`) get removed.

- [ ] **Step 1: Write the failing fallback test**

Append to `tests/test_vllm_runners.py`:

```python
def test_callback_falls_back_from_persistent_to_ephemeral(monkeypatch):
    """If the persistent runner raises on its first run, the callback should swap
    in an EphemeralVLLMRunner and retry — without re-running inference twice."""
    sys.modules.setdefault("torch", MagicMock())

    from tuning.training import passk_callback as cb_mod
    from tuning.training.passk import runners as runners_mod
    from tuning.training.config_training import PassAtKConfig

    # Stub the runner constructors so we can inspect what the callback selects.
    persistent_run_calls = []
    ephemeral_run_calls = []

    class FakePersistent(runners_mod.PersistentVLLMRunner):
        def run(self, model, eval_strategy, adapter_path):
            persistent_run_calls.append(adapter_path)
            raise RuntimeError("persistent failed")

    class FakeEphemeral(runners_mod.EphemeralVLLMRunner):
        def run(self, model, eval_strategy, adapter_path):
            ephemeral_run_calls.append(adapter_path)
            return [{"prompt": "hi", "responses": ["ok"]}]

    monkeypatch.setattr(cb_mod, "PersistentVLLMRunner", FakePersistent)
    monkeypatch.setattr(cb_mod, "EphemeralVLLMRunner", FakeEphemeral)

    eval_strategy = _make_eval_strategy()
    eval_strategy.score_responses.return_value = {"pass_at_1": 0.5}

    tokenizer = MagicMock()
    tokenizer.chat_template = "t"

    config = PassAtKConfig(
        target_pass_at_k=[],
        use_persistent_vllm=True,
        num_inference_gpus=1,
        enabled=True,
    )

    callback = cb_mod.PassAtKStoppingCallback(
        config=config, tokenizer=tokenizer, model_name="m",
        base_model_hf="m", primary_eval=eval_strategy, monitor_evals=[],
    )
    # Replace the auto-saved-adapter step with a no-op stub.
    monkeypatch.setattr(callback, "_save_adapter_if_needed",
                        lambda model, adapter_dir: "/tmp/a")

    scores, results = callback._run_eval_with_results(MagicMock(), eval_strategy)

    assert scores == {"pass_at_1": 0.5}
    assert len(persistent_run_calls) == 1
    assert len(ephemeral_run_calls) == 1
    # Runner instance has been replaced.
    assert isinstance(callback._runner, FakeEphemeral)
```

- [ ] **Step 2: Run the failing test**

Run: `pytest tests/test_vllm_runners.py::test_callback_falls_back_from_persistent_to_ephemeral -v`
Expected: FAIL — `_runner` doesn't exist yet.

- [ ] **Step 3: Add runner imports and constructor logic to the callback**

In `tuning/training/passk_callback.py`, after the existing imports add:

```python
from tuning.training.passk.runners import (
    RunnerConfig,
    VLLMRunner,
    ExternalVLLMRunner,
    PersistentVLLMRunner,
    EphemeralVLLMRunner,
    DataParallelVLLMRunner,
)
```

Inside `__init__`, replace the existing persistence/inference setup with the block below.

**Keep these two attributes** (`tests/test_multi_gpu_inference.py:63-64,92` assert on them — they accurately describe the resolved post-override mode):
- `self.use_persistent_vllm`
- `self.num_inference_gpus`

**Remove these** (no test or production reader after the refactor):
- `self._vllm_engine`
- `self._external_vllm`
- `self._lora_request_id`
- `self._chat_template` (init-time print at line 202 reads this — replace that print's reference with `tokenizer.chat_template`)
- `self.base_model_hf`, `self.vllm_gpu_memory_utilization`, `self.lora_max_rank` (these become RunnerConfig fields)

Replacement block (place where the old setup was):

```python
        self.use_persistent_vllm = config.use_persistent_vllm
        if self.num_inference_gpus > 1 and self.use_persistent_vllm:
            print(f"[PassAtKCallback] WARNING: num_inference_gpus="
                  f"{self.num_inference_gpus} requires ephemeral mode. "
                  f"Overriding use_persistent_vllm=True → False.")
            self.use_persistent_vllm = False

        self._runner_config = RunnerConfig(
            base_model_hf=base_model_hf,
            vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            lora_max_rank=getattr(config, "lora_max_rank", 32),
            chat_template=tokenizer.chat_template,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            available_gpus=self._available_gpus,
            num_inference_gpus=self.num_inference_gpus,
        )
        self._runner = self._build_runner(config)
```

Add this method on the class:

```python
    def _build_runner(self, config) -> VLLMRunner:
        if config.num_inference_gpus > 1:
            return DataParallelVLLMRunner(self._runner_config)
        if config.use_persistent_vllm:
            return PersistentVLLMRunner(self._runner_config)
        return EphemeralVLLMRunner(self._runner_config)
```

Update `set_trainer_vllm`:

```python
    def set_trainer_vllm(self, llm):
        self._runner = ExternalVLLMRunner(self._runner_config, llm=llm)
```

- [ ] **Step 4: Replace `_run_eval_with_results` body**

Replace the entire current body (lines ~547-601) with:

```python
    def _save_adapter_if_needed(self, model, adapter_dir: str):
        if isinstance(self._runner, ExternalVLLMRunner):
            return None
        self._save_lora_adapter(model, adapter_dir)
        return adapter_dir

    def _run_eval_with_results(self, model, eval_strategy):
        with tempfile.TemporaryDirectory() as adapter_dir:
            adapter_path = self._save_adapter_if_needed(model, adapter_dir)
            try:
                model_results = self._runner.run(model, eval_strategy, adapter_path)
            except Exception as exc:
                if isinstance(self._runner, PersistentVLLMRunner):
                    print(f"[PassAtKCallback] Persistent vLLM failed: {exc}, "
                          f"swapping to ephemeral runner and retrying")
                    self._runner.cleanup()
                    self._runner = EphemeralVLLMRunner(self._runner_config)
                    model_results = self._runner.run(model, eval_strategy, adapter_path)
                else:
                    raise

        print(f"[PassAtKCallback] Scoring responses with "
              f"{eval_strategy.__class__.__name__}...")
        scores = eval_strategy.score_responses(model_results, self.tokenizer)
        return scores, model_results
```

- [ ] **Step 5: Delete the now-redundant helpers**

In `tuning/training/passk_callback.py`, delete these methods and their call sites:
- `_init_persistent_vllm`
- `_run_vllm_inference`
- `_create_ephemeral_vllm`
- `_cleanup_ephemeral_vllm`
- `_cleanup_vllm`  *(but keep one final cleanup call in `on_train_end` — see step 6)*
- `_run_data_parallel_inference`
- `_format_outputs`

Also remove the instance fields these touched (`self._vllm_engine`, `self._external_vllm`, `self._lora_request_id`).

- [ ] **Step 6: Update `on_train_end` to call `self._runner.cleanup()` instead of `_cleanup_vllm`**

Replace the `self._cleanup_vllm()` call at the end of `on_train_end` with:

```python
        self._runner.cleanup()
```

- [ ] **Step 7: Run new and existing tests**

Run: `pytest tests/test_vllm_runners.py tests/test_callback_step_bridging.py tests/test_external_vllm_reuse.py tests/test_passk_callback_wandb_tables.py tests/test_multi_gpu_inference.py tests/test_eval_strategy.py -v`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add tuning/training/passk_callback.py tests/test_vllm_runners.py
git commit -m "refactor(passk): replace vLLM-mode ladder with VLLMRunner dispatch"
```

---

## Task 10: Extract logging helpers into `passk/logging.py`

**Files:**
- Create: `tuning/training/passk/logging.py`
- Modify: `tuning/training/passk_callback.py`

The current `_log_raw_generation_table` (passk_callback.py:476-545) plus the W&B-log-and-print pattern repeated for primary/monitor evals (passk_callback.py:626-660) consolidate into one module. `print` stays as the mechanism — only the prefix is in one place.

- [ ] **Step 1: Write the new logging module**

Create `tuning/training/passk/logging.py`:

```python
# ABOUTME: W&B and console logging helpers for the Pass@K callback.
# ABOUTME: One place that knows the [PassAtKCallback] prefix and the train/global_step keys.

import datetime
import json
from typing import Dict, List

import wandb


_LOG_PREFIX = "[PassAtKCallback]"


def log_eval_metrics(
    *,
    eval_strategy,
    scores: Dict[str, float],
    raw_results: List[Dict],
    global_step: int,
    step_offset: int,
    thresholds_remaining: List[float],
    is_primary: bool,
) -> None:
    """Single entry point for wandb metrics + raw-generations table + console summary."""
    log_dict = {
        "train/global_step": global_step,
        "train/total_global_step": global_step + step_offset,
    }
    log_dict.update(eval_strategy.wandb_metrics(scores))
    wandb.log(log_dict)

    stopping_key = eval_strategy.stopping_metric()
    stopping_value = scores.get(stopping_key)
    _log_raw_generation_table(
        eval_strategy=eval_strategy,
        model_results=raw_results,
        global_step=global_step,
        step_offset=step_offset,
        stopping_metric_name=stopping_key,
        stopping_metric_value=stopping_value,
        thresholds_remaining=thresholds_remaining,
    )

    label = "Primary" if is_primary else f"Monitor ({eval_strategy.__class__.__name__})"
    score_summary = ", ".join(
        f"{k}={v:.4f}" for k, v in scores.items() if isinstance(v, float)
    )
    if is_primary:
        print(f"\n{_LOG_PREFIX} Step {global_step}: {score_summary} "
              f"({scores.get('num_prompts_evaluated', '?')} prompts)")
    else:
        print(f"{_LOG_PREFIX} {label}: {score_summary}")


def _log_raw_generation_table(
    *,
    eval_strategy,
    model_results: List[Dict],
    global_step: int,
    step_offset: int,
    stopping_metric_name: str,
    stopping_metric_value,
    thresholds_remaining,
) -> None:
    """Best-effort: log raw generations as a per-step W&B Table."""
    eval_slug = eval_strategy.id
    table_key = f"raw_generations/{eval_slug}/step_{global_step}"
    try:
        table = wandb.Table(columns=[
            "global_step", "eval_name", "prompt_index", "prompt", "responses",
            "num_responses", "per_response_correct", "per_response_instructions",
            "prompt_accuracy", "stopping_metric_name", "stopping_metric_value",
            "thresholds_remaining", "timestamp_utc",
        ])
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        thresholds_remaining_json = json.dumps(thresholds_remaining)

        for prompt_index, item in enumerate(model_results):
            prompt = item.get("prompt", "")
            responses = item.get("responses", [])
            if not isinstance(responses, list):
                responses = [responses]
            try:
                responses_json = json.dumps(responses)
            except TypeError:
                responses_json = json.dumps([str(r) for r in responses])

            correctness = item.get("per_response_correct", [])
            prompt_accuracy = (sum(correctness) / len(correctness)
                               if correctness else None)
            instructions = item.get("per_response_instructions", [])

            table.add_data(
                global_step, eval_slug, prompt_index, prompt, responses_json,
                len(responses),
                json.dumps(correctness) if correctness else None,
                json.dumps(instructions) if instructions else None,
                prompt_accuracy, stopping_metric_name, stopping_metric_value,
                thresholds_remaining_json, timestamp_utc,
            )
        wandb.log({
            "train/global_step": global_step,
            "train/total_global_step": global_step + step_offset,
            table_key: table,
        })
    except Exception as exc:
        print(f"{_LOG_PREFIX} Warning: failed to log raw generation table "
              f"({table_key}): {exc}")
```

- [ ] **Step 2: Replace inline logging in the callback**

In `tuning/training/passk_callback.py`:

1. Add the import:
   ```python
   from tuning.training.passk.logging import log_eval_metrics
   ```
2. Delete the existing `_log_raw_generation_table` method (lines 476-545).
3. In `on_evaluate`, replace the primary-eval logging block (the `wandb.log` call, the `_log_raw_generation_table` call, and the `print(... Step ... Data Points ...)` call) with:

   ```python
        log_eval_metrics(
            eval_strategy=self.primary_eval,
            scores=scores,
            raw_results=raw_results,
            global_step=state.global_step,
            step_offset=self._step_offset,
            thresholds_remaining=self._decision_engine.target_thresholds,
            is_primary=True,
        )
   ```

4. Replace the monitor-eval inner block with:

   ```python
            log_eval_metrics(
                eval_strategy=monitor_eval,
                scores=monitor_scores,
                raw_results=monitor_raw_results,
                global_step=state.global_step,
                step_offset=self._step_offset,
                thresholds_remaining=self._decision_engine.target_thresholds,
                is_primary=False,
            )
   ```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_passk_callback_wandb_tables.py tests/test_callback_step_bridging.py tests/test_external_vllm_reuse.py -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tuning/training/passk/logging.py tuning/training/passk_callback.py
git commit -m "refactor(passk): extract eval-metric logging into passk.logging"
```

---

## Task 11: Slim `on_evaluate` and rename `prevResults`

**Files:**
- Modify: `tuning/training/passk_callback.py`

Final cleanup of `on_evaluate`. Add the `_eval_and_log` and `_compute_data_points_seen` helpers, rename `prevResults` → `_primary_metric_history`, and remove the read-only mirrors (`self.target_pass_at_k_thresholds`, `self.early_tuples`) added during Task 4.

- [ ] **Step 1: Add helper methods, rename history field**

In `tuning/training/passk_callback.py`:

1. In `__init__`, change `self.prevResults = []` to `self._primary_metric_history = []`.
2. Remove the read-only mirrors `self.target_pass_at_k_thresholds = ...` and `self.early_tuples = ...` added in Task 4. Update the init-time print statements (around line 191) to read from the engine:
   ```python
        if not self._decision_engine.early_tuples:
            print(f"[PassAtKCallback] Initialized with {primary_eval.label_prefix} "
                  f"thresholds={self._decision_engine.target_thresholds}")
            print(f"[PassAtKCallback] Training will stop when hardest threshold is "
                  f"reached: {self._decision_engine.target_thresholds[0]}")
        else:
            print(f"[PassAtKCallback] Initialized with "
                  f"early_tuples={self._decision_engine.early_tuples}")
            print(f"[PassAtKCallback] Training will stop when all early_tuples have "
                  f"triggered")
   ```
3. Add helper methods:

```python
    def _compute_data_points_seen(self, args, state) -> int:
        train_batch_size = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        world_size = getattr(args, "world_size", 1)
        return state.global_step * train_batch_size * grad_accum * world_size

    def _eval_and_log(self, model, eval_strategy, state, *, is_primary: bool):
        scores, raw_results = self._run_eval_with_results(model, eval_strategy)
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

- [ ] **Step 2: Replace `on_evaluate` body**

Replace the current `on_evaluate` method (everything from `def on_evaluate(...)` through the trailing `return control`) with:

```python
    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        """Called after evaluation, run evals and stop if target reached."""
        if model is None:
            model = kwargs.get("model")
        if model is None:
            print("[PassAtKCallback] Warning: model is None, skipping eval")
            return control

        data_points_seen = self._compute_data_points_seen(args, state)

        primary_scores = self._eval_and_log(model, self.primary_eval, state,
                                            is_primary=True)
        primary_metric = primary_scores[self.primary_eval.stopping_metric()]
        self._primary_metric_history.append(primary_metric)

        for monitor_eval in self.monitor_evals:
            self._eval_and_log(model, monitor_eval, state, is_primary=False)

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
            print(f"[PassAtKCallback] Saved checkpoint: {decision.label}")

        self._last_eval_step = state.global_step
        return control
```

- [ ] **Step 3: Run all callback tests**

Run: `pytest tests/test_callback_step_bridging.py tests/test_passk_callback_wandb_tables.py tests/test_external_vllm_reuse.py tests/test_eval_strategy.py tests/test_multi_gpu_inference.py tests/test_seed_wiring.py tests/test_checkpoint_decision_engine.py tests/test_vllm_runners.py -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tuning/training/passk_callback.py
git commit -m "refactor(passk): slim on_evaluate, rename prevResults to _primary_metric_history"
```

---

## Task 12: Move callback class to `passk/callback.py`; reduce `passk_callback.py` to a shim

**Files:**
- Create: `tuning/training/passk/callback.py` (target of the move)
- Modify: `tuning/training/passk/__init__.py` (re-export)
- Modify: `tuning/training/passk_callback.py` (one-line shim)

After Tasks 2-11, `passk_callback.py` already imports from `passk.data_parallel`, `passk.decisions`, `passk.runners`, and `passk.logging`. It now contains just the `PassAtKStoppingCallback` class plus its imports. This task moves that file to `passk/callback.py` verbatim and turns `passk_callback.py` into a re-export shim.

- [ ] **Step 1: Move file content**

Run: `git mv tuning/training/passk_callback.py tuning/training/passk/callback.py`

- [ ] **Step 2: Update the ABOUTME header**

Edit `tuning/training/passk/callback.py` so the first two lines are:

```python
# ABOUTME: PassAtKStoppingCallback — slim TrainerCallback that orchestrates eval and
# ABOUTME: checkpoint saving via VLLMRunner and CheckpointDecisionEngine.
```

- [ ] **Step 3: Update intra-package imports inside `callback.py`**

In `tuning/training/passk/callback.py`, change imports of sibling passk modules from absolute to relative for clarity:

```python
from .data_parallel import partition_prompts, _data_parallel_worker
from .decisions import CheckpointDecisionEngine
from .logging import log_eval_metrics
from .runners import (
    RunnerConfig, VLLMRunner,
    ExternalVLLMRunner, PersistentVLLMRunner,
    EphemeralVLLMRunner, DataParallelVLLMRunner,
)
```

(External-package imports — `tuning.config`, `tuning.training.callback_utils`, `tuning.training.eval_strategy` — stay absolute.)

- [ ] **Step 4: Re-export from the package init**

Replace `tuning/training/passk/__init__.py` with:

```python
# ABOUTME: Pass@K callback subpackage — split from the previous monolithic passk_callback.py.
# ABOUTME: Public re-export below preserves the historical import path.

from .callback import PassAtKStoppingCallback

__all__ = ["PassAtKStoppingCallback"]
```

- [ ] **Step 5: Create the shim file at the old path**

Create `tuning/training/passk_callback.py` with this exact content:

```python
# ABOUTME: Backwards-compatible re-export. Production code imports
# ABOUTME: PassAtKStoppingCallback from this path; the implementation lives in passk/.

from tuning.training.passk.callback import PassAtKStoppingCallback

__all__ = ["PassAtKStoppingCallback"]
```

- [ ] **Step 6: Run the full pass@k-related test suite**

Run: `pytest tests/test_callback_step_bridging.py tests/test_passk_callback_wandb_tables.py tests/test_external_vllm_reuse.py tests/test_eval_strategy.py tests/test_multi_gpu_inference.py tests/test_seed_wiring.py tests/test_checkpoint_decision_engine.py tests/test_vllm_runners.py tests/test_passk_early_data_chat_templating.py -v`
Expected: all tests pass.

- [ ] **Step 7: Verify production import paths still resolve**

Run: `python -c "from tuning.training.passk_callback import PassAtKStoppingCallback; from tuning.training.passk import PassAtKStoppingCallback as P2; assert PassAtKStoppingCallback is P2"`
Expected: exits 0 with no output.

- [ ] **Step 8: Commit**

```bash
git add tuning/training/passk_callback.py tuning/training/passk/callback.py \
        tuning/training/passk/__init__.py
git commit -m "refactor(passk): move callback class to passk.callback, leave shim at old path"
```

---

## Task 13: Final regression check + smoke pipeline run

**Files:**
- None (run-only)

- [ ] **Step 1: Run the full project test suite**

Run: `pytest tests/ -v --ignore=tests/test_unified_early_pipeline.py --ignore=tests/test_simplerl_rlvr.py --ignore=tests/test_grpo_config.py 2>&1 | tail -80`
(The three excluded tests still have unified_early_pipeline import paths that Plan 2 will fix; they're out of scope here. They should still pass as-is — exclusion is only to keep this run focused.)

Then also run the excluded tests to confirm they remain green (they don't import from `passk_callback`, so they shouldn't be affected):
Run: `pytest tests/test_unified_early_pipeline.py tests/test_simplerl_rlvr.py tests/test_grpo_config.py -v 2>&1 | tail -40`
Expected: all tests pass.

- [ ] **Step 2: Run a smoke pipeline (Slurm fire-and-forget)**

Per project memory (`feedback_fire_and_forget_sbatch.md`), prefer fire-and-forget sbatch:

Run: `sbatch tuning/slurm/unified_early_pipeline_short.sh --model llama3-1B --wandb-project tuning-refactor-smoke --dataset gsm8k --train-size 100 --sft-num-epochs 1 --sft-batch-size 4 --sft-eval-steps 8 --sft-passk-num-prompts 16 --sft-passk-targets 0.0`
Expected: `Submitted batch job <id>` printed. Note the job ID; check W&B project `tuning-refactor-smoke` for the run.

- [ ] **Step 3: Final commit if any tail-end fixes were needed**

If steps 1 or 2 surfaced anything, fix and commit. Otherwise nothing to commit — the smoke run validates that import-order and runtime paths are healthy.

If a final wrap-up commit is needed:

```bash
git add -- <only-the-fixed-files>
git commit -m "refactor(passk): post-smoke fixups"
```

---

## Self-review checklist

(Maintainer-internal — for the engineer following this plan.)

- [ ] Every task ends with tests run and a commit.
- [ ] No task references a class, helper, or method that wasn't defined in an earlier task.
- [ ] Test imports updated whenever a module moves (Task 2: `test_multi_gpu_inference.py`, `test_seed_wiring.py`).
- [ ] The shim in `passk_callback.py` (Task 12 step 5) preserves the historical import path that production code depends on.
- [ ] `_save_sweetspot_checkpoint` accepts a string label (verified via `callback_utils.save_sweetspot_checkpoint`).
- [ ] `_format_outputs` removed from callback (Task 9 step 5) — the equivalent now lives on `VLLMRunner._format_outputs`.
