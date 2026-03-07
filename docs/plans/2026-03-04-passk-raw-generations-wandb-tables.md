# Pass@K Raw Generation W&B Tables Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Log full evaluation raw generations to W&B as per-eval-step tables for primary and monitor eval strategies during training.

**Architecture:** Extend the generation-eval callback to emit a fresh W&B table per eval step from raw prompt/response outputs, while preserving existing scalar metric logging and stopping logic. Keep table logging best-effort so failures in observability do not interrupt training. Add focused unit tests for key naming, row structure, monitor coverage, and graceful failure behavior.

**Tech Stack:** Python, `transformers` Trainer callback, Weights & Biases (`wandb`), `pytest`, `unittest.mock`.

---

### Task 1: Add callback tests for per-step raw table logging

**Files:**
- Create: `tests/test_passk_callback_wandb_tables.py`
- Modify: `tuning/training/passk_callback.py`

**Step 1: Write the failing test**

Add tests that instantiate `PassAtKStoppingCallback` with lightweight stubs and verify:
- Primary eval emits a table log key `eval/raw_generations/<eval_name>/step_<global_step>`
- Monitor eval also emits its own table key
- Table rows include prompt + JSON-serialized responses

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_passk_callback_wandb_tables.py -v`
Expected: FAIL because table logging behavior is not implemented.

**Step 3: Write minimal implementation**

Implement callback table logging logic in `passk_callback.py`:
- Capture raw generation outputs from eval execution
- Build per-step `wandb.Table`
- Log one table per eval strategy per step

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_passk_callback_wandb_tables.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_passk_callback_wandb_tables.py tuning/training/passk_callback.py
git commit -m "feat: log per-step raw eval generations to wandb tables"
```

### Task 2: Add resiliency behavior tests and complete error handling

**Files:**
- Modify: `tests/test_passk_callback_wandb_tables.py`
- Modify: `tuning/training/passk_callback.py`

**Step 1: Write the failing test**

Add tests that verify:
- Table logging exceptions are caught and do not stop callback flow
- Empty raw results skip table logging without raising

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_passk_callback_wandb_tables.py -k \"exception or empty\" -v`
Expected: FAIL before error handling is complete.

**Step 3: Write minimal implementation**

Add guarded table logging helper with:
- try/except around table build + `wandb.log`
- serialization fallback for non-JSON-safe responses
- warning prints for empty results or exceptions

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_passk_callback_wandb_tables.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_passk_callback_wandb_tables.py tuning/training/passk_callback.py
git commit -m "fix: make raw eval table logging best-effort"
```

### Task 3: Run regression checks for nearby behavior

**Files:**
- Verify: `tests/test_eval_strategy.py`
- Verify: `tests/test_multi_gpu_inference.py`

**Step 1: Run targeted regressions**

Run:
- `pytest tests/test_eval_strategy.py -v`
- `pytest tests/test_multi_gpu_inference.py -v`

Expected: PASS; confirms callback/eval interfaces remain compatible.

**Step 2: Run new tests together with regressions**

Run:
- `pytest tests/test_passk_callback_wandb_tables.py tests/test_eval_strategy.py tests/test_multi_gpu_inference.py -v`

Expected: PASS.

**Step 3: Commit final implementation**

```bash
git add tuning/training/passk_callback.py tests/test_passk_callback_wandb_tables.py
git commit -m "test: cover wandb raw generation table logging in passk callback"
```
