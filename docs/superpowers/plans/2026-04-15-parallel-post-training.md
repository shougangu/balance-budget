# Parallel Post-Training Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `--parallel N` fan out post-training checkpoints across N sbatch GPU allocations, so N checkpoints can train concurrently instead of sequentially.

**Architecture:** Add a `claimed` field to checkpoint rows in the JSONL metadata. A new `claim_next_checkpoint()` (via `_update_row` helper, no locking) picks the next unclaimed+uncompleted row and marks it claimed. Workers exit with code 42 when nothing is available to claim. The orchestrator submits `N-1` sbatch worker jobs (fire and forget) that replay the same sbatch command with `--run-dpo/grpo --run-all --metadata-file` but with `--parallel` stripped so workers don't recursively dispatch. The orchestrator runs the post-training loop itself as the Nth worker.

**Tech Stack:** Python stdlib (`subprocess`, `json`, `argparse`, `sys`), pytest, Slurm (`sbatch`). No new dependencies.

---

## File Structure

| File | Role |
|------|------|
| `tuning/training/unified_early_pipeline.py` | New `_update_row()`, `claim_next_checkpoint()`, `_submit_sbatch_worker()`, `_dispatch_parallel_workers()`; refactored `mark_completed()`; `--parallel` CLI arg; updates to `run_dpo`/`run_grpo` and the orchestrator loop |
| `tests/test_unified_early_pipeline.py` | Unit tests for `claim_next_checkpoint`, `--parallel` arg, exit-42 behavior, `_submit_sbatch_worker`, `_dispatch_parallel_workers` |

Spec: `docs/superpowers/specs/2026-04-15-parallel-post-training-design.md`

---

### Task 1: Add `claim_next_checkpoint()` function

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (add function near `next_checkpoint` at line 454)
- Test: `tests/test_unified_early_pipeline.py` (add class near `TestMetadataWorkQueue` at line 231)

**Goal:** Extract a shared `_update_row` helper for the read-modify-write pattern, use it to implement `claim_next_checkpoint`, and refactor `mark_completed` to use it too.

- [ ] **Step 1.1: Write the failing tests**

Add to `tests/test_unified_early_pipeline.py` right after the existing `TestMetadataWorkQueue` class. Also add `claim_next_checkpoint` to the imports at line 10-19:

```python
from tuning.training.unified_early_pipeline import (
    parse_early_tuple,
    load_checkpoints,
    _parse_args,
    next_checkpoint,
    claim_next_checkpoint,   # NEW
    mark_completed,
    print_metadata_paths,
    parse_metadata_from_output,
    _build_base_cmd,
)
```

Then add the test class:

```python
class TestClaimNextCheckpoint:
    def test_claims_first_available_row(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        result = claim_next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp1"

    def test_marks_row_as_claimed(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        claim_next_checkpoint(str(f))
        with open(f) as fh:
            lines = [json.loads(l) for l in fh]
        assert lines[0]["claimed"] is True
        assert lines[1].get("claimed", False) is False

    def test_skips_already_claimed_rows(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        claimed_row = {**PASSK_ROW, "claimed": True}
        _write_jsonl(f, [claimed_row, PPL_ROW])
        result = claim_next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp2"

    def test_skips_completed_rows(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        completed_row = {**PASSK_ROW, "completed": True}
        _write_jsonl(f, [completed_row, PPL_ROW])
        result = claim_next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp2"

    def test_returns_none_when_nothing_available(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [
            {**PASSK_ROW, "completed": True},
            {**PPL_ROW, "claimed": True},
        ])
        assert claim_next_checkpoint(str(f)) is None

    def test_preserves_other_fields(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW])
        claim_next_checkpoint(str(f))
        with open(f) as fh:
            row = json.loads(fh.readline())
        assert row["data_points_seen"] == 512
        assert row["threshold_type"] == "pass_at_1"
        assert row["claimed"] is True

    def test_sequential_claims_pick_different_rows(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW, PASSK_ROW_2])
        first = claim_next_checkpoint(str(f))
        second = claim_next_checkpoint(str(f))
        third = claim_next_checkpoint(str(f))
        fourth = claim_next_checkpoint(str(f))
        assert first["checkpoint_path"] == "/models/cp1"
        assert second["checkpoint_path"] == "/models/cp2"
        assert third["checkpoint_path"] == "/models/cp3"
        assert fourth is None
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd /project/6105902/shougan/balance-budget
python -m pytest tests/test_unified_early_pipeline.py::TestClaimNextCheckpoint -v
```

Expected: ImportError (`claim_next_checkpoint` is not defined).

- [ ] **Step 1.3: Add `_update_row` helper, implement `claim_next_checkpoint`, refactor `mark_completed`**

Add `_update_row` and `claim_next_checkpoint` to `tuning/training/unified_early_pipeline.py`, immediately after `next_checkpoint` (currently ends at line 462):

```python
def _update_row(metadata_file, predicate, updates):
    """Find first row matching predicate, apply updates dict, rewrite file.

    Returns the updated row, or None if no row matched.
    """
    with open(metadata_file) as f:
        lines = f.readlines()
    target = None
    with open(metadata_file, "w") as f:
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if target is None and predicate(row):
                row.update(updates)
                target = row
            f.write(json.dumps(row) + "\n")
    return target


def claim_next_checkpoint(metadata_file):
    """Pick the next unclaimed+uncompleted checkpoint and mark it claimed.

    No file locking: race window is near-simultaneous sbatch starts, worst case
    is one checkpoint trained twice. Fine for our use case.
    """
    row = _update_row(
        metadata_file,
        lambda r: not r.get("claimed") and not r.get("completed"),
        {"claimed": True},
    )
    if row:
        print(f"Claimed checkpoint: {row['checkpoint_path']} (threshold {row.get('threshold_value')}, type {row.get('threshold_type')})")
    return row
```

Then refactor `mark_completed` (currently at line 465) to use the helper:

```python
def mark_completed(metadata_file, checkpoint_path):
    """Mark a checkpoint as completed in the metadata file."""
    _update_row(
        metadata_file,
        lambda r: r["checkpoint_path"] == checkpoint_path,
        {"completed": True},
    )
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestClaimNextCheckpoint tests/test_unified_early_pipeline.py::TestMetadataWorkQueue -v
```

Expected: all tests pass (7 new + existing `mark_completed` tests verify the refactor).

- [ ] **Step 1.5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Add _update_row helper, claim_next_checkpoint, and refactor mark_completed"
```

---

### Task 2: Add `--parallel` CLI arg

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (inside `_parse_args`, near the stage group at line 83-94)
- Test: `tests/test_unified_early_pipeline.py` (add to `TestParseArgs` class)

- [ ] **Step 2.1: Write the failing tests**

Find `TestParseArgs` class in `tests/test_unified_early_pipeline.py` (around line 142) and add:

```python
class TestParallelArg:
    def test_default_parallel_is_1(self):
        args = _parse_args(REQUIRED)
        assert args.parallel == 1

    def test_parallel_accepts_integer(self):
        args = _parse_args(REQUIRED + ["--parallel", "3"])
        assert args.parallel == 3

```

Add this as a new class right after the existing `TestParseArgs` or wherever ordering makes sense.

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestParallelArg -v
```

Expected: failures (`parallel` attr doesn't exist).

- [ ] **Step 2.3: Add the arg**

In the stage group inside `_parse_args` (after line 94):

```python
stage.add_argument("--parallel", type=int, default=1,
                   help="Number of concurrent post-training workers. "
                        "When >1, the orchestrator submits --parallel-1 sbatch jobs "
                        "and runs as the Nth worker itself.")
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestParallelArg -v
```

Expected: 2 passed.

- [ ] **Step 2.5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Add --parallel CLI arg for parallel post-training workers"
```

---

### Task 3: Worker mode exits 42 when no checkpoint to claim

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:494-515` (`run_dpo`)
- Modify: `tuning/training/unified_early_pipeline.py:623-644` (`run_grpo`)
- Test: `tests/test_unified_early_pipeline.py` (new `TestWorkerExitCode` class)

**Goal:** Swap `next_checkpoint()` for `claim_next_checkpoint()` inside `run_dpo` and `run_grpo`. When nothing to claim, `sys.exit(42)`.

- [ ] **Step 3.1: Write the failing tests**

Add to `tests/test_unified_early_pipeline.py`:

```python
class TestWorkerExitCode:
    def test_run_dpo_exits_42_when_nothing_to_claim(self, tmp_path):
        from tuning.training import unified_early_pipeline as uep
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "completed": True}])
        args = argparse.Namespace(metadata_file=[str(f)])
        with pytest.raises(SystemExit) as exc:
            uep.run_dpo(args)
        assert exc.value.code == 42

    def test_run_grpo_exits_42_when_nothing_to_claim(self, tmp_path):
        from tuning.training import unified_early_pipeline as uep
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "claimed": True}])
        args = argparse.Namespace(metadata_file=[str(f)])
        with pytest.raises(SystemExit) as exc:
            uep.run_grpo(args)
        assert exc.value.code == 42
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestWorkerExitCode -v
```

Expected: failures (current `run_dpo` uses `next_checkpoint` and returns without exit code).

- [ ] **Step 3.3: Update `run_dpo`**

In `tuning/training/unified_early_pipeline.py`, replace lines 500-504 (inside `run_dpo`):

Old:
```python
    metadata_file = args.metadata_file[0]
    checkpoint = next_checkpoint(metadata_file)
    if checkpoint is None:
        print("All checkpoints completed, nothing to do.")
        return
```

New:
```python
    metadata_file = args.metadata_file[0]
    checkpoint = claim_next_checkpoint(metadata_file)
    if checkpoint is None:
        print("No checkpoints available to claim, nothing to do.")
        sys.exit(42)
```

- [ ] **Step 3.4: Update `run_grpo`**

In `tuning/training/unified_early_pipeline.py`, replace lines 630-634 (inside `run_grpo`):

Old:
```python
    metadata_file = args.metadata_file[0]
    checkpoint = next_checkpoint(metadata_file)
    if checkpoint is None:
        print("All checkpoints completed, nothing to do.")
        return
```

New:
```python
    metadata_file = args.metadata_file[0]
    checkpoint = claim_next_checkpoint(metadata_file)
    if checkpoint is None:
        print("No checkpoints available to claim, nothing to do.")
        sys.exit(42)
```

- [ ] **Step 3.5: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestWorkerExitCode -v
```

Expected: 2 passed.

- [ ] **Step 3.6: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Worker modes claim checkpoints and exit 42 when none available"
```

---

### Task 4: Post-training loop handles exit 42

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:797-804` (just the inner `while` loop)

**Goal:** The loop currently polls `next_checkpoint()` in the orchestrator process. Now each subprocess claims its own checkpoint. The loop keeps spawning subprocesses until one returns 42.

- [ ] **Step 4.1: Replace the inner `while` loop**

In `tuning/training/unified_early_pipeline.py`, replace lines 797-804 (the inner `while` loop inside the post-training block):

Old:
```python
        while next_checkpoint(metadata_file) is not None:
            pt_cmd = [sys.executable] + base_cmd + [
                pt_flag, "--metadata-file", metadata_file,
            ]
            print(f"[orchestrator] Running {pt_method.upper()}: {' '.join(pt_cmd)}")
            result = subprocess.run(pt_cmd)
            if result.returncode != 0:
                sys.exit(f"{pt_method.upper()} subprocess failed with return code {result.returncode}")
```

New:
```python
        while True:
            pt_cmd = [sys.executable] + base_cmd + [
                pt_flag, "--metadata-file", metadata_file,
            ]
            print(f"[orchestrator] Running {pt_method.upper()}: {' '.join(pt_cmd)}")
            result = subprocess.run(pt_cmd)
            if result.returncode == 42:
                print(f"[orchestrator] No more checkpoints in {metadata_file}, moving on")
                break
            if result.returncode != 0:
                sys.exit(f"{pt_method.upper()} subprocess failed with return code {result.returncode}")
```

- [ ] **Step 4.2: Smoke-test the full test suite compiles and runs**

```bash
python -m pytest tests/test_unified_early_pipeline.py -v
```

Expected: all existing tests pass.

- [ ] **Step 4.3: Commit**

```bash
git add tuning/training/unified_early_pipeline.py
git commit -m "Post-training loop iterates until worker exits 42"
```

---

### Task 5: Orchestrator dispatches N-1 sbatch workers when `--parallel > 1`

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (new functions near `_build_base_cmd` at line 746; dispatch call in `main()`)
- Test: `tests/test_unified_early_pipeline.py` (new `TestSubmitSbatchWorker` and `TestDispatchParallelWorkers` classes)

**Goal:** When `--parallel > 1`, submit `parallel - 1` sbatch worker jobs before entering the post-training loop. Workers replay the same sbatch command with `--run-dpo/grpo --run-all --metadata-file <each>`, entering orchestrator mode but skipping SFT. Workers don't recursively dispatch because `--parallel` is stripped from their args, so they default to `parallel=1` and the dispatch is a no-op. `claim_next_checkpoint` ensures no two workers process the same checkpoint.

- [ ] **Step 5.1: Add a module-level constant for the sbatch script path**

At the top of `tuning/training/unified_early_pipeline.py`, just after the imports (around line 10):

```python
SBATCH_WORKER_SCRIPT = "tuning/slurm/unified_early_pipeline.sh"
```

- [ ] **Step 5.2: Write the failing tests**

Add `_submit_sbatch_worker` and `_dispatch_parallel_workers` to the imports in `tests/test_unified_early_pipeline.py` (lines 10-19).

Add to `tests/test_unified_early_pipeline.py`:

```python
class TestSubmitSbatchWorker:
    def test_parses_job_id_from_stdout(self, monkeypatch):
        from tuning.training import unified_early_pipeline as uep
        fake = type("R", (), {"stdout": "Submitted batch job 12345\n", "stderr": "", "returncode": 0})()
        calls = []

        def fake_run(cmd, capture_output, text):
            calls.append(cmd)
            return fake

        monkeypatch.setattr(uep.subprocess, "run", fake_run)
        job_id = uep._submit_sbatch_worker(
            "tuning/slurm/unified_early_pipeline.sh",
            ["--model", "llama3-3B"],
        )
        assert job_id == "12345"
        assert calls[0][0] == "sbatch"
        assert calls[0][1] == "tuning/slurm/unified_early_pipeline.sh"
        assert "--model" in calls[0]
        assert "llama3-3B" in calls[0]

    def test_nonzero_return_code_exits(self, monkeypatch):
        from tuning.training import unified_early_pipeline as uep
        fake = type("R", (), {"stdout": "", "stderr": "sbatch: error: boom\n", "returncode": 1})()

        monkeypatch.setattr(uep.subprocess, "run", lambda *a, **k: fake)
        with pytest.raises(SystemExit):
            uep._submit_sbatch_worker("script.sh", [])

    def test_unparseable_stdout_exits(self, monkeypatch):
        from tuning.training import unified_early_pipeline as uep
        fake = type("R", (), {"stdout": "weird output\n", "stderr": "", "returncode": 0})()

        monkeypatch.setattr(uep.subprocess, "run", lambda *a, **k: fake)
        with pytest.raises(SystemExit):
            uep._submit_sbatch_worker("script.sh", [])


class TestDispatchParallelWorkers:
    def test_parallel_1_does_nothing(self, monkeypatch):
        from tuning.training import unified_early_pipeline as uep
        calls = []
        monkeypatch.setattr(uep, "_submit_sbatch_worker",
                            lambda *a, **k: (calls.append(a), "999")[1])
        uep._dispatch_parallel_workers(
            parallel=1,
            base_cmd=["pipeline.py", "--model", "llama3-3B"],
            pt_flag="--run-dpo",
            metadata_files=["/tmp/a.jsonl"],
        )
        assert calls == []

    def test_parallel_3_submits_2_workers(self, monkeypatch, tmp_path):
        from tuning.training import unified_early_pipeline as uep
        calls = []
        monkeypatch.setattr(uep, "_submit_sbatch_worker",
                            lambda *a, **k: (calls.append(a), "999")[1])
        mf = tmp_path / "meta.jsonl"
        mf.write_text("{}\n")
        uep._dispatch_parallel_workers(
            parallel=3,
            base_cmd=["pipeline.py", "--model", "llama3-3B"],
            pt_flag="--run-dpo",
            metadata_files=[str(mf)],
        )
        assert len(calls) == 2

    def test_worker_argv_includes_pt_flag_and_metadata(self, monkeypatch, tmp_path):
        from tuning.training import unified_early_pipeline as uep
        calls = []
        monkeypatch.setattr(uep, "_submit_sbatch_worker",
                            lambda script, argv: (calls.append(argv), "999")[1])
        mf = tmp_path / "meta.jsonl"
        mf.write_text("{}\n")
        uep._dispatch_parallel_workers(
            parallel=2,
            base_cmd=["pipeline.py", "--model", "llama3-3B"],
            pt_flag="--run-grpo",
            metadata_files=[str(mf)],
        )
        worker_argv = calls[0]
        assert "--run-grpo" in worker_argv
        assert "--run-all" in worker_argv
        assert "--metadata-file" in worker_argv
        assert str(mf) in worker_argv
        assert "--model" in worker_argv
        assert "llama3-3B" in worker_argv

    def test_strips_parallel_from_worker_argv(self, monkeypatch, tmp_path):
        from tuning.training import unified_early_pipeline as uep
        calls = []
        monkeypatch.setattr(uep, "_submit_sbatch_worker",
                            lambda script, argv: (calls.append(argv), "999")[1])
        mf = tmp_path / "meta.jsonl"
        mf.write_text("{}\n")
        uep._dispatch_parallel_workers(
            parallel=2,
            base_cmd=["pipeline.py", "--model", "llama3-3B", "--parallel", "3"],
            pt_flag="--run-dpo",
            metadata_files=[str(mf)],
        )
        worker_argv = calls[0]
        assert "--parallel" not in worker_argv
        assert "3" not in worker_argv
        assert "--model" in worker_argv

    def test_skips_missing_metadata_files(self, monkeypatch, tmp_path):
        from tuning.training import unified_early_pipeline as uep
        calls = []
        monkeypatch.setattr(uep, "_submit_sbatch_worker",
                            lambda script, argv: (calls.append(argv), "999")[1])
        real_mf = tmp_path / "real.jsonl"
        real_mf.write_text("{}\n")
        missing_mf = str(tmp_path / "missing.jsonl")
        uep._dispatch_parallel_workers(
            parallel=2,
            base_cmd=["pipeline.py"],
            pt_flag="--run-dpo",
            metadata_files=[str(real_mf), missing_mf],
        )
        worker_argv = calls[0]
        assert str(real_mf) in worker_argv
        assert missing_mf not in worker_argv
```

- [ ] **Step 5.3: Run tests to verify they fail**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestSubmitSbatchWorker tests/test_unified_early_pipeline.py::TestDispatchParallelWorkers -v
```

Expected: ImportError (functions not defined).

- [ ] **Step 5.4: Implement `_submit_sbatch_worker`**

Add to `tuning/training/unified_early_pipeline.py`, immediately after `_build_base_cmd`:

```python
def _submit_sbatch_worker(sbatch_script, worker_args):
    """Submit an sbatch worker job, return the Slurm job ID as a string.

    Exits the orchestrator on sbatch error or unparseable output.
    """
    cmd = ["sbatch", sbatch_script, *worker_args]
    print(f"[orchestrator] Submitting sbatch worker: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"sbatch failed (code {result.returncode}): {result.stderr.strip()}")
    # Expected stdout: "Submitted batch job 12345"
    tokens = result.stdout.strip().split()
    if len(tokens) < 4 or tokens[0] != "Submitted":
        sys.exit(f"Unexpected sbatch stdout: {result.stdout!r}")
    return tokens[-1]
```

- [ ] **Step 5.5: Implement `_dispatch_parallel_workers`**

Add immediately after `_submit_sbatch_worker`:

```python
def _dispatch_parallel_workers(parallel, base_cmd, pt_flag, metadata_files):
    """Submit parallel-1 sbatch workers for post-training.

    No-op when parallel <= 1. Strips --parallel from worker args so
    workers don't recursively dispatch.
    """
    if parallel <= 1:
        return

    # Build worker args from base_cmd, stripping --parallel to prevent recursive dispatch
    worker_argv = []
    skip_next = False
    for tok in base_cmd[1:]:  # drop script path, keep args
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
        job_id = _submit_sbatch_worker(SBATCH_WORKER_SCRIPT, worker_argv)
        print(f"[orchestrator] Submitted worker {i+1}/{parallel-1}: job {job_id}")
```

- [ ] **Step 5.6: Call the dispatch function from `main()`**

In `main()`, insert this block immediately after `pt_flag = f"--run-{pt_method}" if pt_method != "dpo" else "--run-dpo"` and before `for metadata_file in all_files:`:

```python
    _dispatch_parallel_workers(
        parallel=args.parallel,
        base_cmd=base_cmd,
        pt_flag=pt_flag,
        metadata_files=all_files,
    )
```

No call-site guard needed — `_dispatch_parallel_workers` strips `--parallel` from worker args, so workers get `parallel=1` (the default) and the dispatch is a no-op.

- [ ] **Step 5.7: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestSubmitSbatchWorker tests/test_unified_early_pipeline.py::TestDispatchParallelWorkers -v
```

Expected: 8 passed.

- [ ] **Step 5.8: Run the full test suite**

```bash
python -m pytest tests/test_unified_early_pipeline.py -v
```

Expected: all tests pass.

- [ ] **Step 5.9: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Orchestrator dispatches N-1 sbatch workers when --parallel > 1"
```

---

### Task 6: Manual smoke test (no test code, but required)

**Files:** None modified.

**Goal:** Verify the real pipeline still works end-to-end. TDD unit tests covered function-level behavior; this confirms the integration actually runs under Slurm.

- [ ] **Step 6.1: Sanity check with `--parallel 1` (current behavior)**

On the cluster, with the .venv active (see feedback_workbb_venv) and `WANDB_API_KEY` exported:

```bash
sbatch tuning/slurm/unified_early_pipeline.sh \
    --model llama3-1B \
    --wandb-project parallel-test \
    --train-size 256 \
    --sft-data-size 128 \
    --dpo-data-size 128 \
    --sft-eval-steps 64 \
    --sft-passk-num-prompts 32 \
    --dpo-passk-num-prompts 32 \
    --parallel 1
```

Expected: runs as before, one sbatch job, sequential post-training. Check the output log `${JOBID}_parallel-test.out` for "No more checkpoints in..." at the end of post-training.

- [ ] **Step 6.2: Test with `--parallel 2`**

```bash
sbatch tuning/slurm/unified_early_pipeline.sh \
    --model llama3-1B \
    --wandb-project parallel-test \
    --train-size 256 \
    --sft-data-size 128 \
    --dpo-data-size 128 \
    --sft-eval-steps 64 \
    --sft-passk-num-prompts 32 \
    --dpo-passk-num-prompts 32 \
    --parallel 2
```

Expected:
- Original job runs SFT, then submits 1 sbatch worker
- `squeue -u $USER` shows 2 jobs running
- Worker's output log shows it skipped SFT, claimed a checkpoint, ran DPO
- Orchestrator's output log shows it also claimed a checkpoint, ran DPO, and exited with "No more checkpoints" when done
- Metadata JSONL shows each checkpoint `claimed: true, completed: true`

- [ ] **Step 6.3: Verify no double-claim**

After the runs in 6.2 finish, inspect the metadata file:

```bash
cat /path/to/metadata.jsonl | python -c "import json, sys; [print(json.loads(l).get('checkpoint_path'), json.loads(l).get('claimed'), json.loads(l).get('completed')) for l in sys.stdin]"
```

Expected: each checkpoint shows up once with both flags true. If any checkpoint was processed by two workers (a race), W&B would show two runs for the same checkpoint path — check `tags` filter.

- [ ] **Step 6.4: Document the result**

No code change. If 6.1-6.3 pass, the feature is working. If anything fails, file the issue back in this plan as a new task before proceeding.

---

## Spec Coverage Checklist (for self-review)

| Spec requirement | Task |
|---|---|
| `claimed` field in metadata | Task 1 |
| `claim_next_checkpoint()` | Task 1 |
| No file locking | Task 1 (implementation note) |
| `run_dpo`/`run_grpo` use claim + exit 42 | Task 3 |
| Post-training loop handles exit 42 | Task 4 |
| `--parallel N` CLI arg | Task 2 |
| Orchestrator submits N-1 sbatch workers | Task 5 |
| Orchestrator becomes Nth worker | Task 5 (falls through to loop) |
| Reuses `unified_early_pipeline.sh` | Task 5 (uses `SBATCH_WORKER_SCRIPT`) |
| Fire-and-forget (no polling) | Task 5 (no wait logic) |
