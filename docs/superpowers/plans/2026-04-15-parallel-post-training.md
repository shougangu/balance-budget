# Parallel Post-Training Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `--parallel N` fan out post-training checkpoints across N sbatch GPU allocations, so N checkpoints can train concurrently instead of sequentially.

**Architecture:** Add a `claimed` field to checkpoint rows in the JSONL metadata. A new `claim_next_checkpoint()` atomically (via regular read-modify-write, no locking) picks the next unclaimed+uncompleted row and marks it claimed. Workers exit with code 42 when nothing is available to claim. The orchestrator submits `N-1` sbatch worker jobs (fire and forget) and runs the post-training loop itself as the Nth worker.

**Tech Stack:** Python stdlib (`subprocess`, `json`, `argparse`, `sys`), pytest, Slurm (`sbatch`). No new dependencies.

---

## File Structure

| File | Role |
|------|------|
| `tuning/training/unified_early_pipeline.py` | New `claim_next_checkpoint()`, `_submit_sbatch_worker()`; `--parallel` CLI arg; updates to `run_dpo`/`run_grpo`/`_build_base_cmd` and the orchestrator loop |
| `tests/test_unified_early_pipeline.py` | Unit tests for `claim_next_checkpoint`, updated `_build_base_cmd` tests, `--parallel` arg tests, exit-42 behavior |

Spec: `docs/superpowers/specs/2026-04-15-parallel-post-training-design.md`

---

### Task 1: Add `claim_next_checkpoint()` function

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (add function near `next_checkpoint` at line 454)
- Test: `tests/test_unified_early_pipeline.py` (add class near `TestMetadataWorkQueue` at line 231)

**Goal:** Atomic-ish read-modify-write that picks the next unclaimed+uncompleted checkpoint and writes `"claimed": true` back before returning.

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

- [ ] **Step 1.3: Implement `claim_next_checkpoint`**

Add to `tuning/training/unified_early_pipeline.py`, immediately after `next_checkpoint` (currently ends at line 462):

```python
def claim_next_checkpoint(metadata_file):
    """Pick the next unclaimed+uncompleted checkpoint and mark it claimed.

    Reads the JSONL, finds the first row where neither 'claimed' nor 'completed'
    is True, writes 'claimed: true' back to the file, returns the row. Returns
    None when nothing is available.

    No file locking: race window is near-simultaneous sbatch starts, worst case
    is one checkpoint trained twice. Fine for our use case.
    """
    with open(metadata_file) as f:
        lines = f.readlines()

    claimed_row = None
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if claimed_row is None and not row.get("claimed") and not row.get("completed"):
            row["claimed"] = True
            claimed_row = row
        new_lines.append(json.dumps(row) + "\n")

    if claimed_row is None:
        return None

    with open(metadata_file, "w") as f:
        f.writelines(new_lines)

    print(f"Claimed checkpoint: {claimed_row['checkpoint_path']} (threshold {claimed_row.get('threshold_value')}, type {claimed_row.get('threshold_type')})")
    return claimed_row
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestClaimNextCheckpoint -v
```

Expected: 7 passed.

- [ ] **Step 1.5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Add claim_next_checkpoint for parallel worker coordination"
```

---

### Task 2: Update `_build_base_cmd` to strip stage and metadata flags

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:746-748`
- Test: `tests/test_unified_early_pipeline.py:416-431` (update existing `TestBuildBaseCmd` class)

**Goal:** Currently `_build_base_cmd` only strips `--run-all`. When workers re-enter orchestrator mode with `--run-dpo --run-all --metadata-file X`, the inner subprocess ends up with duplicate `--run-dpo` / `--metadata-file` args. Strip all stage flags and `--metadata-file` so they can be re-added cleanly.

- [ ] **Step 2.1: Write the failing tests**

Replace the existing `TestBuildBaseCmd` class in `tests/test_unified_early_pipeline.py` (starting at line 416) with:

```python
class TestBuildBaseCmd:
    def test_strips_run_all(self):
        original = ["/usr/bin/python", "pipeline.py", "--model", "llama3-3B", "--run-all", "--wandb-project", "tuning"]
        result = _build_base_cmd(original)
        assert "--run-all" not in result
        assert "--model" in result

    def test_strips_run_dpo(self):
        original = ["/usr/bin/python", "pipeline.py", "--run-dpo", "--model", "llama3-3B"]
        result = _build_base_cmd(original)
        assert "--run-dpo" not in result

    def test_strips_run_grpo(self):
        original = ["/usr/bin/python", "pipeline.py", "--run-grpo", "--model", "llama3-3B"]
        result = _build_base_cmd(original)
        assert "--run-grpo" not in result

    def test_strips_run_sft(self):
        original = ["/usr/bin/python", "pipeline.py", "--run-sft", "--model", "llama3-3B"]
        result = _build_base_cmd(original)
        assert "--run-sft" not in result

    def test_strips_metadata_file_and_value(self):
        original = ["/usr/bin/python", "pipeline.py", "--metadata-file", "/tmp/a.jsonl", "--model", "llama3-3B"]
        result = _build_base_cmd(original)
        assert "--metadata-file" not in result
        assert "/tmp/a.jsonl" not in result
        assert "--model" in result

    def test_strips_multiple_metadata_files(self):
        original = [
            "python", "pipeline.py",
            "--metadata-file", "/tmp/a.jsonl",
            "--metadata-file", "/tmp/b.jsonl",
            "--model", "llama3-3B",
        ]
        result = _build_base_cmd(original)
        assert "--metadata-file" not in result
        assert "/tmp/a.jsonl" not in result
        assert "/tmp/b.jsonl" not in result

    def test_strips_parallel_and_value(self):
        original = ["python", "pipeline.py", "--parallel", "3", "--model", "llama3-3B"]
        result = _build_base_cmd(original)
        assert "--parallel" not in result
        assert "3" not in result
        assert "--model" in result

    def test_preserves_other_args(self):
        original = ["/usr/bin/python", "pipeline.py", "--model", "llama3-3B", "--run-all", "--train-size", "5000"]
        result = _build_base_cmd(original)
        assert "--train-size" in result
        assert "5000" in result

    def test_no_orchestrator_flags_unchanged(self):
        original = ["/usr/bin/python", "pipeline.py", "--model", "llama3-3B"]
        assert _build_base_cmd(original) == original
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestBuildBaseCmd -v
```

Expected: several failures (new strip cases all fail, `--parallel` case also fails since arg doesn't exist yet but the string-based strip function doesn't care).

- [ ] **Step 2.3: Update `_build_base_cmd`**

Replace the function in `tuning/training/unified_early_pipeline.py` (currently lines 746-748):

```python
def _build_base_cmd(argv):
    """Build base subprocess command by stripping orchestrator-only flags.

    Strips --run-all, --run-sft, --run-dpo, --run-grpo, --parallel (plus its
    value), and --metadata-file (plus each value). The caller re-adds the
    exact stage flag and metadata files needed for the subprocess.
    """
    flag_without_value = {"--run-all", "--run-sft", "--run-dpo", "--run-grpo"}
    flag_with_value = {"--metadata-file", "--parallel"}
    result = []
    skip_next = False
    for tok in argv:
        if skip_next:
            skip_next = False
            continue
        if tok in flag_without_value:
            continue
        if tok in flag_with_value:
            skip_next = True
            continue
        result.append(tok)
    return result
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestBuildBaseCmd -v
```

Expected: 9 passed.

- [ ] **Step 2.5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Strip all stage and metadata flags in _build_base_cmd"
```

---

### Task 3: Add `--parallel` CLI arg

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (inside `_parse_args`, near the stage group at line 83-94)
- Test: `tests/test_unified_early_pipeline.py` (add to `TestParseArgs` class)

- [ ] **Step 3.1: Write the failing tests**

Find `TestParseArgs` class in `tests/test_unified_early_pipeline.py` (around line 142) and add:

```python
class TestParallelArg:
    def test_default_parallel_is_1(self):
        args = _parse_args(REQUIRED)
        assert args.parallel == 1

    def test_parallel_accepts_integer(self):
        args = _parse_args(REQUIRED + ["--parallel", "3"])
        assert args.parallel == 3

    def test_parallel_rejects_zero(self):
        with pytest.raises(SystemExit):
            _parse_args(REQUIRED + ["--parallel", "0"])

    def test_parallel_rejects_negative(self):
        with pytest.raises(SystemExit):
            _parse_args(REQUIRED + ["--parallel", "-2"])
```

Add this as a new class right after the existing `TestParseArgs` or wherever ordering makes sense.

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestParallelArg -v
```

Expected: failures (`parallel` attr doesn't exist).

- [ ] **Step 3.3: Add the arg and validator**

Add a validator helper near `parse_early_tuple` (around line 61):

```python
def _positive_int(s):
    """Argparse type: accept integers >= 1."""
    try:
        v = int(s)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"Invalid integer: {s!r}")
    if v < 1:
        raise argparse.ArgumentTypeError(f"Must be >= 1, got {v}")
    return v
```

In the stage group inside `_parse_args` (after line 94):

```python
stage.add_argument("--parallel", type=_positive_int, default=1,
                   help="Number of concurrent post-training workers. "
                        "When >1, the orchestrator submits --parallel-1 sbatch jobs "
                        "and runs as the Nth worker itself.")
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestParallelArg -v
```

Expected: 4 passed.

- [ ] **Step 3.5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Add --parallel CLI arg for parallel post-training workers"
```

---

### Task 4: Worker mode exits 42 when no checkpoint to claim

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:494-515` (`run_dpo`)
- Modify: `tuning/training/unified_early_pipeline.py:623-644` (`run_grpo`)
- Test: `tests/test_unified_early_pipeline.py` (new `TestWorkerExitCode` class)

**Goal:** Swap `next_checkpoint()` for `claim_next_checkpoint()` inside `run_dpo` and `run_grpo`. When nothing to claim, `sys.exit(42)`.

- [ ] **Step 4.1: Write the failing tests**

Add to `tests/test_unified_early_pipeline.py`:

```python
class TestWorkerExitCode:
    def test_run_dpo_exits_42_when_nothing_to_claim(self, tmp_path, monkeypatch):
        from tuning.training import unified_early_pipeline as uep
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "completed": True}])
        args = argparse.Namespace(metadata_file=[str(f)])
        with pytest.raises(SystemExit) as exc:
            uep.run_dpo(args)
        assert exc.value.code == 42

    def test_run_grpo_exits_42_when_nothing_to_claim(self, tmp_path, monkeypatch):
        from tuning.training import unified_early_pipeline as uep
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "claimed": True}])
        args = argparse.Namespace(metadata_file=[str(f)])
        with pytest.raises(SystemExit) as exc:
            uep.run_grpo(args)
        assert exc.value.code == 42
```

- [ ] **Step 4.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestWorkerExitCode -v
```

Expected: failures (current `run_dpo` uses `next_checkpoint` and returns without exit code).

- [ ] **Step 4.3: Update `run_dpo`**

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

- [ ] **Step 4.4: Update `run_grpo`**

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

- [ ] **Step 4.5: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestWorkerExitCode -v
```

Expected: 2 passed.

- [ ] **Step 4.6: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Worker modes claim checkpoints and exit 42 when none available"
```

---

### Task 5: Post-training loop handles exit 42

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:797-804` (just the inner `while` loop)

**Goal:** The loop currently polls `next_checkpoint()` in the orchestrator process. Now each subprocess claims its own checkpoint. The loop keeps spawning subprocesses until one returns 42.

- [ ] **Step 5.1: Replace the inner `while` loop**

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

- [ ] **Step 5.2: Smoke-test the full test suite compiles and runs**

```bash
python -m pytest tests/test_unified_early_pipeline.py -v
```

Expected: all existing tests pass.

- [ ] **Step 5.3: Commit**

```bash
git add tuning/training/unified_early_pipeline.py
git commit -m "Post-training loop iterates until worker exits 42"
```

---

### Task 6: Add `_submit_sbatch_worker` helper

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (new function near `_build_base_cmd` at line 746)
- Test: `tests/test_unified_early_pipeline.py` (new `TestSubmitSbatchWorker` class)

**Goal:** Function that shells out to `sbatch`, parses the job ID from stdout, returns it. Errors exit the orchestrator.

- [ ] **Step 6.1: Write the failing tests**

Add to `tests/test_unified_early_pipeline.py`. Also add `_submit_sbatch_worker` to the imports at lines 10-19.

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
```

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestSubmitSbatchWorker -v
```

Expected: ImportError (`_submit_sbatch_worker` is not defined).

- [ ] **Step 6.3: Implement `_submit_sbatch_worker`**

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

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestSubmitSbatchWorker -v
```

Expected: 3 passed.

- [ ] **Step 6.5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Add _submit_sbatch_worker helper for parallel workers"
```

---

### Task 7: Orchestrator dispatches N-1 sbatch workers when `--parallel > 1`

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (the orchestrator block after SFT, around line 788)

**Goal:** Before entering the post-training loop, if `--parallel > 1` and the orchestrator is not already a worker (i.e., not already called with `--run-dpo` or `--run-grpo`), submit `parallel - 1` sbatch worker jobs. The orchestrator then runs the post-training loop itself (the Nth worker).

Workers are invoked with the same sbatch script (`tuning/slurm/unified_early_pipeline.sh`), and args reconstructed from `base_cmd` plus `--run-<method> --run-all --metadata-file <each>`. `--parallel` is stripped from `base_cmd` in Task 2, so workers don't recursively spawn more workers.

- [ ] **Step 7.1: Add a module-level constant for the sbatch script path**

At the top of `tuning/training/unified_early_pipeline.py`, just after the imports (around line 10):

```python
SBATCH_WORKER_SCRIPT = "tuning/slurm/unified_early_pipeline.sh"
```

- [ ] **Step 7.2: Write the failing tests for the dispatch function**

Add `_dispatch_parallel_workers` to the imports in `tests/test_unified_early_pipeline.py` (lines 10-19).

Add to `tests/test_unified_early_pipeline.py`:

```python
class TestDispatchParallelWorkers:
    def test_parallel_1_does_nothing(self, monkeypatch):
        from tuning.training import unified_early_pipeline as uep
        calls = []
        monkeypatch.setattr(uep, "_submit_sbatch_worker",
                            lambda *a, **k: (calls.append(a), "999")[1])
        uep._dispatch_parallel_workers(
            parallel=1,
            is_worker=False,
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
            is_worker=False,
            base_cmd=["pipeline.py", "--model", "llama3-3B"],
            pt_flag="--run-dpo",
            metadata_files=[str(mf)],
        )
        assert len(calls) == 2

    def test_worker_invocation_does_not_redispatch(self, monkeypatch, tmp_path):
        from tuning.training import unified_early_pipeline as uep
        calls = []
        monkeypatch.setattr(uep, "_submit_sbatch_worker",
                            lambda *a, **k: (calls.append(a), "999")[1])
        mf = tmp_path / "meta.jsonl"
        mf.write_text("{}\n")
        uep._dispatch_parallel_workers(
            parallel=3,
            is_worker=True,
            base_cmd=["pipeline.py"],
            pt_flag="--run-dpo",
            metadata_files=[str(mf)],
        )
        assert calls == []

    def test_worker_argv_includes_pt_flag_and_metadata(self, monkeypatch, tmp_path):
        from tuning.training import unified_early_pipeline as uep
        calls = []
        monkeypatch.setattr(uep, "_submit_sbatch_worker",
                            lambda script, argv: (calls.append(argv), "999")[1])
        mf = tmp_path / "meta.jsonl"
        mf.write_text("{}\n")
        uep._dispatch_parallel_workers(
            parallel=2,
            is_worker=False,
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
            is_worker=False,
            base_cmd=["pipeline.py"],
            pt_flag="--run-dpo",
            metadata_files=[str(real_mf), missing_mf],
        )
        worker_argv = calls[0]
        assert str(real_mf) in worker_argv
        assert missing_mf not in worker_argv
```

- [ ] **Step 7.3: Run the tests to verify they fail**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestDispatchParallelWorkers -v
```

Expected: ImportError (`_dispatch_parallel_workers` is not defined).

- [ ] **Step 7.4: Implement `_dispatch_parallel_workers`**

Add to `tuning/training/unified_early_pipeline.py`, immediately after `_submit_sbatch_worker`:

```python
def _dispatch_parallel_workers(parallel, is_worker, base_cmd, pt_flag, metadata_files):
    """Submit parallel-1 sbatch workers for post-training.

    No-op when parallel <= 1 or when this invocation is already a worker
    (already re-entered from a previous dispatch, indicated by --run-dpo
    or --run-grpo being set by the outer orchestrator).
    """
    if parallel <= 1 or is_worker:
        return

    worker_argv = list(base_cmd[1:])  # drop script path, keep args
    worker_argv += [pt_flag, "--run-all"]
    for mf in metadata_files:
        if Path(mf).is_file():
            worker_argv += ["--metadata-file", mf]

    for i in range(parallel - 1):
        job_id = _submit_sbatch_worker(SBATCH_WORKER_SCRIPT, worker_argv)
        print(f"[orchestrator] Submitted worker {i+1}/{parallel-1}: job {job_id}")
```

- [ ] **Step 7.5: Call the dispatch function from `main()`**

In `main()`, insert this line immediately after `pt_flag = f"--run-{pt_method}" if pt_method != "dpo" else "--run-dpo"` and before `for metadata_file in all_files:`:

```python
    _dispatch_parallel_workers(
        parallel=args.parallel,
        is_worker=args.run_dpo or args.run_grpo,
        base_cmd=base_cmd,
        pt_flag=pt_flag,
        metadata_files=all_files,
    )
```

The worker invocation uses `--run-dpo --run-all --metadata-file ...`, which lands in the orchestrator branch of `main()` but skips SFT (because `args.run_dpo` is True), and enters the post-training loop. Inside that worker, `args.parallel` is stripped by `_build_base_cmd`, so the worker won't recursively spawn more workers.

- [ ] **Step 7.6: Run the new tests**

```bash
python -m pytest tests/test_unified_early_pipeline.py::TestDispatchParallelWorkers -v
```

Expected: 5 passed.

- [ ] **Step 7.7: Run the full test suite**

```bash
python -m pytest tests/test_unified_early_pipeline.py -v
```

Expected: all tests pass (new `TestDispatchParallelWorkers` + everything from prior tasks).

- [ ] **Step 7.8: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Orchestrator dispatches N-1 sbatch workers when --parallel > 1"
```

---

### Task 8: Manual smoke test (no test code, but required)

**Files:** None modified.

**Goal:** Verify the real pipeline still works end-to-end. TDD unit tests covered function-level behavior; this confirms the integration actually runs under Slurm.

- [ ] **Step 8.1: Sanity check with `--parallel 1` (current behavior)**

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

- [ ] **Step 8.2: Test with `--parallel 2`**

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

- [ ] **Step 8.3: Verify no double-claim**

After the runs in 8.2 finish, inspect the metadata file:

```bash
cat /path/to/metadata.jsonl | python -c "import json, sys; [print(json.loads(l).get('checkpoint_path'), json.loads(l).get('claimed'), json.loads(l).get('completed')) for l in sys.stdin]"
```

Expected: each checkpoint shows up once with both flags true. If any checkpoint was processed by two workers (a race), W&B would show two runs for the same checkpoint path — check `tags` filter.

- [ ] **Step 8.4: Document the result**

No code change. If 8.1-8.3 pass, the feature is working. If anything fails, file the issue back in this plan as a new task before proceeding.

---

## Spec Coverage Checklist (for self-review)

| Spec requirement | Task |
|---|---|
| `claimed` field in metadata | Task 1 |
| `claim_next_checkpoint()` | Task 1 |
| No file locking | Task 1 (implementation note) |
| `run_dpo`/`run_grpo` use claim + exit 42 | Task 4 |
| Post-training loop handles exit 42 | Task 5 |
| `--parallel N` CLI arg | Task 3 |
| Orchestrator submits N-1 sbatch workers | Task 7 |
| Orchestrator becomes Nth worker | Task 7 (falls through to loop) |
| Reuses `unified_early_pipeline.sh` | Task 7 (uses `SBATCH_WORKER_SCRIPT`) |
| Fire-and-forget (no polling) | Task 7 (no wait logic) |
| `--parallel` stripped from worker invocations | Task 2 |
