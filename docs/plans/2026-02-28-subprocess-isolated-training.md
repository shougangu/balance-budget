# Subprocess-Isolated Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `unified_early_pipeline.py` so each training stage (SFT, each DPO checkpoint) runs as an isolated subprocess, guaranteeing full GPU memory release between runs.

**Architecture:** `main()` with `--run-all` becomes a lightweight orchestrator that never imports torch. It spawns subprocesses for `--run-sft` and `--run-dpo`, forwarding `sys.argv` directly. The metadata JSONL file serves as a work queue — each DPO subprocess processes one checkpoint and marks it `completed: true`.

**Tech Stack:** Python subprocess, argparse, JSON

---

### Task 1: Add `next_checkpoint` and `mark_completed` helpers

These functions treat the metadata JSONL file as a work queue.

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (add two functions after `load_checkpoints`)
- Test: `tests/test_unified_early_pipeline.py` (add new test class)

**Step 1: Write the failing tests**

Add to `tests/test_unified_early_pipeline.py`:

```python
from tuning.training.unified_early_pipeline import next_checkpoint, mark_completed

class TestMetadataWorkQueue:
    def test_next_checkpoint_returns_first_row(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        result = next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp1"

    def test_next_checkpoint_skips_completed(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        completed_row = {**PASSK_ROW, "completed": True}
        _write_jsonl(f, [completed_row, PPL_ROW])
        result = next_checkpoint(str(f))
        assert result["checkpoint_path"] == "/models/cp2"

    def test_next_checkpoint_returns_none_when_all_completed(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [{**PASSK_ROW, "completed": True}])
        assert next_checkpoint(str(f)) is None

    def test_mark_completed_sets_flag(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        mark_completed(str(f), "/models/cp1")
        with open(f) as fh:
            lines = [json.loads(l) for l in fh]
        assert lines[0]["completed"] is True
        assert "completed" not in lines[1]

    def test_mark_completed_preserves_other_fields(self, tmp_path):
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW])
        mark_completed(str(f), "/models/cp1")
        with open(f) as fh:
            row = json.loads(fh.readline())
        assert row["data_points_seen"] == 512
        assert row["threshold_type"] == "pass_at_1"
        assert row["completed"] is True
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_unified_early_pipeline.py::TestMetadataWorkQueue -v`
Expected: ImportError — `next_checkpoint` and `mark_completed` don't exist yet.

**Step 3: Write minimal implementation**

Add to `unified_early_pipeline.py` after `load_checkpoints`:

```python
def next_checkpoint(metadata_file):
    """Return the first non-completed checkpoint row, or None."""
    with open(metadata_file) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("completed"):
                return row
    return None


def mark_completed(metadata_file, checkpoint_path):
    """Mark a checkpoint as completed in the metadata file."""
    with open(metadata_file) as f:
        lines = f.readlines()
    with open(metadata_file, "w") as f:
        for line in lines:
            row = json.loads(line)
            if row["checkpoint_path"] == checkpoint_path:
                row["completed"] = True
            f.write(json.dumps(row) + "\n")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_unified_early_pipeline.py::TestMetadataWorkQueue -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "feat: add next_checkpoint and mark_completed metadata helpers"
```

---

### Task 2: Make `run_sft` print metadata paths to stdout

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:285-294` (`run_sft` return block)
- Test: `tests/test_unified_early_pipeline.py`

**Step 1: Write the failing test**

```python
class TestRunSftMetadataOutput:
    def test_metadata_prefix_format(self, capsys, tmp_path):
        """Verify the METADATA_FILE: prefix format used for IPC."""
        # We can't run actual SFT in tests, but we can test the output format
        # by calling the print logic directly. Extract it as a helper.
        from tuning.training.unified_early_pipeline import print_metadata_paths
        paths = [str(tmp_path / "a.jsonl"), str(tmp_path / "b.jsonl")]
        print_metadata_paths(paths)
        captured = capsys.readouterr()
        lines = [l for l in captured.out.splitlines() if l.startswith("METADATA_FILE:")]
        assert len(lines) == 2
        assert lines[0] == f"METADATA_FILE:{paths[0]}"
        assert lines[1] == f"METADATA_FILE:{paths[1]}"

    def test_parse_metadata_from_stdout(self, tmp_path):
        """Verify the orchestrator can parse METADATA_FILE: lines from mixed output."""
        from tuning.training.unified_early_pipeline import parse_metadata_from_output
        output = f"Some training log\nMETADATA_FILE:{tmp_path}/a.jsonl\nMore logs\nMETADATA_FILE:{tmp_path}/b.jsonl\n"
        result = parse_metadata_from_output(output)
        assert result == [f"{tmp_path}/a.jsonl", f"{tmp_path}/b.jsonl"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_unified_early_pipeline.py::TestRunSftMetadataOutput -v`
Expected: ImportError.

**Step 3: Write minimal implementation**

Add to `unified_early_pipeline.py`:

```python
def print_metadata_paths(paths):
    """Print metadata file paths with a prefix for subprocess IPC."""
    for p in paths:
        print(f"METADATA_FILE:{p}")


def parse_metadata_from_output(output):
    """Extract metadata file paths from subprocess stdout."""
    return [
        line.split(":", 1)[1]
        for line in output.splitlines()
        if line.startswith("METADATA_FILE:")
    ]
```

Then in `run_sft`, after computing `metadata_paths`, add:

```python
    print_metadata_paths(metadata_paths)
    return metadata_paths
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_unified_early_pipeline.py::TestRunSftMetadataOutput -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "feat: add metadata path IPC via stdout prefix"
```

---

### Task 3: Refactor `run_dpo` to process one checkpoint

Change `run_dpo` from looping over all checkpoints to processing exactly one (the next non-completed), then marking it completed.

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:333-427` (`run_dpo`)
- Test: `tests/test_unified_early_pipeline.py`

**Step 1: Write the failing test**

```python
class TestRunDpoSingleCheckpoint:
    def test_run_dpo_marks_checkpoint_completed(self, tmp_path):
        """Verify that after run_dpo processes a checkpoint, it's marked completed."""
        f = tmp_path / "meta.jsonl"
        _write_jsonl(f, [PASSK_ROW, PPL_ROW])
        # We can't run real DPO, but we can verify the contract:
        # after processing, the checkpoint should be marked completed.
        mark_completed(str(f), PASSK_ROW["checkpoint_path"])
        row = next_checkpoint(str(f))
        assert row["checkpoint_path"] == "/models/cp2"
```

Note: The actual `run_dpo` refactor can't be unit tested without a GPU. The test above validates the work-queue contract. Integration testing happens by running the pipeline.

**Step 2: Run test to verify it passes** (this one tests the helpers, so it should pass already)

Run: `pytest tests/test_unified_early_pipeline.py::TestRunDpoSingleCheckpoint -v`

**Step 3: Refactor `run_dpo`**

Change the signature and body:

```python
def run_dpo(args):
    """Run DPO for the next non-completed checkpoint in the metadata file."""
    metadata_file = args.metadata_file[0]  # single file in subprocess mode
    checkpoint = next_checkpoint(metadata_file)
    if checkpoint is None:
        print("All checkpoints completed, nothing to do.")
        return

    remaining = args.train_size - checkpoint["data_points_seen"]
    if remaining <= 0:
        print(f"Skipping {checkpoint['checkpoint_path']}: no data budget remaining")
        mark_completed(metadata_file, checkpoint["checkpoint_path"])
        return

    import wandb
    from tuning.config import HF_MODEL_MAP, set_chat_template
    from tuning.training.config_training import (
        DatasetConfig, SFTRunConfig, PTRunConfig, ModelLoadConfig,
        LoraConfig, DPOTrainingConfig,
    )
    from tuning.training.dpo_training import train_model_dpo

    set_chat_template(args.model)
    gpu_util = MODEL_TO_GPU_2[args.model]
    model_name = Path(checkpoint["checkpoint_path"]).name

    dataset_config = DatasetConfig(
        dataset=args.dataset, dataset_type="pt", train_size=remaining,
    )
    sft_run_config = SFTRunConfig(
        dataset_config=DatasetConfig(
            dataset=args.dataset, dataset_type="sft",
            train_size=checkpoint["data_points_seen"],
            dynamic_path=model_name,
        ),
        model_name=args.model,
        model_name_hf=HF_MODEL_MAP[args.model],
        task_name=args.task_name,
    )
    run_config = PTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[args.model],
        model_name=args.model,
        sft_run_config=sft_run_config,
        task_name=args.task_name,
        pft_method="dpo",
        do_training=True,
    )
    lora_config = LoraConfig()
    lora_config.use_gradient_checkpointing = True
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length
    training_args = DPOTrainingConfig()
    training_args.eval_steps = args.dpo_eval_steps
    training_args.per_device_train_batch_size = args.dpo_batch_size
    training_args.gradient_accumulation_steps = args.dpo_grad_accum

    passk_config = _dpo_passk_config(args, gpu_util)
    ppl_config = _dpo_ppl_config(args)

    tags = ["dpo", str(checkpoint["threshold_value"])]
    if passk_config is not None:
        tags.append(f"p{passk_config.k_values[0]}")
    if ppl_config is not None:
        tags.append("ppl")

    with wandb.init(
        name=run_config.model_name, project=args.wandb_project,
        job_type="dpo", tags=tags, config={"stage": "dpo"},
    ):
        train_model_dpo(
            run_config=run_config, lora_config=lora_config,
            model_load_config=model_load_config, training_args=training_args,
            passk_config=passk_config, perplexity_config=ppl_config,
        )

    mark_completed(metadata_file, checkpoint["checkpoint_path"])
```

Remove all GPU cleanup code (`del model`, `cleanup_gpu()`, `nvidia-smi`, memory logging) — process exit handles this.

**Step 4: Run existing tests to verify nothing broke**

Run: `pytest tests/test_unified_early_pipeline.py -v`
Expected: All existing tests still PASS.

**Step 5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "refactor: run_dpo processes one checkpoint per invocation"
```

---

### Task 4: Refactor `main()` as subprocess orchestrator

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:427-449` (`main`)
- Test: `tests/test_unified_early_pipeline.py`

**Step 1: Write the failing test**

```python
class TestBuildSubprocessCmd:
    def test_strips_run_all(self):
        from tuning.training.unified_early_pipeline import _build_base_cmd
        original = ["/usr/bin/python", "pipeline.py", "--model", "llama3-3B", "--run-all", "--wandb-project", "tuning"]
        result = _build_base_cmd(original)
        assert "--run-all" not in result
        assert "--model" in result

    def test_preserves_other_args(self):
        from tuning.training.unified_early_pipeline import _build_base_cmd
        original = ["/usr/bin/python", "pipeline.py", "--model", "llama3-3B", "--run-all", "--train-size", "5000"]
        result = _build_base_cmd(original)
        assert "--train-size" in result
        assert "5000" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_unified_early_pipeline.py::TestBuildSubprocessCmd -v`
Expected: ImportError.

**Step 3: Write implementation**

Add helper:

```python
def _build_base_cmd(argv):
    """Build base subprocess command by stripping orchestrator-only flags."""
    return [a for a in argv if a != "--run-all"]
```

Rewrite `main()`:

```python
def main():
    args = _parse_args()
    print(args)

    if not any([args.run_sft, args.run_dpo, args.run_all]):
        args.run_sft = True
        args.run_dpo = True

    # Direct worker mode: run in-process
    if args.run_sft and not args.run_all:
        run_sft(args)
        return

    if args.run_dpo and not args.run_all:
        run_dpo(args)
        return

    # Orchestrator mode: spawn subprocesses
    base_cmd = _build_base_cmd(sys.argv)

    # SFT subprocess
    sft_cmd = [sys.executable] + base_cmd[1:] + ["--run-sft"]
    result = subprocess.run(sft_cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        sys.exit(f"SFT subprocess failed with return code {result.returncode}")

    metadata_files = parse_metadata_from_output(result.stdout)
    if not metadata_files and not args.metadata_file:
        sys.exit("No metadata files from SFT and no --metadata-file provided")
    all_files = metadata_files + (args.metadata_file or [])

    # DPO subprocess loop
    for metadata_file in all_files:
        while next_checkpoint(metadata_file) is not None:
            dpo_cmd = [sys.executable] + base_cmd[1:] + [
                "--run-dpo", "--metadata-file", metadata_file,
            ]
            result = subprocess.run(dpo_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            if result.returncode != 0:
                sys.exit(f"DPO subprocess failed with return code {result.returncode}")
```

**Step 4: Run all tests**

Run: `pytest tests/test_unified_early_pipeline.py -v`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "refactor: main() orchestrates SFT and DPO as subprocesses"
```

---

### Task 5: Guard torch/unsloth imports for orchestrator mode

The orchestrator must not import torch/CUDA. Move the top-of-file CUDA restriction and unsloth import behind a guard so they only run in worker mode.

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:1-12` (top-of-file imports)

**Step 1: Refactor top of file**

```python
import os
import argparse
import json
import sys
import subprocess
from pathlib import Path


def _init_cuda_env():
    """Restrict CUDA to first visible GPU for training. Must be called before torch imports."""
    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if all_gpus:
        os.environ["CUDA_VISIBLE_DEVICES_ALL"] = all_gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus.split(",")[0]


def _is_worker_mode():
    """Check if we're running as a worker (--run-sft or --run-dpo without --run-all)."""
    return ("--run-sft" in sys.argv or "--run-dpo" in sys.argv) and "--run-all" not in sys.argv


# Only init CUDA and import unsloth in worker mode
if _is_worker_mode():
    _init_cuda_env()
    import unsloth  # noqa: F401
```

**Step 2: Run all tests**

Run: `pytest tests/test_unified_early_pipeline.py -v`
Expected: All tests PASS. (Tests import specific functions, not the whole module in worker mode.)

**Step 3: Commit**

```bash
git add tuning/training/unified_early_pipeline.py
git commit -m "refactor: guard CUDA init and unsloth import behind worker mode check"
```

---

### Task 6: Remove GPU cleanup code from `run_dpo`

Now that each DPO run is a subprocess, the manual cleanup is unnecessary.

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py` (`run_dpo`)

**Step 1: Remove cleanup code**

Remove from `run_dpo`:
- `del trainer.ref_model` / `del trainer.model` / `trainer.accelerator.free_memory()`
- `del model, tokenizer, trainer, _`
- `cleanup_gpu()`
- `subprocess.check_output("nvidia-smi")`
- `torch.cuda.memory_allocated` / `memory_reserved` logging
- `torch.cuda.memory._record_memory_history` / `_dump_snapshot` profiling code

The process exits after `mark_completed`, which frees everything.

**Step 2: Run all tests**

Run: `pytest tests/test_unified_early_pipeline.py -v`
Expected: All tests PASS.

**Step 3: Commit**

```bash
git add tuning/training/unified_early_pipeline.py
git commit -m "cleanup: remove manual GPU cleanup from run_dpo (subprocess exit handles it)"
```

---

### Task 7: End-to-end smoke test

**This is a manual step.** Run the pipeline on a GPU node to verify the subprocess orchestration works.

**Step 1: Allocate GPU**

```bash
f1  # 3-hour H100
```

**Step 2: Run with small budget**

```bash
python tuning/training/unified_early_pipeline.py \
    --model llama3-1B \
    --wandb-project tuning \
    --train-size 500 \
    --run-all
```

**Step 3: Verify**

- SFT runs as subprocess, prints `METADATA_FILE:` lines
- Each DPO checkpoint runs as separate subprocess
- `nvidia-smi` between runs shows GPU memory fully freed
- Metadata file has `"completed": true` on processed rows
- Re-running the same command skips completed checkpoints
