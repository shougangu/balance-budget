# Subprocess-Isolated Training Runs

## Problem

`unified_early_pipeline.py` runs SFT and multiple DPO training stages in a single process. CUDA memory cannot be fully freed without process exit, causing memory leaks across DPO checkpoint iterations.

## Design

Split the pipeline into orchestrator and worker modes within the same file.

### Orchestrator mode (`--run-all`)

Never imports torch/CUDA. Spawns subprocesses, reads metadata files, loops until all checkpoints are processed.

### Worker mode (`--run-sft`, `--run-dpo`)

Runs in a subprocess with a fresh CUDA context. Executes a single training stage, then exits (freeing all GPU memory).

## Flow

```
orchestrator (no CUDA)
  ├─ subprocess: --run-sft → writes metadata JSONL, prints METADATA_FILE:<path>
  │                           process exits, GPU freed
  ├─ reads metadata file path from stdout
  ├─ subprocess: --run-dpo --metadata-file <path>
  │     reads first non-completed line, trains, marks completed:true
  │     process exits, GPU freed
  ├─ subprocess: --run-dpo --metadata-file <path>
  │     reads next non-completed line, trains, marks completed:true
  │     process exits, GPU freed
  └─ ... until all lines completed
```

## Subprocess invocation

Forward `sys.argv` directly, stripping `--run-all` and adding stage-specific flags:

```python
base_cmd = [sys.executable] + [a for a in sys.argv if a != "--run-all"]
sft_cmd = base_cmd + ["--run-sft"]
dpo_cmd = base_cmd + ["--run-dpo", "--metadata-file", metadata_path]
```

## Metadata file as work queue

- SFT callbacks write checkpoint JSONL as today
- Each DPO subprocess reads the first line without `"completed": true`, trains that checkpoint, marks it `"completed": true`
- Crash recovery: re-run the same command, completed checkpoints are skipped

## Changes

1. `main()` — in `--run-all` mode, becomes orchestrator (subprocess spawning + stdout parsing)
2. `run_sft()` — prints `METADATA_FILE:<path>` to stdout at end
3. `run_dpo()` — processes exactly one checkpoint per invocation, marks it completed
4. GPU cleanup code in `run_dpo` — removed (process exit handles it)
5. Top-of-file `CUDA_VISIBLE_DEVICES` logic — unchanged, reused by each subprocess

## Unchanged

- `sft_training.py`, `dpo_training.py` — untouched
- Callback code — untouched
- CLI args forwarding — automatic via `sys.argv`
- `--run-sft` / `--run-dpo` standalone usage — still works
