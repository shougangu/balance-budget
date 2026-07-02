# ABOUTME: JSONL helpers for SFT-checkpoint metadata: claim, complete, list, parse-from-stdout.
# ABOUTME: Pure I/O — no torch/wandb dependencies.

import json
import sys
from pathlib import Path


def load_checkpoints(metadata_files, merge):
    """Load and filter checkpoint rows from one or more JSONL metadata files.

    Args:
        metadata_files: List of file paths to read.
        merge: "union" keeps all rows; "passk" keeps only pass_at_* rows;
               "ppl" keeps only perplexity rows.

    Returns:
        Deduplicated list of checkpoint dicts (first occurrence wins).
    """
    checkpoints = []
    seen_paths = set()
    for path in metadata_files:
        if not Path(path).is_file():
            print(f"Warning: metadata file {path} does not exist, skipping")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                ttype = row.get("threshold_type", "")
                if merge == "passk" and not ttype.startswith("pass_at_"):
                    continue
                if merge == "ppl" and ttype != "perplexity":
                    continue
                cp = row["checkpoint_path"]
                if cp not in seen_paths:
                    seen_paths.add(cp)
                    checkpoints.append(row)
    if not checkpoints:
        sys.exit(
            f"No checkpoints found in {metadata_files} with merge strategy '{merge}'"
        )
    return checkpoints


def next_checkpoint(metadata_file):
    """Return the first non-completed checkpoint row, or None."""
    with open(metadata_file) as f:
        for line in f:
            row = json.loads(line)
            if not row.get("completed"):
                print(f"Next checkpoint: {row['checkpoint_path']} "
                      f"(threshold {row.get('threshold_value')}, "
                      f"type {row.get('threshold_type')})")
                return row
    return None


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


def _claim_row(metadata_file, predicate):
    row = _update_row(metadata_file, predicate, {"claimed": True})
    if row:
        print(f"Claimed checkpoint: {row['checkpoint_path']} "
              f"(threshold {row.get('threshold_value')}, "
              f"type {row.get('threshold_type')})")
    return row


def claim_next_checkpoint(metadata_file):
    """Pick the next unclaimed+uncompleted checkpoint and mark it claimed.

    No file locking: race window is near-simultaneous sbatch starts; worst case
    is one checkpoint trained twice. Fine for our use case.
    """
    return _claim_row(
        metadata_file,
        lambda r: not r.get("claimed") and not r.get("completed"),
    )


def claim_checkpoint(metadata_file, checkpoint_path):
    """Claim the row for checkpoint_path; None when absent, claimed, or completed."""
    return _claim_row(
        metadata_file,
        lambda r: (r["checkpoint_path"] == checkpoint_path
                   and not r.get("claimed") and not r.get("completed")),
    )


def mark_completed(metadata_file, checkpoint_path):
    """Mark a checkpoint as completed in the metadata file."""
    _update_row(
        metadata_file,
        lambda r: r["checkpoint_path"] == checkpoint_path,
        {"completed": True},
    )


def record_wandb_run_id(metadata_file, checkpoint_path, wandb_run_id):
    """Persist the active wandb run id onto the row for ``checkpoint_path`` so
    downstream stages can resume the same run. No-op when wandb_run_id is empty."""
    if not wandb_run_id:
        return
    _update_row(
        metadata_file,
        lambda r: r["checkpoint_path"] == checkpoint_path,
        {"wandb_run_id": wandb_run_id},
    )


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
