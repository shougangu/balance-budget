#!/bin/bash
# ABOUTME: Gate job for the calibration sweep: reads finished smoke reports and, only when
# ABOUTME: every one scores MATH-500 pass@1 above GATE_MIN, runs SWEEP_CMD to submit the full cells.
#SBATCH --job-name=cal-gate
#SBATCH -c 1
#SBATCH --mem=2G
#SBATCH --time=0:15:00
#SBATCH --output=outputs/eval_calibration/%j_gate.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python - "$@" <<'PY'
import json, os, sys
floor = float(os.environ.get("GATE_MIN", "0.4"))
for path in sys.argv[1:]:
    score = json.load(open(path))["benchmarks"]["math500"]["pass_at_1"]
    print(f"{path}: math500 pass@1 = {score:.3f}")
    if score < floor:
        sys.exit(f"gate failed: {path} below {floor}")
PY
eval "$SWEEP_CMD"
