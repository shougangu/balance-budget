"""Run IFeval inference+evaluation repeatedly and log extracted metrics."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
IFEVAL_METRIC_BLOCK_RE = re.compile(
    rf"(?P<path>[^\n]*eval_results_(?P<mode>strict|loose)\.jsonl)\s+Accuracy Scores:\s*"
    rf"prompt-level:\s*(?P<prompt>{FLOAT_RE})\s*"
    rf"instruction-level:\s*(?P<instruction>{FLOAT_RE})",
    re.IGNORECASE,
)


def parse_ifeval_stdout(stdout_text: str) -> Dict[str, Dict[str, float]]:
    """Extract strict/loose prompt+instruction scores from eval stdout."""
    parsed: Dict[str, Dict[str, float]] = {}
    for match in IFEVAL_METRIC_BLOCK_RE.finditer(stdout_text):
        mode = match.group("mode").lower()
        parsed[mode] = {
            "prompt_level": float(match.group("prompt")),
            "instruction_level": float(match.group("instruction")),
        }

    missing_modes = [mode for mode in ("strict", "loose") if mode not in parsed]
    if missing_modes:
        snippet = stdout_text[:500].replace("\n", "\\n")
        raise ValueError(
            f"Unable to parse IFeval metrics for {missing_modes} from stdout. "
            f"Output snippet: {snippet}"
        )

    return parsed


def _run_ifeval_evaluation_capture_stdout(model_name: str) -> str:
    """Run IFeval evaluation and return combined stdout/stderr text."""
    from tuning.config import IFEVAL_OUTPUTS_DIR, RESPONSES_FILENAME

    output_path = f"{IFEVAL_OUTPUTS_DIR}/{model_name}/{RESPONSES_FILENAME}"
    eval_cmd = [
        "python3",
        "-m",
        "instruction_following_eval.evaluation_main",
        "--input_data=./instruction_following_eval/data/input_data.jsonl",
        f"--input_response_data={output_path}",
        f"--output_dir={IFEVAL_OUTPUTS_DIR}/{model_name}/",
    ]

    result = subprocess.run(eval_cmd, check=False, capture_output=True, text=True)
    combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)

    # Preserve full command output visibility while still capturing for parsing.
    if combined_output:
        print(combined_output, end="" if combined_output.endswith("\n") else "\n")

    if result.returncode != 0:
        raise RuntimeError(
            f"IFeval evaluation failed for {model_name} with return code "
            f"{result.returncode}."
        )

    return combined_output


def _run_ifeval_inference(model_name: str) -> None:
    from tuning.inference.ifeval_inference import run_inference_ifeval

    run_inference_ifeval(model_name)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_log_file() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / f"ifeval_evaluation_results_{ts}.jsonl"


def _summarize_model(model_name: str, run_rows: List[Dict[str, float]]) -> Dict[str, float]:
    strict_prompt_values = [row["strict_prompt_level"] for row in run_rows]
    strict_instruction_values = [row["strict_instruction_level"] for row in run_rows]
    loose_prompt_values = [row["loose_prompt_level"] for row in run_rows]
    loose_instruction_values = [row["loose_instruction_level"] for row in run_rows]

    def _safe_std(values: List[float]) -> float:
        return pstdev(values) if len(values) > 1 else 0.0

    return {
        "type": "model_summary",
        "timestamp_utc": _timestamp_utc(),
        "model": model_name,
        "runs": len(run_rows),
        "strict_prompt_level_mean": mean(strict_prompt_values),
        "strict_prompt_level_std": _safe_std(strict_prompt_values),
        "strict_instruction_level_mean": mean(strict_instruction_values),
        "strict_instruction_level_std": _safe_std(strict_instruction_values),
        "loose_prompt_level_mean": mean(loose_prompt_values),
        "loose_prompt_level_std": _safe_std(loose_prompt_values),
        "loose_instruction_level_mean": mean(loose_instruction_values),
        "loose_instruction_level_std": _safe_std(loose_instruction_values),
    }


def _write_jsonl(log_file: Path, rows: List[Dict[str, object]]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def run_harness(models: List[str], runs: int, log_file: Path) -> Path:
    all_rows: List[Dict[str, object]] = [
        {
            "type": "run_metadata",
            "timestamp_utc": _timestamp_utc(),
            "models": models,
            "runs_per_model": runs,
        }
    ]

    for model_name in models:
        model_rows: List[Dict[str, float]] = []
        for run_idx in range(1, runs + 1):
            print(f"Running ifeval for {model_name} (run {run_idx}/{runs})")
            _run_ifeval_inference(model_name)

            print(f"Running IFeval evaluation for {model_name} (run {run_idx}/{runs})")
            stdout_text = _run_ifeval_evaluation_capture_stdout(model_name)
            parsed = parse_ifeval_stdout(stdout_text)

            run_row: Dict[str, float] = {
                "strict_prompt_level": parsed["strict"]["prompt_level"],
                "strict_instruction_level": parsed["strict"]["instruction_level"],
                "loose_prompt_level": parsed["loose"]["prompt_level"],
                "loose_instruction_level": parsed["loose"]["instruction_level"],
            }

            all_rows.append(
                {
                    "type": "run_result",
                    "timestamp_utc": _timestamp_utc(),
                    "model": model_name,
                    "run_index": run_idx,
                    **run_row,
                }
            )
            model_rows.append(run_row)

        summary = _summarize_model(model_name, model_rows)
        all_rows.append(summary)
        print(
            f"Summary for {model_name}: "
            f"strict(prompt={summary['strict_prompt_level_mean']:.6f}, "
            f"instruction={summary['strict_instruction_level_mean']:.6f}) | "
            f"loose(prompt={summary['loose_prompt_level_mean']:.6f}, "
            f"instruction={summary['loose_instruction_level_mean']:.6f})"
        )

    _write_jsonl(log_file, all_rows)
    print(f"Wrote results to {log_file}")
    return log_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run IFeval inference+evaluation repeatedly for a list of models and log "
            "strict/loose prompt+instruction metrics."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model names.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of inference+evaluation repetitions per model (default: 3).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL output path. Defaults to "
            "evaluation/ifeval_evaluation_results_<timestamp>.jsonl."
        ),
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    log_file = args.log_file or _default_log_file()
    run_harness(models=args.models, runs=args.runs, log_file=log_file)


if __name__ == "__main__":
    main()
