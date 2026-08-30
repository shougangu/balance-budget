# ABOUTME: Regrades every eval point of the budget grid for one math benchmark from its W&B
# ABOUTME: generation table, reporting pass@1 under the production grader and accept-either.

"""Rescore the grid's generations for one benchmark under every grader.

Reads the eval-trajectories CSV for (model, run_id, global_step), pulls each
run's `raw_generations/<benchmark>/step_<N>` table, and writes one row per
eval point:

  run, project, tag, global_step, total_minutes,
  pass_at_1_production      the grader the benchmark logged with before the
                            accept-either verifier (numeric for GSM8K,
                            math-verify for MATH-500/AMC)
  pass_at_1_accept_either   tuning.evaluation.math_scoring.is_correct
  pass_at_1_union           correct under any grader, including a re-parsed
                            last \\boxed{}
  only_numeric_rate         responses only the numeric grader accepts
  only_math_verify_rate     responses only math-verify accepts
  boxed_wrong_numeric_right_rate   responses whose \\boxed{} disagrees with
                            the reference yet the numeric grader accepts them

Finished points are appended to a JSONL cache so an interrupted run resumes.

Usage:
    python scripts/regrade_grid.py --benchmark gsm8k --trajectories <csv> --out <csv> [--workers 6]
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.budget_frontier import LINEAGES
from scripts.regrade_generations import extract_last_boxed, grade_all, load_table, references_for

BENCHMARKS = ("gsm8k", "math500", "amc")
GRADERS = ("production", "math_verify", "gsm8k_numeric", "accept_either", "union")
FIELDS = ["run", "project", "tag", "global_step", "total_minutes",
          *(f"pass_at_1_{grader}" for grader in GRADERS),
          "only_numeric_rate", "only_math_verify_rate", "boxed_wrong_numeric_right_rate"]


def grade_rows(rows, benchmark: str) -> dict:
    """Aggregate (reference, responses) pairs into the per-eval-point metrics."""
    per_prompt = {grader: [] for grader in GRADERS}
    only_numeric = only_math_verify = boxed_wrong_numeric_right = total = 0
    for reference, responses in rows:
        grades = [grade_all(r, reference, benchmark) for r in responses]
        for key in per_prompt:
            per_prompt[key].append(sum(g[key] for g in grades) / len(grades))
        for response, g in zip(responses, grades):
            total += 1
            numeric, math_verify = g["gsm8k_numeric"], g["math_verify"]
            only_numeric += numeric and not math_verify
            only_math_verify += math_verify and not numeric
            boxed_wrong_numeric_right += (
                numeric and not math_verify and extract_last_boxed(response) is not None
            )
    return {
        **{f"pass_at_1_{k}": 100 * sum(v) / len(v) for k, v in per_prompt.items()},
        "only_numeric_rate": 100 * only_numeric / total,
        "only_math_verify_rate": 100 * only_math_verify / total,
        "boxed_wrong_numeric_right_rate": 100 * boxed_wrong_numeric_right / total,
    }


def grade_point(project: str, run_id: str, step: int, benchmark: str) -> dict:
    _, columns, rows = load_table(project, run_id, benchmark, step)
    references = references_for(benchmark)
    prompt_col, responses_col = columns.index("prompt"), columns.index("responses")
    return grade_rows(
        [(references[row[prompt_col]], json.loads(row[responses_col])) for row in rows],
        benchmark,
    )


def load_points(trajectories: str) -> list[dict]:
    points = {}
    with open(trajectories) as fh:
        for row in csv.DictReader(fh):
            lineage = LINEAGES[row["model"]]
            key = (row["run_id"], int(row["global_step"]))
            points.setdefault(key, {
                "run": row["run_id"], "project": lineage.project, "tag": lineage.tag,
                "global_step": int(row["global_step"]),
                "total_minutes": float(row["total_minutes"]),
            })
    return list(points.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    cache_path = args.out + ".jsonl"
    done = {}
    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            for line in fh:
                if not line.startswith("{"):
                    continue  # a killed writer can leave NUL-filled lines behind
                rec = json.loads(line)
                done[(rec["run"], rec["global_step"])] = rec

    points = load_points(args.trajectories)
    todo = [p for p in points if (p["run"], p["global_step"]) not in done]
    print(f"[regrade-grid] {args.benchmark}: {len(points)} points, {len(done)} cached, "
          f"{len(todo)} to grade")

    with ProcessPoolExecutor(max_workers=args.workers) as pool, open(cache_path, "a") as cache:
        futures = {pool.submit(grade_point, p["project"], p["run"], p["global_step"],
                               args.benchmark): p
                   for p in todo}
        for future in as_completed(futures):
            point = futures[future]
            try:
                rec = {**point, **future.result()}
            except Exception as exc:
                print(f"[regrade-grid] FAILED {point['run']} step {point['global_step']}: {exc!r}")
                continue
            done[(rec["run"], rec["global_step"])] = rec
            cache.write(json.dumps(rec) + "\n")
            cache.flush()
            print(f"[regrade-grid] {rec['run']} step {rec['global_step']}: "
                  f"{rec['pass_at_1_production']:.2f} -> {rec['pass_at_1_accept_either']:.2f} "
                  f"| union {rec['pass_at_1_union']:.2f}")

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for rec in sorted(done.values(), key=lambda r: (r["project"], r["run"], r["global_step"])):
            writer.writerow({k: rec[k] for k in FIELDS})
    print(f"[regrade-grid] {len(done)} rows -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
