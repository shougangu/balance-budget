# ABOUTME: Regrades W&B-logged eval generations under alternative graders to measure
# ABOUTME: how much of our reported math accuracy is lost to answer extraction.

"""Audit our math grader against the generations it already scored.

Pulls a `raw_generations/<benchmark>/step_<N>` table off a W&B run and rescores
every response with:

  repo           the scorer the run used (math-verify, then #### fallback)
  math_verify    math-verify on the raw response only
  gsm8k_numeric  the lm-eval numeric path (strict ####, then last number)
  boxed_only     math-verify restricted to the last \\boxed{...} span
  union          correct under any of the above

It also reports how often a response carries no extractable final answer at all,
which separates "the model was wrong" from "we could not read its answer".

Usage:
    python scripts/regrade_generations.py --project '[1]math-l8b' --run 0do9os0f \
        --benchmark math500 --step best
"""

import argparse
import json
import os
import re

from tuning.evaluation.gsm8k_scoring import is_correct as gsm8k_is_correct
from tuning.evaluation.math500_scoring import is_correct as repo_is_correct
from tuning.evaluation.math_scoring import is_correct as accept_either_is_correct

BOXED_PATTERN = re.compile(r"\\boxed\s*{")
HASH_PATTERN = re.compile(r"####\s*\S")


def has_boxed(response: str) -> bool:
    """True when the response contains a \\boxed{...} span."""
    return bool(BOXED_PATTERN.search(response or ""))


def has_hash_answer(response: str) -> bool:
    """True when the response contains a GSM8K-style '#### <answer>' line."""
    return bool(HASH_PATTERN.search(response or ""))


def extract_last_boxed(response: str) -> str | None:
    """Return the contents of the last \\boxed{...}, brace-matched."""
    matches = list(BOXED_PATTERN.finditer(response or ""))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(response)):
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
            if depth == 0:
                return response[start:i]
    return None


def _math_verify(candidate: str, reference: str) -> bool:
    from math_verify import LatexExtractionConfig, parse, verify
    from math_verify.errors import TimeoutException

    try:
        gold = parse(rf"\boxed{{{reference}}}", extraction_config=[LatexExtractionConfig()])
        pred = parse(candidate, extraction_config=[LatexExtractionConfig()])
        return bool(gold and pred and verify(gold, pred))
    except (TimeoutException, Exception):
        return False


def grade_all(response: str, reference: str, benchmark: str = "math500") -> dict:
    """Grade one response under every scorer, plus their union.

    `production` is the scorer each benchmark used before the accept-either
    verifier: the lm-eval numeric path for GSM8K, math-verify for MATH-500 and
    AMC. `accept_either` is `tuning.evaluation.math_scoring.is_correct`, what
    every math EvalStrategy logs now. `union` is broader still: it also credits
    a re-parsed last \\boxed{} that math-verify missed on the raw response.
    """
    grades = {
        "production": bool(
            gsm8k_is_correct(response, reference) if benchmark == "gsm8k"
            else repo_is_correct(response, reference)
        ),
        "math_verify": _math_verify(response, reference),
        "gsm8k_numeric": bool(gsm8k_is_correct(response, reference)),
        "accept_either": bool(accept_either_is_correct(response, reference)),
    }
    boxed = extract_last_boxed(response)
    grades["boxed_only"] = (
        _math_verify(rf"\boxed{{{boxed}}}", reference) if boxed is not None else False
    )
    grades["union"] = any(grades.values())
    return grades


def load_table(project: str, run_id: str, benchmark: str, step):
    import wandb

    api = wandb.Api()
    entity = api.default_entity
    run = api.run(f"{entity}/{project}/{run_id}")

    prefix = f"run-{run_id}-raw_generations{benchmark}step_"
    candidates = {}
    for artifact in run.logged_artifacts():
        if not artifact.name.startswith(prefix):
            continue
        step_num = int(artifact.name[len(prefix):].split("-")[0])
        candidates[step_num] = artifact

    if not candidates:
        raise SystemExit(f"No {benchmark} generation tables logged on run {run_id}")

    chosen_step = max(candidates) if step in (None, "best", "last") else int(step)
    artifact = candidates[chosen_step]
    print(f"[regrade] {run_id} {benchmark} step {chosen_step} "
          f"(available: {sorted(candidates)})")

    directory = artifact.download()
    table_file = next(
        os.path.join(root, name)
        for root, _, files in os.walk(directory)
        for name in files if name.endswith(".table.json")
    )
    with open(table_file) as fh:
        payload = json.load(fh)
    return chosen_step, payload["columns"], payload["data"]


def references_for(benchmark: str) -> dict:
    from tuning.data.test_dataset import (
        get_amc_test_dataset,
        get_gsm8k_test_dataset,
        get_math500_test_dataset,
    )

    loader = {
        "math500": get_math500_test_dataset,
        "gsm8k": get_gsm8k_test_dataset,
        "amc": get_amc_test_dataset,
    }[benchmark]
    dataset = loader()
    return dict(zip(dataset["prompt"], dataset["reference_answer"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--benchmark", default="math500", choices=["math500", "gsm8k", "amc"])
    parser.add_argument("--step", default="best")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    step, columns, rows = load_table(args.project, args.run, args.benchmark, args.step)
    prompt_col = columns.index("prompt")
    responses_col = columns.index("responses")
    references = references_for(args.benchmark)

    graders = ["production", "math_verify", "gsm8k_numeric", "boxed_only", "union"]
    per_prompt = {g: [] for g in graders}
    no_answer = []
    total_responses = 0

    for row in rows:
        prompt = row[prompt_col]
        responses = json.loads(row[responses_col])
        reference = references[prompt]
        grades = [grade_all(r, reference, args.benchmark) for r in responses]
        for grader in graders:
            per_prompt[grader].append(sum(g[grader] for g in grades) / len(grades))
        no_answer.append(
            sum(not (has_boxed(r) or has_hash_answer(r)) for r in responses) / len(responses)
        )
        total_responses += len(responses)

    report = {
        "project": args.project,
        "run": args.run,
        "benchmark": args.benchmark,
        "step": step,
        "num_prompts": len(rows),
        "num_responses": total_responses,
        "pass_at_1": {g: sum(v) / len(v) for g, v in per_prompt.items()},
        "no_extractable_answer_rate": sum(no_answer) / len(no_answer),
    }
    report["headroom_from_grading"] = (
        report["pass_at_1"]["union"] - report["pass_at_1"]["production"]
    )

    print(json.dumps(report, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"[regrade] report -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
