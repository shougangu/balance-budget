# ABOUTME: Aggregates the calibration sweep into one table comparing published numbers,
# ABOUTME: the reference protocol reproduced locally, and our training-time protocol.

"""Turn the per-cell calibration JSONs into a readable comparison.

For each external checkpoint it lines up:

  published      the number the model's own paper / card reports
  theirs_greedy  that protocol reproduced here (native template, boxed
                 instruction, greedy) with our scorer
  theirs_sampled the same prompts at our temperature
  ours_native    our prompts and decoding on the model's own chat template
  ours_simple    our exact training-time protocol

so the published-to-ours gap splits into a grading component
(published - theirs_greedy), a decoding component (theirs_greedy -
theirs_sampled), a prompt component (theirs_sampled - ours_native) and a
template component (ours_native - ours_simple).

A *_maj256 cell adds a majority-voting section: greedy (from the matching
*_greedy cell), sampled pass@1, maj@{4,16,64,256} and pass@n from the same
generations.

Usage:
    python scripts/eval_calibration_report.py --dir outputs/eval_calibration
"""

import argparse
import glob
import json
import os

# Numbers as reported by each model's own paper or model card. Benchmarks differ
# from ours where noted: "MATH" is the full 5,000-problem test set, not MATH-500.
PUBLISHED = {
    "openmath2": {
        "source": "OpenMathInstruct-2 (arXiv:2410.01560), GPT-4o-judged, greedy",
        "math500": ("MATH (5k)", 67.8),
        "gsm8k": ("GSM8K", 91.7),
        "amc": ("AMC 2023", 40.0),
    },
    "llama31-8b-instruct": {
        "source": "Meta Llama 3.1 model card; AMC from OpenMathInstruct-2",
        "math500": ("MATH (5k), 0-shot CoT", 51.9),
        "gsm8k": ("GSM8K, 8-shot CoT", 84.5),
        "amc": ("AMC 2023", 22.5),
        "ifeval": ("IFEval", 80.4),
    },
    "llama32-3b-instruct": {
        "source": "Meta Llama 3.2 model card",
        "math500": ("MATH (5k), 0-shot CoT", 48.0),
        "gsm8k": ("GSM8K, 8-shot CoT", 77.7),
        "ifeval": ("IFEval", 77.4),
    },
    "tulu3-8b": {
        "source": "Tulu 3 (arXiv:2411.15124); IFBench from arXiv:2507.02833",
        "math500": ("MATH (5k), 4-shot CoT flex", 43.7),
        "gsm8k": ("GSM8K, 8-shot CoT", 87.6),
        "ifeval": ("IFEval, prompt-level loose", 82.4),
        "ifbench": ("IFBench", 28.9),
    },
    "gemma3-12b-it": {
        "source": "Gemma 3 technical report (arXiv:2503.19786) Table 18",
        "math500": ("MATH (5k), 0-shot", 83.8),
        "gsm8k": ("GSM8K, 0-shot CoT", 94.4),
        "ifeval": ("IFEval", 88.9),
    },
    "gemma3-4b-it": {
        "source": "Gemma 3 technical report (arXiv:2503.19786) Table 18",
        "math500": ("MATH (5k), 0-shot", 75.6),
        "gsm8k": ("GSM8K, 0-shot CoT", 89.2),
        "ifeval": ("IFEval", 90.2),
    },
}

MATH_ARM_ORDER = ["theirs_greedy", "theirs_sampled", "ours_native", "ours_simple"]
IF_ARM_ORDER = ["if_native", "if_ours"]
MATH_BENCHMARKS = ["math500", "gsm8k", "amc"]
IF_BENCHMARKS = ["ifeval", "ifbench"]


def load_cells(directory):
    cells = {}
    for path in sorted(glob.glob(os.path.join(directory, "*__*.json"))):
        name = os.path.basename(path)[: -len(".json")]
        model, _, arm = name.partition("__")
        with open(path) as fh:
            cells.setdefault(model, {})[arm] = json.load(fh)
    return cells


def pct(value):
    return "  --  " if value is None else f"{100 * value:6.1f}"


def score(cell, benchmark, key="pass_at_1"):
    if cell is None:
        return None
    return cell.get("benchmarks", {}).get(benchmark, {}).get(key)


def render_model(model, arms):
    lines = [f"\n{'=' * 78}", f"{model}", f"{'=' * 78}"]
    published = PUBLISHED.get(model, {})
    if published.get("source"):
        lines.append(f"published source: {published['source']}")

    for benchmarks, arm_order in ((MATH_BENCHMARKS, MATH_ARM_ORDER),
                                  (IF_BENCHMARKS, IF_ARM_ORDER)):
        present = [b for b in benchmarks
                   if any(score(arms.get(a), b) is not None for a in arm_order)]
        if not present:
            continue

        header = f"{'benchmark':<12}{'published':>11}" + "".join(
            f"{a:>16}" for a in arm_order)
        lines.append("")
        lines.append(header)
        lines.append("-" * len(header))
        for benchmark in present:
            label, reported = published.get(benchmark, (benchmark, None))
            row = f"{benchmark:<12}" + (
                f"{reported:>11.1f}" if reported is not None else f"{'--':>11}")
            for arm in arm_order:
                row += f"{pct(score(arms.get(arm), benchmark)):>16}"
            lines.append(row)
            if reported is not None:
                lines.append(f"{'  (' + label + ')':<12}")

        if arm_order is MATH_ARM_ORDER:
            lines.append("")
            lines.append("gap decomposition (percentage points)")
            head = (f"{'benchmark':<12}{'grading':>10}{'decoding':>10}"
                    f"{'prompt':>10}{'template':>10}{'total':>10}")
            lines.append(head)
            lines.append("-" * len(head))
            for benchmark in present:
                _, reported = published.get(benchmark, (benchmark, None))
                values = {a: score(arms.get(a), benchmark) for a in MATH_ARM_ORDER}
                lines.append(_gap_row(benchmark, reported, values))
    lines.extend(render_majority(model, arms))
    return lines


MAJ_K = (4, 16, 64, 256)


def render_majority(model, arms):
    """Lines for the majority-voting section, or [] when the model has no maj256 cell."""
    lines = []
    for arm in sorted(a for a in arms if a.endswith("_maj256")):
        greedy = arms.get(arm.replace("_maj256", "_greedy"))
        cell = arms[arm]
        present = [b for b in MATH_BENCHMARKS if score(cell, b) is not None]
        if not present:
            continue
        header = (f"{'benchmark':<12}{'greedy':>8}{'pass@1':>8}"
                  + "".join(f"{'maj@' + str(k):>9}" for k in MAJ_K) + f"{'pass@n':>9}")
        lines += ["", f"majority voting ({arm}; n = samples per prompt in that cell)",
                  header, "-" * len(header)]
        for benchmark in present:
            n = max(int(k[len("pass_at_"):]) for k in cell["benchmarks"][benchmark]
                    if k.startswith("pass_at_") and k[len("pass_at_"):].isdigit())
            row = f"{benchmark:<12}{pct(score(greedy, benchmark)).strip():>8}"
            row += f"{pct(score(cell, benchmark)).strip():>8}"
            for k in MAJ_K:
                row += f"{pct(score(cell, benchmark, f'maj_at_{k}')).strip():>9}"
            row += f"{pct(score(cell, benchmark, f'pass_at_{n}')).strip():>9}"
            lines.append(row)
    return lines


def _delta(a, b):
    if a is None or b is None:
        return "     --"
    return f"{100 * (a - b):+7.1f}"


def _gap_row(benchmark, reported, values):
    reported_frac = None if reported is None else reported / 100.0
    row = f"{benchmark:<12}"
    row += f"{_delta(reported_frac, values['theirs_greedy']):>10}"
    row += f"{_delta(values['theirs_greedy'], values['theirs_sampled']):>10}"
    row += f"{_delta(values['theirs_sampled'], values['ours_native']):>10}"
    row += f"{_delta(values['ours_native'], values['ours_simple']):>10}"
    row += f"{_delta(reported_frac, values['ours_simple']):>10}"
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="outputs/eval_calibration")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cells = load_cells(args.dir)
    if not cells:
        raise SystemExit(f"No calibration cells found under {args.dir}")

    lines = []
    for model in PUBLISHED:
        if model in cells:
            lines.extend(render_model(model, cells[model]))
    for model in cells:
        if model not in PUBLISHED:
            lines.extend(render_model(model, cells[model]))

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"\nreport -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
