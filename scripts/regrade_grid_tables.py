# ABOUTME: Renders the budget-split grid (budget x SFT fraction) per model from a regraded
# ABOUTME: CSV, showing the production score alongside the accept-either verifier's score.

"""Usage: python scripts/regrade_grid_tables.py --benchmark gsm8k --trajectories <csv> --regraded <csv>

Each cell is the last eval point at or before the cell budget (the row the
trajectories CSV assigns to that cell), printed as
`production -> accept_either -> union`.
"""

import argparse
import csv

FRACTIONS = [0, 25, 50, 75, 100]
BUDGETS = [4, 16, 64]
MODELS = ["l8b", "g12b", "l3b", "g4b"]


def final_cells(trajectories: str) -> dict:
    cells = {}
    with open(trajectories) as fh:
        for row in csv.DictReader(fh):
            key = (row["model"], int(row["budget_hours"]), int(row["sft_fraction"]))
            if key not in cells or float(row["total_minutes"]) > float(cells[key]["total_minutes"]):
                cells[key] = row
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--regraded", required=True)
    args = parser.parse_args()

    with open(args.regraded) as fh:
        regraded = {(r["run"], int(r["global_step"])): r for r in csv.DictReader(fh)}
    cells = final_cells(args.trajectories)

    for model in MODELS:
        print(f"\n### {model} / {args.benchmark} pass@1 — production -> accept_either -> union\n")
        print("| budget | " + " | ".join(f"{f}% SFT" for f in FRACTIONS) + " |")
        print("|---|" + "---|" * len(FRACTIONS))
        for budget in BUDGETS:
            out = []
            for frac in FRACTIONS:
                cell = cells.get((model, budget, frac))
                rec = regraded.get((cell["run_id"], int(cell["global_step"]))) if cell else None
                if rec is None:
                    out.append("--")
                    continue
                prod = float(rec["pass_at_1_production"])
                alt = float(rec["pass_at_1_accept_either"])
                union = float(rec["pass_at_1_union"])
                out.append(f"{prod:.1f} -> {alt:.1f} -> {union:.1f}")
            print(f"| {budget}h | " + " | ".join(out) + " |")


if __name__ == "__main__":
    main()
