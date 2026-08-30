# ABOUTME: Tests for scripts/eval_calibration_report.py: the majority-voting section built
# ABOUTME: from a model's greedy and maj256 calibration cells.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_calibration_report as report  # noqa: E402


def _cell(**benchmarks):
    return {"benchmarks": benchmarks}


def test_majority_section_lines_up_greedy_pass_and_maj():
    arms = {
        "ours_greedy": _cell(math500={"pass_at_1": 0.558}),
        "ours_maj256": _cell(
            math500={"pass_at_1": 0.555, "pass_at_256": 0.93, "maj_at_4": 0.61,
                     "maj_at_16": 0.64, "maj_at_64": 0.662, "maj_at_256": 0.668},
            amc={"pass_at_1": 0.254, "pass_at_64": 0.75, "maj_at_4": 0.275,
                 "maj_at_16": 0.35, "maj_at_64": 0.45},
        ),
    }
    lines = report.render_majority("l8b-64h-100", arms)
    text = "\n".join(lines)
    assert "ours_maj256" in text
    math_row = next(l for l in lines if l.startswith("math500"))
    assert math_row.split() == ["math500", "55.8", "55.5", "61.0", "64.0", "66.2", "66.8", "93.0"]
    amc_row = next(l for l in lines if l.startswith("amc"))
    assert amc_row.split() == ["amc", "--", "25.4", "27.5", "35.0", "45.0", "--", "75.0"]


def test_majority_section_uses_theirs_greedy_for_theirs_arm():
    arms = {
        "theirs_greedy": _cell(math500={"pass_at_1": 0.677}),
        "theirs_maj256": _cell(math500={"pass_at_1": 0.677, "pass_at_256": 0.956,
                                        "maj_at_256": 0.75}),
    }
    math_row = next(l for l in report.render_majority("openmath2", arms)
                    if l.startswith("math500"))
    assert math_row.split()[:2] == ["math500", "67.7"]


def test_majority_section_absent_without_maj_arm():
    assert report.render_majority("x", {"ours_greedy": _cell(math500={"pass_at_1": 0.5})}) == []
