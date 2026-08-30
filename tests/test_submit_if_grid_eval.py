# ABOUTME: Tests for scripts/submit_if_grid_eval.py: the 15 instruction-following grid
# ABOUTME: cells it re-evaluates and the sampling protocol it asks for.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import submit_if_grid_eval as sub  # noqa: E402


def test_every_budget_fraction_cell_has_a_checkpoint():
    assert set(sub.CELLS) == {
        (budget, fraction)
        for budget in (4, 16, 64)
        for fraction in (0, 25, 50, 75, 100)
    }
    for path in sub.CELLS.values():
        assert path.startswith("tuning/models/gemma3-12B_p@1-")


def test_each_budget_reads_its_own_mark_checkpoint():
    for (budget, _), path in sub.CELLS.items():
        assert f"_p@1-{budget * 60}m_" in path


def test_flags_score_ifeval_at_four_samples_and_ifbench_at_eight():
    flags = sub.cell_flags(4, 100)
    assert flags[flags.index("--n-samples") + 1] == "4"
    assert flags[flags.index("--ifbench-n-samples") + 1] == "8"
    assert flags[flags.index("--k-values") + 1:flags.index("--k-values") + 5] == [
        "1", "2", "4", "8",
    ]
    assert flags[flags.index("--benchmarks") + 1] == "ifeval,ifbench"


def test_flags_match_the_training_time_protocol():
    flags = sub.cell_flags(64, 25)
    assert flags[flags.index("--template") + 1] == "repo"
    assert flags[flags.index("--prompt-style") + 1] == "ours"
    assert flags[flags.index("--temperature") + 1] == "0.5"
    assert "--ifeval-strict" not in flags
    assert "--no-ifbench-strict" not in flags


def test_each_cell_writes_its_own_json():
    out = dict(sub.cell_outputs())
    assert len(set(out.values())) == 15
    assert out[(16, 50)].endswith("if_grid_n48/16h_50.json")
