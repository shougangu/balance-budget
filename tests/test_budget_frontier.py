# ABOUTME: Tests for the SFT/RL budget frontier figure: FLOP accounting, worker tag
# ABOUTME: decoding, token lookup at eval steps, Pareto frontier and first-touch labels.

import pytest

from scripts.budget_frontier import (
    Point,
    first_frontier_touch,
    pareto_frontier,
    parse_worker_tags,
    rl_flops,
    sft_flops,
    tokens_at_step,
)


def test_sft_flops_is_six_n_tokens():
    assert sft_flops(n_params=10, examples=3, mean_tokens=4.0) == 6 * 10 * 12


def test_rl_flops_counts_rollout_reference_and_update_passes():
    assert rl_flops(n_params=10, rollout_tokens=5) == 10 * 10 * 5
    assert rl_flops(n_params=10, rollout_tokens=5, reference_model=False) == 8 * 10 * 5


def test_parse_worker_tags_reads_mark_and_sft_examples():
    assert parse_worker_tags(["60.0", "91248", "padfree", "redo"]) == (60.0, 91248)
    assert parse_worker_tags(["1105200", "720.0", "padfree", "redo"]) == (720.0, 1105200)


def test_parse_worker_tags_rejects_sft_and_untagged_runs():
    assert parse_worker_tags(["redo", "sft"]) is None
    assert parse_worker_tags(["0", "zero-sft"]) is None


def test_tokens_at_step_uses_latest_logged_step_at_or_before():
    rows = [(1, 100), (2, 250), (5, 700)]
    assert tokens_at_step(rows, 0) == 0
    assert tokens_at_step(rows, 2) == 250
    assert tokens_at_step(rows, 4) == 250
    assert tokens_at_step(rows, 9) == 700


def _pt(run, x, y, rl=0.0):
    return Point(
        run_id=run, mark_minutes=0.0, total_minutes=0.0,
        sft_flops=x - rl, rl_flops=rl, value=y,
    )


def test_pareto_frontier_is_upper_envelope_in_compute_order():
    points = [_pt("a", 1, 0.3), _pt("b", 2, 0.2), _pt("c", 3, 0.5), _pt("d", 4, 0.5), _pt("e", 5, 0.6)]
    frontier = pareto_frontier(points)
    assert [p.run_id for p in frontier] == ["a", "c", "e"]


def test_pareto_frontier_breaks_ties_at_equal_compute_by_value():
    points = [_pt("lo", 2, 0.2), _pt("hi", 2, 0.4)]
    assert [p.run_id for p in pareto_frontier(points)] == ["hi"]


def test_first_frontier_touch_reports_rl_fraction_of_first_frontier_point():
    a1, a2, a3 = _pt("a", 2, 0.2, rl=1), _pt("a", 4, 0.5, rl=3), _pt("a", 6, 0.6, rl=5)
    b1 = _pt("b", 5, 0.45, rl=0)
    frontier = pareto_frontier([a1, a2, a3, b1])
    touches = first_frontier_touch(frontier)
    assert touches["a"] is a1
    assert touches["a"].rl_fraction == pytest.approx(0.5)
    assert "b" not in touches


def test_point_rl_fraction_and_total():
    p = _pt("a", 10, 0.1, rl=4)
    assert p.total_flops == 10
    assert p.rl_fraction == pytest.approx(0.4)
