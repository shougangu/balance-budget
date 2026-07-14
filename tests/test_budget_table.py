# ABOUTME: Tests for the budget_table script that builds compute-budget vs SFT-fraction tables.
# ABOUTME: Covers grid resolution, cell rendering, and the cross-benchmark average table.

import pytest

from scripts.budget_table import (
    AVERAGE_LABEL,
    BudgetRun,
    EvalEvent,
    STATUS_RUNNING,
    STATUS_VALUE,
    _render_cell,
    add_average_cells,
    build_grid,
    filter_runs_by_tag,
    format_id_csv,
    format_id_table,
    match_fraction_budget,
)


def _ev(tm, **vals):
    return EvalEvent(total_minutes=tm, values=vals)


class TestMatchFractionBudget:
    def test_maps_offset_to_cell(self):
        # 4h of SFT (240 min) sits at fraction*budget = 4 -> 25% @ 16h.
        assert match_fraction_budget(240.0) == (0.25, 16)

    def test_returns_none_when_offset_matches_no_cell(self):
        assert match_fraction_budget(7.0) is None


class TestBuildGrid:
    def test_places_sft_and_from_base_grpo(self):
        sft = BudgetRun("sft", ["sft"], "finished", "2026-01-01", 0.0,
                        [_ev(4 * 60 + 3, gsm8k=0.60, math500=0.34, amc=0.20)], 4 * 60 + 3)
        base = BudgetRun("base", ["grpo"], "finished", "2026-01-02", 0.0,
                         [_ev(4 * 60 + 1, gsm8k=0.55, math500=0.30, amc=0.18)], 4 * 60 + 1)
        grid = build_grid([sft, base])
        assert grid[("gsm8k", 4, 1.0)].value == pytest.approx(0.60)
        assert grid[("gsm8k", 4, 0.0)].value == pytest.approx(0.55)

    def test_carries_wandb_id_into_cells(self):
        run = BudgetRun(
            "sft", ["sft"], "finished", "2026-01-01", 0.0,
            [_ev(4 * 60 + 3, gsm8k=0.60)], 4 * 60 + 3,
            run_id="abc123",
        )
        grid = build_grid([run])
        assert grid[("gsm8k", 4, 1.0)].run_id == "abc123"


class TestIdTables:
    def _grid(self):
        run = BudgetRun(
            "sft", ["sft"], "finished", "2026-01-01", 0.0,
            [_ev(4 * 60 + 3, gsm8k=0.60)], 4 * 60 + 3,
            run_id="abc123",
        )
        return build_grid([run], budgets=[4])

    def test_markdown_contains_only_run_id_in_populated_cell(self):
        table = format_id_table(
            self._grid(), "gsm8k", budgets=[4], fractions=[0.0, 1.0]
        )
        assert "| 4 Hours |  | abc123 |" in table
        assert "60.0%" not in table

    def test_csv_contains_only_run_id_in_populated_cell(self):
        table = format_id_csv(
            self._grid(), "gsm8k", budgets=[4], fractions=[0.0, 1.0]
        )
        assert table.splitlines()[-1] == "4 Hours,,abc123"
        assert "60.0%" not in table


class TestAverageCells:
    def _sft_grid(self):
        sft = BudgetRun("sft", ["sft"], "finished", "2026-01-01", 0.0,
                        [_ev(4 * 60 + 3, gsm8k=0.60, math500=0.30, amc=0.20)], 4 * 60 + 3)
        return build_grid([sft])

    def test_averages_readings_across_benchmarks(self):
        grid = self._sft_grid()
        add_average_cells(grid)
        cell = grid[(AVERAGE_LABEL, 4, 1.0)]
        assert cell.status == STATUS_VALUE
        assert cell.value == pytest.approx((0.60 + 0.30 + 0.20) / 3)
        assert _render_cell(cell) == "36.7%"

    def test_carries_marker_when_no_readings(self):
        live = BudgetRun("live", ["grpo"], "running", "2026-01-01", 0.0,
                         [_ev(60.0, gsm8k=0.4, math500=0.3, amc=0.2)], 60.0)
        grid = build_grid([live])
        add_average_cells(grid)
        cell = grid[(AVERAGE_LABEL, 4, 0.0)]
        assert cell.value is None
        assert cell.status == STATUS_RUNNING
        assert _render_cell(cell) == "o"

    def test_no_average_cell_when_benchmarks_absent(self):
        grid = {}
        add_average_cells(grid)
        assert grid == {}


class TestFilterRunsByTag:
    def _run(self, name, tags):
        return BudgetRun(name, tags, "finished", "2026-01-01", 0.0, [], 0.0)

    def test_keeps_only_tagged_runs(self):
        runs = [
            self._run("a", ["sft", "try3"]),
            self._run("b", ["sft"]),
            self._run("c", ["grpo", "try3"]),
        ]
        kept = filter_runs_by_tag(runs, "try3")
        assert [r.name for r in kept] == ["a", "c"]

    def test_empty_tag_returns_all(self):
        runs = [self._run("a", ["sft"]), self._run("b", ["grpo"])]
        assert filter_runs_by_tag(runs, None) == runs
