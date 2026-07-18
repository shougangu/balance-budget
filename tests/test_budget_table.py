# ABOUTME: Tests for the budget_table script that builds compute-budget vs SFT-fraction tables.
# ABOUTME: Covers grid resolution, cell rendering, and the cross-benchmark average table.

import pytest

from scripts.budget_table import (
    AVERAGE_LABEL,
    BudgetRun,
    Cell,
    EvalEvent,
    MetricSource,
    STATUS_RUNNING,
    STATUS_VALUE,
    TEST_TIME_REWARD_LABEL,
    TEST_TIME_REWARD_KEY,
    _fetch_events,
    _render_cell,
    add_average_cells,
    build_grid,
    filter_runs_by_tag,
    format_id_csv,
    format_id_table,
    format_combined_csv,
    match_fraction_budget,
    pass_at_1_standard_deviation,
    read_max_through_endpoint,
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

    def test_max_pass_uses_best_value_through_budget(self):
        run = BudgetRun(
            "base", ["grpo"], "finished", "2026-01-01", 0.0,
            [
                _ev(60, gsm8k=0.70),
                _ev(4 * 60 + 2, gsm8k=0.60),
                _ev(5 * 60, gsm8k=0.90),
            ],
            5 * 60,
        )

        regular = build_grid([run], budgets=[4])
        maximum = build_grid([run], budgets=[4], max_pass=True)

        assert regular[("gsm8k", 4, 0.0)].value == pytest.approx(0.60)
        assert maximum[("gsm8k", 4, 0.0)].value == pytest.approx(0.70)

    def test_reports_test_time_reward_for_grpo(self):
        run = BudgetRun(
            "base", ["grpo"], "finished", "2026-01-01", 0.0,
            [
                _ev(60, **{TEST_TIME_REWARD_LABEL: 0.75}),
                _ev(4 * 60 + 2, **{TEST_TIME_REWARD_LABEL: 0.65}),
            ],
            4 * 60 + 2,
        )

        regular = build_grid([run], budgets=[4])
        maximum = build_grid([run], budgets=[4], max_pass=True)

        assert regular[(TEST_TIME_REWARD_LABEL, 4, 0.0)].value == pytest.approx(0.65)
        assert maximum[(TEST_TIME_REWARD_LABEL, 4, 0.0)].value == pytest.approx(0.75)

    def test_does_not_add_reward_marker_for_sft(self):
        run = BudgetRun("sft", ["sft"], "running", "2026-01-01", 0.0, [], 60.0)
        grid = build_grid([run], budgets=[4])
        assert (TEST_TIME_REWARD_LABEL, 4, 1.0) not in grid


class TestVarianceEventSelection:
    def test_max_uses_variance_source_from_the_maximizing_event(self):
        early_source = MetricSource(global_step=10)
        boundary_source = MetricSource(global_step=20)
        run = BudgetRun(
            "base", ["grpo"], "finished", "2026-01-01", 0.0,
            [
                EvalEvent(60, {"gsm8k": 0.70}, {"gsm8k": early_source}),
                EvalEvent(242, {"gsm8k": 0.60}, {"gsm8k": boundary_source}),
            ],
            242,
        )

        regular = build_grid([run], budgets=[4])[("gsm8k", 4, 0.0)]
        maximum = build_grid([run], budgets=[4], max_pass=True)[("gsm8k", 4, 0.0)]

        assert regular.variance_source is boundary_source
        assert maximum.variance_source is early_source


class TestVarianceMath:
    @staticmethod
    def _pass_table():
        return {
            "columns": ["prompt", "per_response_correct"],
            "data": [
                ["q1", "[true, false, true, false]"],
                ["q2", "[true, true, true, true]"],
            ],
        }

    def test_pass_at_1_uses_sum_variance_rule(self):
        standard_deviation = pass_at_1_standard_deviation(self._pass_table())

        # q1 contributes .5(1-.5)/4=.0625; q2 contributes zero;
        # Var(mean)=(.0625+0)/2^2=.015625.
        assert standard_deviation == pytest.approx(0.125)

    def test_combined_csv_joins_mean_and_standard_deviation(self):
        cell = Cell("gsm8k", 4, 0.0, 0.602, standard_deviation=0.004)
        grid = {("gsm8k", 4, 0.0): cell}
        table = format_combined_csv(grid, "gsm8k", budgets=[4], fractions=[0.0])
        assert table.splitlines()[-1] == "4 Hours,60.2% ± 0.4%"

    def test_combined_csv_shows_bare_mean_when_no_standard_deviation(self):
        cell = Cell("gsm8k", 4, 0.0, 0.602)
        grid = {("gsm8k", 4, 0.0): cell}
        table = format_combined_csv(grid, "gsm8k", budgets=[4], fractions=[0.0])
        assert table.splitlines()[-1] == "4 Hours,60.2%"

    def test_test_time_reward_carries_mean_without_standard_deviation(self):
        cell = Cell(TEST_TIME_REWARD_LABEL, 4, 0.0, 0.5)
        grid = {(TEST_TIME_REWARD_LABEL, 4, 0.0): cell}
        table = format_combined_csv(
            grid, TEST_TIME_REWARD_LABEL, budgets=[4], fractions=[0.0],
        )
        assert table.splitlines()[-1] == "4 Hours,50.0%"


class TestReadMaxThroughEndpoint:
    def test_requires_an_eval_at_the_budget_boundary(self):
        series = [(60, 0.70), (4 * 60 + 31, 0.60)]
        assert read_max_through_endpoint(series, 4 * 60) is None


class TestFetchEvents:
    def test_fetches_reward_without_using_it_as_the_sft_offset(self):
        class RawRun:
            def scan_history(self, keys, page_size):
                metric = keys[1]
                if metric == TEST_TIME_REWARD_KEY:
                    return [
                        {"train/total_minutes": 120.0, metric: 0.5},
                        {"train/total_minutes": 240.0, metric: 0.7},
                    ]
                if metric == "eval/gsm8k_pass_at_1":
                    return [{"train/total_minutes": 240.0, metric: 0.6}]
                return []

        events, offset = _fetch_events(RawRun())

        assert offset == pytest.approx(240.0)
        assert events[0].values[TEST_TIME_REWARD_LABEL] == pytest.approx(0.5)
        assert events[1].values[TEST_TIME_REWARD_LABEL] == pytest.approx(0.7)
        assert events[1].values["gsm8k"] == pytest.approx(0.6)

    def test_captures_pass_at_1_global_step_for_variance_lookup(self):
        class RawRun:
            def scan_history(self, keys, page_size):
                metric = keys[1]
                if metric == "eval/gsm8k_pass_at_1":
                    return [{
                        "train/total_minutes": 240.0,
                        "train/global_step": 12,
                        metric: 0.6,
                    }]
                return []

        events, _ = _fetch_events(RawRun())

        assert events[0].sources["gsm8k"].global_step == 12


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

    def test_propagates_independent_variances_into_average(self):
        grid = {
            ("gsm8k", 4, 0.0): Cell("gsm8k", 4, 0.0, 0.5, standard_deviation=0.03),
            ("math500", 4, 0.0): Cell("math500", 4, 0.0, 0.4, standard_deviation=0.04),
            ("amc", 4, 0.0): Cell("amc", 4, 0.0, 0.3, standard_deviation=0.0),
        }

        add_average_cells(grid, budgets=[4], fractions=[0.0])

        assert grid[(AVERAGE_LABEL, 4, 0.0)].standard_deviation == pytest.approx(
            (0.03 ** 2 + 0.04 ** 2) ** 0.5 / 3
        )

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
