# ABOUTME: Tests for the sweetspot_table script that generates SFT→DPO analysis tables.
# ABOUTME: Covers URL parsing, run matching, metric extraction, and table formatting.

import pytest

from scripts.sweetspot_table import (
    DpoRow,
    RunInfo,
    SftBaseline,
    extract_dpo_row,
    extract_sft_baseline,
    find_dpo_forks,
    find_latest_sft_run,
    format_table,
    parse_wandb_url,
)


class TestParseWandbUrl:
    def test_parses_full_url(self):
        url = "https://wandb.ai/shougan-university-of-waterloo/gsm8k-llama3-3B"
        assert parse_wandb_url(url) == ("shougan-university-of-waterloo", "gsm8k-llama3-3B")

    def test_parses_url_with_trailing_slash(self):
        url = "https://wandb.ai/shougan-university-of-waterloo/gsm8k-llama3-3B/"
        assert parse_wandb_url(url) == ("shougan-university-of-waterloo", "gsm8k-llama3-3B")

    def test_parses_url_with_query_params(self):
        url = "https://wandb.ai/shougan-university-of-waterloo/gsm8k-llama3-3B?nw=nwusershougan"
        assert parse_wandb_url(url) == ("shougan-university-of-waterloo", "gsm8k-llama3-3B")

    def test_rejects_invalid_url(self):
        with pytest.raises(ValueError):
            parse_wandb_url("not-a-url")

    def test_rejects_non_wandb_url(self):
        with pytest.raises(ValueError):
            parse_wandb_url("https://example.com/foo/bar")


def _make_run(name, tags, created_at, state="finished"):
    return RunInfo(name=name, tags=tags, created_at=created_at, state=state)


class TestFindLatestSftRun:
    def test_finds_most_recent_sft(self):
        runs = [
            _make_run("old-sft", ["sft", "p1"], "2026-03-07T09:00:00Z"),
            _make_run("new-sft", ["sft", "p1", "gsm8k"], "2026-03-15T06:00:00Z"),
            _make_run("dpo-run", ["dpo", "p1", "0.5", "6656"], "2026-03-15T12:00:00Z"),
        ]
        result = find_latest_sft_run(runs)
        assert result.name == "new-sft"

    def test_ignores_non_finished_runs(self):
        runs = [
            _make_run("crashed", ["sft", "p1"], "2026-03-16T00:00:00Z", state="crashed"),
            _make_run("good", ["sft", "p1"], "2026-03-15T06:00:00Z"),
        ]
        result = find_latest_sft_run(runs)
        assert result.name == "good"

    def test_raises_when_no_sft_runs(self):
        runs = [
            _make_run("dpo-only", ["dpo", "p1"], "2026-03-15T12:00:00Z"),
        ]
        with pytest.raises(ValueError, match="No finished SFT"):
            find_latest_sft_run(runs)


class TestFindDpoForks:
    def test_finds_dpo_runs_matching_sft_group(self):
        sft = _make_run("sft", ["sft", "p1", "gsm8k", "early_pair:none"], "2026-03-15T06:00:00Z")
        runs = [
            sft,
            _make_run("dpo1", ["dpo", "p1", "gsm8k", "0.5", "6656"], "2026-03-15T12:00:00Z"),
            _make_run("dpo2", ["dpo", "p1", "gsm8k", "0.45", "4608"], "2026-03-15T11:00:00Z"),
            # Different group — no gsm8k tag
            _make_run("dpo-other", ["dpo", "p1", "0.5", "7168"], "2026-03-07T13:00:00Z"),
        ]
        forks = find_dpo_forks(sft, runs)
        names = {f.name for f in forks}
        assert names == {"dpo1", "dpo2"}

    def test_excludes_dpo_runs_before_sft(self):
        sft = _make_run("sft", ["sft", "p1", "gsm8k", "early_pair:none"], "2026-03-15T06:00:00Z")
        runs = [
            sft,
            _make_run("old-dpo", ["dpo", "p1", "gsm8k", "0.5", "6656"], "2026-03-14T12:00:00Z"),
        ]
        forks = find_dpo_forks(sft, runs)
        assert forks == []

    def test_excludes_non_finished_dpo(self):
        sft = _make_run("sft", ["sft", "p1", "early_pair:none"], "2026-03-15T06:00:00Z")
        runs = [
            sft,
            _make_run("crashed", ["dpo", "p1", "0.5", "6656"], "2026-03-15T12:00:00Z", state="crashed"),
        ]
        forks = find_dpo_forks(sft, runs)
        assert forks == []


class TestExtractSftBaseline:
    def test_computes_total_budget_and_accuracies(self):
        sft_run = RunInfo(
            name="sft",
            tags=["sft", "p1"],
            created_at="2026-03-15T06:00:00Z",
            config={
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "num_train_epochs": 2,
            },
            summary={
                "train/global_step": 1250,
                "eval/gsm8k_pass_at_1": 0.5110,
            },
        )
        history = [0.0832, 0.2100, 0.3500, 0.5110]
        baseline = extract_sft_baseline(sft_run, history)
        assert baseline.total_budget == 10000
        assert baseline.base_acc == pytest.approx(0.0832)
        assert baseline.sft_full_acc == pytest.approx(0.5110)

    def test_uses_summary_when_no_history(self):
        sft_run = RunInfo(
            name="sft",
            tags=["sft", "p1"],
            created_at="2026-03-15T06:00:00Z",
            config={
                "per_device_train_batch_size": 16,
                "gradient_accumulation_steps": 1,
                "num_train_epochs": 2,
            },
            summary={
                "train/global_step": 1250,
                "eval/gsm8k_pass_at_1": 0.5110,
            },
        )
        baseline = extract_sft_baseline(sft_run, history=[])
        assert baseline.base_acc is None
        assert baseline.sft_full_acc == pytest.approx(0.5110)


class TestExtractDpoRow:
    def test_extracts_all_fields(self):
        dpo_run = RunInfo(
            name="dpo",
            tags=["dpo", "p1", "gsm8k", "0.50", "6656"],
            created_at="2026-03-15T12:00:00Z",
        )
        history = [0.5083, 0.5200, 0.5342, 0.5342]
        row = extract_dpo_row(dpo_run, total_budget=10000, history=history)
        assert row.pass_at_1_threshold == pytest.approx(0.50)
        assert row.sft_data == 6656
        assert row.dpo_data == 3344
        assert row.sft_acc_at_stop == pytest.approx(0.5083)
        assert row.best_dpo_acc == pytest.approx(0.5342)
        assert row.final_dpo_acc == pytest.approx(0.5342)

    def test_handles_single_eval_point(self):
        dpo_run = RunInfo(
            name="dpo",
            tags=["dpo", "p1", "0.25", "512"],
            created_at="2026-03-15T08:00:00Z",
        )
        history = [0.2662]
        row = extract_dpo_row(dpo_run, total_budget=10000, history=history)
        assert row.sft_acc_at_stop == pytest.approx(0.2662)
        assert row.best_dpo_acc == pytest.approx(0.2662)
        assert row.final_dpo_acc == pytest.approx(0.2662)


class TestFormatTable:
    def test_formats_complete_table(self):
        baseline = SftBaseline(
            total_budget=10000,
            base_acc=0.0832,
            sft_full_acc=0.5110,
        )
        rows = [
            DpoRow(
                pass_at_1_threshold=0.50,
                sft_data=6656,
                dpo_data=3344,
                sft_acc_at_stop=0.5083,
                best_dpo_acc=0.5342,
                final_dpo_acc=0.5342,
            ),
            DpoRow(
                pass_at_1_threshold=0.25,
                sft_data=512,
                dpo_data=9488,
                sft_acc_at_stop=0.2569,
                best_dpo_acc=0.3364,
                final_dpo_acc=0.3253,
            ),
        ]
        table = format_table(baseline, rows)
        lines = table.strip().split("\n")
        # Header + SFT-only row + 2 DPO rows = 4 lines
        assert len(lines) == 4
        # SFT-only row
        assert "SFT-only" in lines[1]
        assert "51.10%" in lines[1]
        assert "baseline" in lines[1]
        # 0.50 row should come before 0.25 (sorted by sft_data desc)
        assert "0.50" in lines[2]
        assert "0.25" in lines[3]
        # vs SFT-full: 53.42% - 51.10% = +2.32%
        assert "+2.32%" in lines[2]
        # vs SFT-0: 53.42% - 8.32% = +45.10%
        assert "+45.10%" in lines[2]
        # vs SFT-0 for SFT-only: 51.10% - 8.32% = +42.78%
        assert "+42.78%" in lines[1]

    def test_handles_no_base_acc(self):
        baseline = SftBaseline(
            total_budget=10000,
            base_acc=None,
            sft_full_acc=0.5110,
        )
        rows = [
            DpoRow(
                pass_at_1_threshold=0.50,
                sft_data=6656,
                dpo_data=3344,
                sft_acc_at_stop=0.5083,
                best_dpo_acc=0.5342,
                final_dpo_acc=0.5342,
            ),
        ]
        table = format_table(baseline, rows)
        # Should not have vs SFT-0 column when base_acc is None
        assert "vs SFT-0" not in table
