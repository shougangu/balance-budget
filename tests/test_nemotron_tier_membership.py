# ABOUTME: Tests for the Nemotron-Math-v2 tier membership analysis.
# ABOUTME: Covers pass-rate flattening and the kept-solutions vs verified-solutions comparison.

from collections import Counter

from scripts.nemotron_length_stats import REGIMES, flatten_pass_rates, tier_agreement


def _regime(count, passed):
    return {"count": count, "pass": passed, "accuracy": passed / count}


def test_flatten_pass_rates_produces_a_count_and_pass_per_regime():
    metadata = {f"reason_{regime}": _regime(8, i) for i, regime in enumerate(REGIMES)}
    flat = flatten_pass_rates(metadata)
    assert flat["low_no_tool_count"] == 8
    assert flat["low_no_tool_pass"] == REGIMES.index("low_no_tool")
    assert flat["high_with_tool_pass"] == REGIMES.index("high_with_tool")
    assert len(flat) == 2 * len(REGIMES)


def test_flatten_pass_rates_treats_a_missing_regime_as_zero_samples():
    flat = flatten_pass_rates({"reason_low_no_tool": _regime(8, 3)})
    assert flat["low_no_tool_count"] == 8 and flat["low_no_tool_pass"] == 3
    assert flat["high_no_tool_count"] == 0 and flat["high_no_tool_pass"] == 0


def test_tier_agreement_matches_kept_rows_against_verified_solutions():
    kept = {1: 6, 2: 0, 4: 3}
    verified = {1: 6, 2: 3, 3: 5, 4: 2}
    agreement = tier_agreement(kept, verified)
    assert agreement["problems_verified"] == 4
    assert agreement["problems_kept"] == 2
    assert agreement["rows_equal_pass"] == 1
    assert agreement["kept_but_not_verified"] == 0
    assert agreement["verified_but_not_kept"] == 2
    assert agreement["rows_exceed_pass"] == 1


def test_tier_agreement_flags_rows_present_with_no_verified_solution():
    agreement = tier_agreement({7: 2}, {7: 0})
    assert agreement["kept_but_not_verified"] == 1
    assert agreement["problems_verified"] == 0


def test_tier_agreement_counts_are_consistent_with_an_exact_match():
    kept = verified = {i: 8 for i in range(50)}
    agreement = tier_agreement(kept, verified)
    assert agreement == Counter({"problems_kept": 50, "problems_verified": 50,
                                 "rows_equal_pass": 50})


def test_kept_counts_by_mode_splits_tool_free_from_tool_rows():
    from scripts.nemotron_length_stats import kept_counts_by_mode

    tool_free, with_tool = kept_counts_by_mode(cot={1: 3, 2: 8}, all_rows={1: 10, 2: 8, 3: 5})
    assert tool_free == {1: 3, 2: 8}
    assert with_tool == {1: 7, 3: 5}


def test_kept_counts_by_mode_never_emits_a_negative_tool_count():
    from scripts.nemotron_length_stats import kept_counts_by_mode

    tool_free, with_tool = kept_counts_by_mode(cot={1: 9}, all_rows={1: 8})
    assert tool_free == {1: 9}
    assert with_tool == {}
