# ABOUTME: Tests the per-eval-point aggregation of the grid regrade, which reports pass@1
# ABOUTME: under the benchmark's production grader and under the accept-either verifier.

from scripts.regrade_grid import grade_rows


def test_grade_rows_gsm8k_credits_boxed_only_under_accept_either():
    rows = [
        ("7", [r"He has \boxed{7} dozens after 4 weeks.", "#### 7"]),
        ("3", ["The answer is 3.", "The answer is 4."]),
    ]
    rec = grade_rows(rows, "gsm8k")
    assert rec["pass_at_1_production"] == 50.0
    assert rec["pass_at_1_accept_either"] == 75.0
    assert rec["only_math_verify_rate"] == 25.0
    assert rec["only_numeric_rate"] == 50.0  # "#### 7" and the bare 3: numeric only


def test_grade_rows_math500_credits_bare_number_under_accept_either():
    rows = [("18", [r"$\boxed{18}$", "so the answer is 18."])]
    rec = grade_rows(rows, "math500")
    assert rec["pass_at_1_production"] == 50.0
    assert rec["pass_at_1_accept_either"] == 100.0
    assert rec["only_numeric_rate"] == 50.0


def test_grade_rows_reports_each_grader_on_its_own():
    rows = [
        ("7", [r"He has \boxed{7} dozens.", "#### 7"]),
        ("3", ["The answer is 3.", "The answer is 4."]),
    ]
    rec = grade_rows(rows, "gsm8k")
    assert rec["pass_at_1_math_verify"] == 25.0
    # both responses in the first row end in 7, which the last-number rule reads
    assert rec["pass_at_1_gsm8k_numeric"] == 75.0
    assert rec["pass_at_1_accept_either"] == 75.0


def test_each_grader_is_bounded_by_accept_either_and_union():
    rows = [("18", [r"$\boxed{18}$", "so the answer is 18.", "the answer is 19."])]
    rec = grade_rows(rows, "math500")
    for grader in ("production", "math_verify", "gsm8k_numeric"):
        assert rec[f"pass_at_1_{grader}"] <= rec["pass_at_1_accept_either"]
    assert rec["pass_at_1_accept_either"] <= rec["pass_at_1_union"]
