# ABOUTME: Tests the answer-presence and alternative-grader helpers used to audit
# ABOUTME: whether our math scorer loses answers the model actually got right.

from scripts.regrade_generations import (
    grade_all,
    has_boxed,
    has_hash_answer,
)


def test_has_boxed_detects_boxed_answer():
    assert has_boxed(r"So the answer is $\boxed{42}$.")
    assert has_boxed(r"\boxed{\frac{1}{2}}")


def test_has_boxed_rejects_response_without_box():
    assert not has_boxed("The answer is 42.")
    assert not has_boxed("")


def test_has_hash_answer_detects_gsm8k_format():
    assert has_hash_answer("Step 1: add\n#### 42")
    assert not has_hash_answer("The answer is 42")


def test_grade_all_agrees_on_a_boxed_correct_answer():
    grades = grade_all(r"The result is $\boxed{42}$.", "42")
    assert grades["production"]
    assert grades["math_verify"]
    assert grades["union"]


def test_grade_all_accepts_a_hash_formatted_answer():
    grades = grade_all("Step 1: add them.\n#### 42", "42")
    assert grades["production"]
    assert grades["gsm8k_numeric"]


def test_grade_all_marks_wrong_answer_incorrect_everywhere():
    grades = grade_all(r"The result is $\boxed{41}$.", "42")
    assert not any(grades.values())


def test_union_is_true_when_any_grader_accepts():
    grades = grade_all("After computing, the answer is 42", "42")
    assert grades["union"] == any(
        v for k, v in grades.items() if k != "union"
    )


def test_production_grader_for_gsm8k_is_the_numeric_path():
    # "Thus, Claire will eat \boxed{7} dozens of eggs in 4 weeks." is scored
    # wrong by the lm-eval numeric path GSM8K actually uses, because the last
    # number in the string is the "4" of "4 weeks".
    response = r"84 / 12 = 7. Thus, Claire will eat \boxed{7} dozens of eggs in 4 weeks."
    assert grade_all(response, "7", benchmark="gsm8k")["production"] is False
    assert grade_all(response, "7", benchmark="gsm8k")["math_verify"] is True


def test_production_grader_for_math500_is_math_verify():
    response = r"The answer is $\boxed{7}$ dozens over 4 weeks."
    assert grade_all(response, "7", benchmark="math500")["production"] is True


def test_production_grader_defaults_to_math_verify():
    assert grade_all(r"$\boxed{7}$", "7")["production"] is True


def test_grade_all_reports_the_accept_either_verifier():
    grades = grade_all(r"He has \boxed{7} dozens after 4 weeks.", "7", benchmark="gsm8k")
    assert not grades["production"]
    assert grades["accept_either"]


def test_grade_all_accept_either_is_false_when_both_paths_miss():
    grades = grade_all("I do not know.", "7", benchmark="gsm8k")
    assert not grades["accept_either"]
    assert not grades["union"]
