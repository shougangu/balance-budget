import pytest

from tuning.evaluation.ifeval_evaluate_harness import parse_ifeval_stdout, format_model_results


SAMPLE_STDOUT = """
================================================================
/tmp/outputs/model-a/eval_results_strict.jsonl Accuracy Scores:
prompt-level: 0.3826247689463956
instruction-level: 0.5035971223021583

change_case 0.42696629213483145
================================================================
/tmp/outputs/model-a/eval_results_loose.jsonl Accuracy Scores:
prompt-level: 0.4584103512014787
instruction-level: 0.579136690647482
"""


def test_parse_ifeval_stdout_extracts_strict_and_loose_scores():
    parsed = parse_ifeval_stdout(SAMPLE_STDOUT)

    assert parsed["strict"]["prompt_level"] == pytest.approx(0.3826247689463956)
    assert parsed["strict"]["instruction_level"] == pytest.approx(0.5035971223021583)
    assert parsed["loose"]["prompt_level"] == pytest.approx(0.4584103512014787)
    assert parsed["loose"]["instruction_level"] == pytest.approx(0.579136690647482)


def test_parse_ifeval_stdout_raises_when_section_missing():
    with pytest.raises(ValueError, match="Unable to parse IFeval metrics"):
        parse_ifeval_stdout("prompt-level: 0.1\\ninstruction-level: 0.2")


def test_format_model_results_single_run():
    scores = [
        {"strict": {"prompt_level": 0.38, "instruction_level": 0.50},
         "loose": {"prompt_level": 0.46, "instruction_level": 0.58}},
    ]
    result = format_model_results("test-model", scores)
    assert "## test-model" in result
    assert "| 1 | 0.3800 | 0.5000 | 0.4600 | 0.5800 |" in result
    assert "| Avg | 0.3800 | 0.5000 | 0.4600 | 0.5800 |" in result


def test_format_model_results_averages_three_runs():
    scores = [
        {"strict": {"prompt_level": 0.30, "instruction_level": 0.40},
         "loose": {"prompt_level": 0.50, "instruction_level": 0.60}},
        {"strict": {"prompt_level": 0.40, "instruction_level": 0.50},
         "loose": {"prompt_level": 0.60, "instruction_level": 0.70}},
        {"strict": {"prompt_level": 0.50, "instruction_level": 0.60},
         "loose": {"prompt_level": 0.70, "instruction_level": 0.80}},
    ]
    result = format_model_results("avg-model", scores)
    assert "| 1 | 0.3000 | 0.4000 | 0.5000 | 0.6000 |" in result
    assert "| 2 | 0.4000 | 0.5000 | 0.6000 | 0.7000 |" in result
    assert "| 3 | 0.5000 | 0.6000 | 0.7000 | 0.8000 |" in result
    assert "| Avg | 0.4000 | 0.5000 | 0.6000 | 0.7000 |" in result
