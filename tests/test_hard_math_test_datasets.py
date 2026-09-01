# ABOUTME: Tests for the hard competition math test sets (AIME 25/26, HMMT Feb 2025).
# ABOUTME: Verifies each loads into the OPENMATH prompt format with string reference answers.

from unittest.mock import patch

from datasets import Dataset

from tuning.data.config import COMPMATH_STRING


def _fake_rows():
    return Dataset.from_dict({
        "problem": ["Compute 1+1.", "Compute 2+2."],
        "answer": [2, 4],
    })


def _check_boxed_format(dataset):
    assert dataset["prompt"][0] == COMPMATH_STRING.format(problem="Compute 1+1.")
    assert dataset["reference_answer"] == ["2", "4"]
    assert dataset["messages"][0][0]["role"] == "system"


def test_aime25_dataset_format():
    from tuning.data.test_dataset import get_aime25_test_dataset
    with patch("datasets.load_dataset", return_value=_fake_rows()) as load:
        dataset = get_aime25_test_dataset()
    assert load.call_args.args[0] == "math-ai/aime25"
    _check_boxed_format(dataset)


def test_aime26_dataset_format():
    from tuning.data.test_dataset import get_aime26_test_dataset
    with patch("datasets.load_dataset", return_value=_fake_rows()) as load:
        dataset = get_aime26_test_dataset()
    assert load.call_args.args[0] == "MathArena/aime_2026"
    _check_boxed_format(dataset)


def test_hmmt_feb25_dataset_format():
    from tuning.data.test_dataset import get_hmmt_feb25_test_dataset
    with patch("datasets.load_dataset", return_value=_fake_rows()) as load:
        dataset = get_hmmt_feb25_test_dataset(num_prompts=1)
    assert load.call_args.args[0] == "MathArena/hmmt_feb_2025"
    assert len(dataset) == 1


def test_hard_math_eval_strategies_delegate_to_their_datasets():
    import tuning.training.eval_strategy as es
    fake = object()
    cases = [
        ("AIME25EvalStrategy", "get_aime25_test_dataset", "aime25"),
        ("AIME26EvalStrategy", "get_aime26_test_dataset", "aime26"),
        ("HMMTFeb25EvalStrategy", "get_hmmt_feb25_test_dataset", "hmmt_feb25"),
    ]
    for cls_name, getter_name, benchmark in cases:
        cls = getattr(es, cls_name)
        assert cls.benchmark == benchmark
        with patch.object(es, getter_name, return_value=fake) as getter:
            result = cls.load_test_dataset(object.__new__(cls), num_prompts=7)
        assert result is fake
        assert getter.call_args.kwargs["num_prompts"] == 7
