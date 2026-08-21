# ABOUTME: Tests building the OpenMath SFT dataset from a named corpus split, so the
# ABOUTME: fair-downsampled 1M subset can be built alongside the full corpus.

from unittest.mock import patch

import pytest
from datasets import Dataset

from tuning.data.config import COMPMATH_STRING, SYSTEM_MESSAGE_OPENMATH
from tuning.data.openmath_sft import build_openmath_sft


@pytest.fixture
def fake_corpus():
    return Dataset.from_dict({
        "problem": ["What is $2+2$?", "Integrate $x$.", "Ignore me."],
        "generated_solution": ["It is $\\boxed{4}$.", "It is $\\boxed{x^2/2}$.", "nope"],
        "problem_source": ["math", "augmented_math", "some_other_source"],
    })


@pytest.fixture
def fake_eval():
    return Dataset.from_dict({"prompt": ["held out"], "reference_answer": ["1"],
                              "messages": [[{"role": "user", "content": "held out"}]]})


def _run_build(fake_corpus, fake_eval, **kwargs):
    with patch("tuning.data.openmath_sft.load_dataset", return_value=fake_corpus) as load, \
         patch("tuning.data.openmath_sft.build_heldout_math_eval", return_value=fake_eval), \
         patch("tuning.data.hf_dataset.HFDataset.save_dataset_to_disk") as save, \
         patch("tuning.data.hf_dataset.HFDataset.clear_old_datasets") as clear:
        dataset = build_openmath_sft(**kwargs)
    return dataset, load, save, clear


def test_build_loads_the_requested_split(fake_corpus, fake_eval):
    _, load, _, _ = _run_build(fake_corpus, fake_eval, split="train_1M",
                               save_name="sft-openmath-1M")
    assert load.call_args.args[0] == "nvidia/OpenMathInstruct-2"
    assert load.call_args.kwargs["split"] == "train_1M"


def test_build_saves_under_the_requested_name(fake_corpus, fake_eval):
    _, _, save, _ = _run_build(fake_corpus, fake_eval, split="train_1M",
                               save_name="sft-openmath-1M")
    assert save.call_args.kwargs["save_name"] == "sft-openmath-1M"


def test_build_does_not_clear_sibling_datasets_by_default(fake_corpus, fake_eval):
    _, _, _, clear = _run_build(fake_corpus, fake_eval, split="train_1M",
                                save_name="sft-openmath-1M")
    clear.assert_not_called()


def test_build_clears_only_its_own_name_when_asked(fake_corpus, fake_eval):
    _, _, _, clear = _run_build(fake_corpus, fake_eval, split="train_1M",
                                save_name="sft-openmath-1M", clear_existing=True)
    assert clear.call_args.kwargs["prefix"] == "sft-openmath-1M"


def test_build_keeps_only_math_sources(fake_corpus, fake_eval):
    dataset, _, _, _ = _run_build(fake_corpus, fake_eval, split="train_1M",
                                  save_name="sft-openmath-1M")
    assert len(dataset["train"]) == 2


def test_build_formats_rows_the_same_way_as_the_full_corpus(fake_corpus, fake_eval):
    dataset, _, _, _ = _run_build(fake_corpus, fake_eval, split="train_1M",
                                  save_name="sft-openmath-1M")
    row = dataset["train"][0]
    expected_prompt = COMPMATH_STRING.format(problem="What is $2+2$?")
    assert row["prompt"] == expected_prompt
    assert row["messages"] == [
        {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
        {"role": "user", "content": expected_prompt},
        {"role": "assistant", "content": "It is $\\boxed{4}$."},
    ]


def test_build_attaches_the_heldout_eval_split(fake_corpus, fake_eval):
    dataset, _, _, _ = _run_build(fake_corpus, fake_eval, split="train_1M",
                                  save_name="sft-openmath-1M")
    assert dataset["test"]["prompt"] == ["held out"]
