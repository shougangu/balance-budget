# ABOUTME: Tests for the full MATH (5000) and AIME 2024 test dataset loaders and eval strategies
# ABOUTME: used to compare our checkpoints against OpenMathInstruct-2's published protocol.

import pytest
from tuning.data.test_dataset import (
    get_aime24_test_dataset,
    get_math_test_dataset,
    last_boxed_content,
)


class TestLastBoxedContent:
    def test_plain(self):
        assert last_boxed_content(r"so $x = \boxed{42}$.") == "42"

    def test_nested_braces(self):
        assert last_boxed_content(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_last_occurrence_wins(self):
        assert last_boxed_content(r"\boxed{1} then \boxed{2}") == "2"

    def test_missing_raises(self):
        with pytest.raises(ValueError):
            last_boxed_content("no answer here")


class TestGetMathTestDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        return get_math_test_dataset()

    def test_has_required_columns(self, dataset):
        assert set(dataset.column_names) >= {"messages", "prompt", "reference_answer"}

    def test_messages_have_system_and_user(self, dataset):
        msgs = dataset[0]["messages"]
        assert [m["role"] for m in msgs] == ["system", "user"]

    def test_full_size(self, dataset):
        assert len(dataset) == 5000

    def test_reference_answers_are_nonempty_strings(self, dataset):
        for ref in dataset["reference_answer"]:
            assert isinstance(ref, str) and ref

    def test_num_prompts_subset(self):
        assert len(get_math_test_dataset(num_prompts=10)) == 10


class TestGetAime24TestDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        return get_aime24_test_dataset()

    def test_full_size(self, dataset):
        assert len(dataset) == 30

    def test_first_reference(self, dataset):
        assert dataset[0]["reference_answer"] == "204"
        assert dataset[0]["prompt"].startswith("Problem: Every morning Aya")

    def test_all_references_are_integers(self, dataset):
        for ref in dataset["reference_answer"]:
            assert ref.isdigit()


class TestEvalStrategies:
    def test_math_strategy(self):
        from tuning.training.eval_strategy import MATHEvalStrategy
        s = MATHEvalStrategy(k_values=[1], n_samples=1, num_prompts=3)
        assert s.id == "math"
        assert len(s.get_test_messages()) == 3
        assert s.wandb_metrics({"pass_at_1": 0.5})["eval/math_pass_at_1"] == 0.5

    def test_aime24_strategy(self):
        from tuning.training.eval_strategy import AIME24EvalStrategy
        s = AIME24EvalStrategy(k_values=[1], n_samples=1)
        assert s.id == "aime24"
        assert len(s.get_test_prompts()) == 30
        assert s.label_prefix == "aime24-p@1"
