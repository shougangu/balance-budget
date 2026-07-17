# ABOUTME: Tests for the OOD math test dataset loaders (Minerva Math, OlympiadBench).
# ABOUTME: Validates dataset structure and OlympiadBench text-only/single-answer filtering.

import pytest
from tuning.data.test_dataset import (
    get_minervamath_test_dataset,
    get_olympiadbench_test_dataset,
)


class TestGetMinervaMathTestDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        return get_minervamath_test_dataset()

    def test_has_required_columns(self, dataset):
        assert "messages" in dataset.column_names
        assert "prompt" in dataset.column_names
        assert "reference_answer" in dataset.column_names

    def test_messages_have_system_and_user(self, dataset):
        msgs = dataset[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_reference_answer_is_nonempty_string(self, dataset):
        assert isinstance(dataset[0]["reference_answer"], str)
        assert len(dataset[0]["reference_answer"]) > 0

    def test_full_size(self, dataset):
        assert len(dataset) == 272

    def test_num_prompts_subset(self):
        dataset = get_minervamath_test_dataset(num_prompts=10)
        assert len(dataset) == 10


class TestGetOlympiadBenchTestDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        return get_olympiadbench_test_dataset()

    def test_has_required_columns(self, dataset):
        assert "messages" in dataset.column_names
        assert "prompt" in dataset.column_names
        assert "reference_answer" in dataset.column_names

    def test_messages_have_system_and_user(self, dataset):
        msgs = dataset[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_reference_answers_are_nonempty_strings(self, dataset):
        for ref in dataset["reference_answer"][:20]:
            assert isinstance(ref, str)
            assert len(ref) > 0

    def test_filtered_to_text_only_single_answer(self, dataset):
        # The raw OE_TO_maths_en_COMP split has 674 rows including
        # multiple-answer problems; the loader keeps single-answer text
        # problems only, so it must be a strict, non-empty subset.
        assert 0 < len(dataset) < 674

    def test_num_prompts_subset(self):
        dataset = get_olympiadbench_test_dataset(num_prompts=10)
        assert len(dataset) == 10
