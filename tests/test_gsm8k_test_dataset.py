# ABOUTME: Tests for GSM8K test dataset loader.
# ABOUTME: Validates dataset structure: messages, prompt, and reference_answer columns.

import pytest
from tuning.data.test_dataset import get_gsm8k_test_dataset


class TestGetGSM8KTestDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
        return get_gsm8k_test_dataset()

    def test_has_required_columns(self, dataset):
        assert "messages" in dataset.column_names
        assert "prompt" in dataset.column_names
        assert "reference_answer" in dataset.column_names

    def test_messages_have_system_and_user(self, dataset):
        msgs = dataset[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_prompt_is_string(self, dataset):
        assert isinstance(dataset[0]["prompt"], str)
        assert len(dataset[0]["prompt"]) > 0

    def test_reference_answer_is_string(self, dataset):
        assert isinstance(dataset[0]["reference_answer"], str)
        assert len(dataset[0]["reference_answer"]) > 0

    def test_dataset_has_rows(self, dataset):
        assert len(dataset) > 0

    def test_num_prompts_subset(self):
        dataset = get_gsm8k_test_dataset(num_prompts=10)
        assert len(dataset) == 10
