# ABOUTME: Tests for IFBench test dataset loader.
# ABOUTME: Validates format, columns, and constraint metadata.

from tuning.data.test_dataset import get_ifbench_test_dataset


def test_ifbench_test_dataset_has_required_columns():
    dataset = get_ifbench_test_dataset(num_prompts=5)
    assert "messages" in dataset.column_names
    assert "prompt" in dataset.column_names
    assert "instruction_id_list" in dataset.column_names
    assert "kwargs" in dataset.column_names


def test_ifbench_test_dataset_messages_format():
    dataset = get_ifbench_test_dataset(num_prompts=5)
    row = dataset[0]
    messages = row["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert len(messages[1]["content"]) > 0


def test_ifbench_test_dataset_prompt_matches_message():
    dataset = get_ifbench_test_dataset(num_prompts=5)
    row = dataset[0]
    assert row["prompt"] == row["messages"][1]["content"]


def test_ifbench_test_dataset_has_constraint_metadata():
    dataset = get_ifbench_test_dataset(num_prompts=5)
    row = dataset[0]
    assert isinstance(row["instruction_id_list"], list)
    assert len(row["instruction_id_list"]) >= 1
    assert isinstance(row["kwargs"], list)
    assert len(row["kwargs"]) == len(row["instruction_id_list"])


def test_ifbench_test_dataset_num_prompts_limits():
    dataset = get_ifbench_test_dataset(num_prompts=3)
    assert len(dataset) == 3


def test_ifbench_test_dataset_full_size():
    dataset = get_ifbench_test_dataset()
    assert len(dataset) >= 72
