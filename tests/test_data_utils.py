# ABOUTME: Tests for data utility functions, especially unique-first sampling.
# ABOUTME: Validates that get_random_train_subset prioritizes unique values when configured.

import random

from datasets import Dataset, DatasetDict
from tuning.data.utils import get_random_train_subset


def _make_dataset(prompts, extras=None):
    """Build a DatasetDict with given prompts and a dummy test split."""
    data = {"prompt": prompts, "chosen": [f"c{i}" for i in range(len(prompts))]}
    if extras:
        data.update(extras)
    train = Dataset.from_dict(data)
    test = Dataset.from_dict({"prompt": ["test"], "chosen": ["test_c"]})
    return DatasetDict({"train": train, "test": test})


def test_unique_column_includes_all_unique_values():
    """Every unique prompt appears at least once when unique_column is set."""
    prompts = []
    for i in range(10):
        prompts.extend([f"q{i}"] * 5)  # 10 unique prompts, 50 rows total
    ds = _make_dataset(prompts)
    result = get_random_train_subset(ds, train_size=20, unique_column="prompt")
    result_prompts = result["train"]["prompt"]
    for i in range(10):
        assert f"q{i}" in result_prompts


def test_unique_column_returns_exact_train_size():
    """Output has exactly train_size rows."""
    prompts = []
    for i in range(5):
        prompts.extend([f"q{i}"] * 4)  # 5 unique, 20 total
    ds = _make_dataset(prompts)
    result = get_random_train_subset(ds, train_size=12, unique_column="prompt")
    assert len(result["train"]) == 12


def test_unique_column_fills_with_duplicates():
    """Remaining slots after unique selection are filled from duplicate rows."""
    prompts = ["q0", "q0", "q0", "q1", "q1", "q1"]
    ds = _make_dataset(prompts)
    result = get_random_train_subset(ds, train_size=4, unique_column="prompt")
    result_prompts = result["train"]["prompt"]
    assert len(result_prompts) == 4
    assert "q0" in result_prompts
    assert "q1" in result_prompts


def test_without_unique_column_behaves_as_before():
    """Without unique_column, sampling is purely random (no guarantee)."""
    prompts = [f"q{i}" for i in range(20)]
    ds = _make_dataset(prompts)
    result = get_random_train_subset(ds, train_size=10)
    assert len(result["train"]) == 10


def test_unique_column_caps_at_dataset_size():
    """When train_size exceeds dataset, returns all rows."""
    prompts = ["q0", "q1", "q2"]
    ds = _make_dataset(prompts)
    result = get_random_train_subset(ds, train_size=100, unique_column="prompt")
    assert len(result["train"]) == 3


def test_unique_column_respects_train_size_when_more_unique_than_requested():
    """When unique values exceed train_size, output is still exactly train_size."""
    prompts = [f"q{i}" for i in range(100)]  # 100 unique, 100 rows
    ds = _make_dataset(prompts)
    result = get_random_train_subset(ds, train_size=30, unique_column="prompt")
    assert len(result["train"]) == 30


def test_unique_column_handles_list_values():
    """unique_column works when column values are lists (e.g. chat-format prompts)."""
    prompts = [
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is 2+2?"}],
        [{"role": "user", "content": "What is 3+3?"}],
        [{"role": "user", "content": "What is 3+3?"}],
        [{"role": "user", "content": "What is 4+4?"}],
    ]
    ds = _make_dataset(prompts)
    result = get_random_train_subset(ds, train_size=3, unique_column="prompt")
    assert len(result["train"]) == 3


def test_same_seed_gives_same_sample():
    """Two calls with the same seed select the same rows in the same order."""
    ds = _make_dataset([f"q{i}" for i in range(100)])
    first = get_random_train_subset(ds, train_size=20, seed=42)
    second = get_random_train_subset(ds, train_size=20, seed=42)
    assert first["train"]["prompt"] == second["train"]["prompt"]


def test_different_seeds_give_different_samples():
    """Different seeds select different rows."""
    ds = _make_dataset([f"q{i}" for i in range(100)])
    first = get_random_train_subset(ds, train_size=20, seed=1)
    second = get_random_train_subset(ds, train_size=20, seed=2)
    assert first["train"]["prompt"] != second["train"]["prompt"]


def test_seeded_sampling_leaves_global_rng_untouched():
    """Sampling must not reseed the process-wide RNG used elsewhere in training."""
    ds = _make_dataset([f"q{i}" for i in range(100)])

    random.seed(123)
    expected = [random.random() for _ in range(3)]

    random.seed(123)
    get_random_train_subset(ds, train_size=20, seed=7)
    actual = [random.random() for _ in range(3)]

    assert actual == expected


def test_seeded_sampling_applies_to_unique_column_path():
    """The unique_column branch is seeded too, not just the plain branch."""
    prompts = []
    for i in range(50):
        prompts.extend([f"q{i}"] * 3)
    ds = _make_dataset(prompts)
    first = get_random_train_subset(ds, train_size=30, unique_column="prompt", seed=5)
    second = get_random_train_subset(ds, train_size=30, unique_column="prompt", seed=5)
    assert first["train"]["prompt"] == second["train"]["prompt"]
