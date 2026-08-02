# ABOUTME: Tests which dataset get_train_dataset returns when a sampled cache exists.
# ABOUTME: A request covering the whole split must read the parent, never a stale -N cache.

from datasets import Dataset, DatasetDict

from tuning.data import train_dataset as train_dataset_module
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import DatasetConfig, SFTRunConfig


def _write(path, prompts):
    DatasetDict(
        {
            "train": Dataset.from_dict({"prompt": prompts}),
            "test": Dataset.from_dict({"prompt": ["heldout"]}),
        }
    ).save_to_disk(str(path))


def _run_config(train_size):
    return SFTRunConfig(
        model_name="llama3-8B",
        task_name="math",
        dataset_config=DatasetConfig(
            dataset="probe", dataset_type="sft", train_size=train_size
        ),
        do_training=True,
    )


def test_exact_coverage_request_returns_parent(tmp_path, monkeypatch):
    """train_size equal to the split size reads the parent, not the sampled cache."""
    monkeypatch.setattr(train_dataset_module, "DATASETS_DIR", str(tmp_path))
    _write(tmp_path / "sft-probe", ["a", "b", "c"])
    _write(tmp_path / "sft-probe-3", ["stale1", "stale2", "stale3"])

    result = get_train_dataset(_run_config(3))

    assert result["train"]["prompt"] == ["a", "b", "c"]


def test_oversized_request_returns_parent(tmp_path, monkeypatch):
    """train_size larger than the split reads the parent, not the sampled cache."""
    monkeypatch.setattr(train_dataset_module, "DATASETS_DIR", str(tmp_path))
    _write(tmp_path / "sft-probe", ["a", "b", "c"])
    _write(tmp_path / "sft-probe-99", ["stale1", "stale2", "stale3"])

    result = get_train_dataset(_run_config(99))

    assert result["train"]["prompt"] == ["a", "b", "c"]


def test_partial_request_still_uses_sampled_cache(tmp_path, monkeypatch):
    """A genuine subset request keeps honouring its cached sample."""
    monkeypatch.setattr(train_dataset_module, "DATASETS_DIR", str(tmp_path))
    _write(tmp_path / "sft-probe", ["a", "b", "c", "d"])
    _write(tmp_path / "sft-probe-2", ["cached1", "cached2"])

    result = get_train_dataset(_run_config(2))

    assert result["train"]["prompt"] == ["cached1", "cached2"]
