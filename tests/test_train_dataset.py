# ABOUTME: Regression tests for training dataset cache selection.
# ABOUTME: Ensures whole-dataset runs use the base Arrow dataset directly.

from datasets import Dataset, DatasetDict, load_from_disk

from tuning.data import train_dataset as td
from tuning.training.config_training import DatasetConfig, SFTRunConfig


def test_full_dataset_cache_miss_uses_base_without_saving(monkeypatch, tmp_path):
    base = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"prompt": ["one", "two", "three"], "messages": [[], [], []]}
            ),
            "test": Dataset.from_dict({"prompt": ["test"], "messages": [[]]}),
        }
    )
    base_path = tmp_path / "sft-tulumix"
    base.save_to_disk(str(base_path))
    persisted_base = load_from_disk(str(base_path))
    monkeypatch.setattr(td, "DATASETS_DIR", str(tmp_path))

    def fail_if_sampled(*args, **kwargs):
        raise AssertionError("the full dataset must not be sampled")

    monkeypatch.setattr(td, "get_random_train_subset", fail_if_sampled)
    config = SFTRunConfig(
        model_name="llama3-8B",
        dataset_config=DatasetConfig(
            dataset="tulumix", dataset_type="sft", train_size=100_000_000
        ),
    )

    loaded = td.get_train_dataset(config)

    assert len(loaded["train"]) == 3
    assert loaded["train"]._fingerprint == persisted_base["train"]._fingerprint
    assert not (tmp_path / "sft-tulumix-100000000").exists()
