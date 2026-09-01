# ABOUTME: Tests the sampled-dataset cache under torchrun: rank 0 builds, other ranks wait
# ABOUTME: for a complete cache, and a half-written cache from a killed run is rebuilt.

from types import SimpleNamespace
from unittest.mock import patch

from datasets import Dataset, DatasetDict

import tuning.data.train_dataset as td


def _base_dataset(tmp_path):
    base = DatasetDict({
        "train": Dataset.from_dict({"prompt": [f"q{i}" for i in range(20)]}),
        "test": Dataset.from_dict({"prompt": ["t0", "t1"]}),
    })
    base.save_to_disk(str(tmp_path / "sft-gsm8k"))
    return base


def _run_config(train_size=5):
    return SimpleNamespace(
        run_type="sft",
        dataset_config=SimpleNamespace(dataset="gsm8k", train_size=train_size),
    )


def _partial_cache(tmp_path, base):
    """What a rank-0 save looks like from another rank mid-write: the
    directory and the train split exist, the test split does not yet."""
    cache = tmp_path / "sft-gsm8k-5"
    cache.mkdir()
    (cache / "dataset_dict.json").write_text('{"splits": ["train", "test"]}')
    base["train"].select(range(5)).save_to_disk(str(cache / "train"))
    return cache


def _dist(rank, on_barrier=None):
    calls = []

    def barrier():
        calls.append("barrier")
        if on_barrier:
            on_barrier()

    fake = SimpleNamespace(
        is_initialized=lambda: True, get_world_size=lambda: 2,
        get_rank=lambda: rank, barrier=barrier,
    )
    return fake, calls


def test_non_zero_rank_waits_for_rank_zero_before_reading_the_cache(tmp_path):
    base = _base_dataset(tmp_path)
    cache = _partial_cache(tmp_path, base)

    def finish_save():
        base["test"].save_to_disk(str(cache / "test"))

    fake_dist, calls = _dist(rank=1, on_barrier=finish_save)
    with patch.object(td, "DATASETS_DIR", str(tmp_path)), patch.object(td, "dist", fake_dist):
        dataset = td.get_train_dataset(_run_config())
    assert calls == ["barrier"]
    assert len(dataset["train"]) == 5 and len(dataset["test"]) == 2


def test_rank_zero_rebuilds_a_half_written_cache(tmp_path):
    base = _base_dataset(tmp_path)
    _partial_cache(tmp_path, base)
    fake_dist, calls = _dist(rank=0)
    with patch.object(td, "DATASETS_DIR", str(tmp_path)), patch.object(td, "dist", fake_dist):
        dataset = td.get_train_dataset(_run_config())
    assert calls == ["barrier"]
    assert len(dataset["train"]) == 5 and len(dataset["test"]) == 2
    assert (tmp_path / "sft-gsm8k-5" / "test" / "state.json").is_file()


def test_complete_cache_is_reused_without_rebuilding(tmp_path):
    base = _base_dataset(tmp_path)
    cache = tmp_path / "sft-gsm8k-5"
    DatasetDict({"train": base["train"].select(range(5)), "test": base["test"]}).save_to_disk(str(cache))
    with patch.object(td, "DATASETS_DIR", str(tmp_path)), \
         patch.object(td, "get_random_train_subset", side_effect=AssertionError("must not rebuild")):
        dataset = td.get_train_dataset(_run_config())
    assert len(dataset["train"]) == 5
