# ABOUTME: Tests weight averaging over equally spaced training checkpoints, the final
# ABOUTME: model construction step the OpenMathInstruct-2 recipe uses.

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.average_checkpoints import (
    average_checkpoints,
    pick_equally_spaced,
)


def _write_checkpoint(directory, tensors, shard_names=None):
    directory.mkdir(parents=True, exist_ok=True)
    if shard_names is None:
        save_file(tensors, str(directory / "model.safetensors"))
    else:
        index = {"metadata": {}, "weight_map": {}}
        for shard, keys in shard_names.items():
            save_file({k: tensors[k] for k in keys}, str(directory / shard))
            for k in keys:
                index["weight_map"][k] = shard
        (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    (directory / "config.json").write_text(json.dumps({"model_type": "llama"}))
    return directory


def test_pick_equally_spaced_includes_both_ends():
    assert pick_equally_spaced(list(range(10)), 2) == [0, 9]


def test_pick_equally_spaced_spreads_across_the_run():
    assert pick_equally_spaced(list(range(11)), 6) == [0, 2, 4, 6, 8, 10]


def test_pick_equally_spaced_returns_all_when_fewer_available():
    assert pick_equally_spaced([1, 2, 3], 6) == [1, 2, 3]


def test_pick_equally_spaced_rejects_non_positive_k():
    with pytest.raises(ValueError):
        pick_equally_spaced([1, 2, 3], 0)


def test_average_of_two_checkpoints_is_the_mean(tmp_path):
    a = _write_checkpoint(tmp_path / "a", {"w": torch.tensor([0.0, 2.0])})
    b = _write_checkpoint(tmp_path / "b", {"w": torch.tensor([2.0, 4.0])})
    out = average_checkpoints([a, b], tmp_path / "avg")
    assert torch.allclose(load_file(str(out / "model.safetensors"))["w"],
                          torch.tensor([1.0, 3.0]))


def test_average_preserves_input_dtype(tmp_path):
    a = _write_checkpoint(tmp_path / "a", {"w": torch.tensor([1.0], dtype=torch.bfloat16)})
    b = _write_checkpoint(tmp_path / "b", {"w": torch.tensor([2.0], dtype=torch.bfloat16)})
    out = average_checkpoints([a, b], tmp_path / "avg")
    assert load_file(str(out / "model.safetensors"))["w"].dtype == torch.bfloat16


def test_average_accumulates_in_fp32(tmp_path):
    # Averaging three bf16 values that bf16 accumulation would round away from.
    values = [1.0, 1.0078125, 1.015625]
    dirs = [_write_checkpoint(tmp_path / f"c{i}", {"w": torch.tensor([v], dtype=torch.bfloat16)})
            for i, v in enumerate(values)]
    out = average_checkpoints(dirs, tmp_path / "avg")
    result = load_file(str(out / "model.safetensors"))["w"].float()
    expected = torch.tensor([sum(values) / 3]).bfloat16().float()
    assert torch.allclose(result, expected)


def test_integer_tensors_are_taken_not_averaged(tmp_path):
    a = _write_checkpoint(tmp_path / "a", {"ids": torch.tensor([0, 1], dtype=torch.long)})
    b = _write_checkpoint(tmp_path / "b", {"ids": torch.tensor([0, 3], dtype=torch.long)})
    out = average_checkpoints([a, b], tmp_path / "avg")
    got = load_file(str(out / "model.safetensors"))["ids"]
    assert got.dtype == torch.long
    assert torch.equal(got, torch.tensor([0, 3]))


def test_mismatched_keys_are_rejected(tmp_path):
    a = _write_checkpoint(tmp_path / "a", {"w": torch.tensor([1.0])})
    b = _write_checkpoint(tmp_path / "b", {"v": torch.tensor([1.0])})
    with pytest.raises(ValueError, match="key"):
        average_checkpoints([a, b], tmp_path / "avg")


def test_sharded_checkpoints_are_averaged_and_index_written(tmp_path):
    tensors_a = {"w1": torch.tensor([0.0]), "w2": torch.tensor([10.0])}
    tensors_b = {"w1": torch.tensor([2.0]), "w2": torch.tensor([20.0])}
    shards = {"model-00001-of-00002.safetensors": ["w1"],
              "model-00002-of-00002.safetensors": ["w2"]}
    a = _write_checkpoint(tmp_path / "a", tensors_a, shards)
    b = _write_checkpoint(tmp_path / "b", tensors_b, shards)
    out = average_checkpoints([a, b], tmp_path / "avg")

    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"w1", "w2"}
    merged = {}
    for shard in set(index["weight_map"].values()):
        merged.update(load_file(str(out / shard)))
    assert torch.allclose(merged["w1"], torch.tensor([1.0]))
    assert torch.allclose(merged["w2"], torch.tensor([15.0]))


def test_config_is_copied_from_the_first_checkpoint(tmp_path):
    a = _write_checkpoint(tmp_path / "a", {"w": torch.tensor([1.0])})
    b = _write_checkpoint(tmp_path / "b", {"w": torch.tensor([3.0])})
    out = average_checkpoints([a, b], tmp_path / "avg")
    assert json.loads((out / "config.json").read_text())["model_type"] == "llama"


def test_single_checkpoint_is_copied_unchanged(tmp_path):
    a = _write_checkpoint(tmp_path / "a", {"w": torch.tensor([7.0])})
    out = average_checkpoints([a], tmp_path / "avg")
    assert torch.allclose(load_file(str(out / "model.safetensors"))["w"],
                          torch.tensor([7.0]))


def test_empty_checkpoint_list_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        average_checkpoints([], tmp_path / "avg")
