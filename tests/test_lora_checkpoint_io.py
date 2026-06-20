# ABOUTME: Tests adapter-only checkpoint saving (forks + finals) and the
# ABOUTME: adapter-dir detection helpers used by load_model_with_lora.

import json
from pathlib import Path
from types import SimpleNamespace

from transformers import TrainerState


class _FakeModel:
    """Records save calls; supports the adapter and merged save APIs."""

    def __init__(self):
        self.saved_pretrained = None
        self.merged_calls = []

    def save_pretrained(self, path, *args, **kwargs):
        self.saved_pretrained = path

    def save_pretrained_merged(self, path, tokenizer, save_method=None):
        self.merged_calls.append((path, save_method))


class _FakeTokenizer:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, path):
        self.saved_to = path


# ---------------------------------------------------------------------------
# Adapter-dir detection
# ---------------------------------------------------------------------------

def test_is_adapter_checkpoint_true_when_adapter_config_present(tmp_path):
    from tuning.training.model_utils import _is_adapter_checkpoint

    (tmp_path / "adapter_config.json").write_text("{}")
    assert _is_adapter_checkpoint(str(tmp_path)) is True


def test_is_adapter_checkpoint_false_for_full_model_dir(tmp_path):
    from tuning.training.model_utils import _is_adapter_checkpoint

    (tmp_path / "config.json").write_text("{}")
    assert _is_adapter_checkpoint(str(tmp_path)) is False


def test_adapter_base_path_reads_base_model_name(tmp_path):
    from tuning.training.model_utils import _adapter_base_path

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/some/base/model"})
    )
    assert _adapter_base_path(str(tmp_path)) == "/some/base/model"


def test_adapter_base_path_without_base_ref_raises(tmp_path):
    import pytest
    from tuning.training.model_utils import _adapter_base_path

    (tmp_path / "adapter_config.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="base_model_name_or_path"):
        _adapter_base_path(str(tmp_path))


# ---------------------------------------------------------------------------
# Saving: finals and forks store adapters only
# ---------------------------------------------------------------------------

def test_save_trained_model_saves_adapter_only(tmp_path):
    from tuning.training.model_utils import save_trained_model

    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    trainer = SimpleNamespace(
        args=SimpleNamespace(to_dict=lambda: {"lr": 1e-5}),
        state=TrainerState(),
    )

    save_trained_model(model, tokenizer, trainer, str(tmp_path))

    assert model.saved_pretrained == str(tmp_path)
    assert model.merged_calls == []  # never merges a full model
    assert tokenizer.saved_to == str(tmp_path)
    assert (tmp_path / "training_config.json").is_file()
    assert (tmp_path / "trainer_state.json").is_file()


def test_save_sweetspot_checkpoint_saves_adapter_only(tmp_path, monkeypatch):
    import tuning.training.callback_utils as cu

    monkeypatch.setattr(cu, "MODELS_DIR", str(tmp_path))
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    state = TrainerState()
    state.global_step = 10
    args = SimpleNamespace(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        world_size=1,
        to_dict=lambda: {},
    )
    metadata_path = str(tmp_path / "meta.jsonl")

    cp = cu.save_sweetspot_checkpoint(
        model=model,
        tokenizer=tokenizer,
        model_name="qwen2-3B",
        threshold_label="pass@1-0.3",
        state=state,
        args=args,
        metadata_path=metadata_path,
    )

    assert model.saved_pretrained == cp
    assert model.merged_calls == []  # no merged_16bit save
    assert tokenizer.saved_to == cp
    assert Path(cp, "training_config.json").is_file()
    rows = [json.loads(line) for line in open(metadata_path)]
    assert rows[0]["checkpoint_path"] == cp
