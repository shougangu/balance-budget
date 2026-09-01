# ABOUTME: Tests for BudgetMarkCallback: GPU-minute marks bank a checkpoint + metadata row
# ABOUTME: without any in-loop eval, survive resume without re-firing, and pre-claim eval-only marks.

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import TrainerControl, TrainerState


class DummyModel:
    def __init__(self):
        self.saved = []

    def save_pretrained(self, path, state_dict=None):
        self.saved.append({"path": path, "state_dict": state_dict})


class DummyTokenizer:
    def save_pretrained(self, path):
        pass


def _args():
    return SimpleNamespace(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        world_size=1,
        to_dict=lambda: {},
    )


def _state(total_minutes, global_step=10):
    state = TrainerState()
    state.global_step = global_step
    state.stateful_callbacks = {
        "OffsetAwareWandbCallback": {
            "args": {"initial_global_step": 0},
            "attributes": {"total_seconds": total_minutes * 60.0},
        }
    }
    return state


def _callback(tmp_path, marks, eval_only=None):
    from tuning.training.budget_marks import BudgetMarkCallback

    return BudgetMarkCallback(
        model_name="qwen3-8B",
        tokenizer=DummyTokenizer(),
        target_total_minutes=marks,
        eval_only_minutes=eval_only,
        metadata_path=str(tmp_path / "marks.json"),
    )


def _run_step(cb, model, total_minutes, tmp_path, args=None):
    with patch("tuning.training.callback_utils.MODELS_DIR", str(tmp_path)):
        cb.on_step_end(args or _args(), _state(total_minutes), TrainerControl(), model=model)


class TestMarkBanking:
    def test_crossing_saves_checkpoint_and_metadata_row(self, tmp_path):
        cb = _callback(tmp_path, marks=[2.0, 5.0])
        model = DummyModel()
        _run_step(cb, model, total_minutes=3.0, tmp_path=tmp_path)

        assert len(model.saved) == 1
        rows = [json.loads(l) for l in (tmp_path / "marks.json").read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["threshold_type"] == "total_minutes"
        assert rows[0]["threshold_value"] == 2.0
        assert rows[0]["total_minutes"] == 3.0
        assert "budget-2m" in rows[0]["checkpoint_path"]

    def test_uncrossed_marks_do_nothing(self, tmp_path):
        cb = _callback(tmp_path, marks=[5.0])
        model = DummyModel()
        _run_step(cb, model, total_minutes=3.0, tmp_path=tmp_path)
        assert model.saved == []
        assert not (tmp_path / "marks.json").exists()

    def test_each_mark_fires_once(self, tmp_path):
        cb = _callback(tmp_path, marks=[2.0])
        model = DummyModel()
        _run_step(cb, model, total_minutes=3.0, tmp_path=tmp_path)
        _run_step(cb, model, total_minutes=4.0, tmp_path=tmp_path)
        assert len(model.saved) == 1

    def test_eval_only_mark_is_pre_claimed(self, tmp_path):
        cb = _callback(tmp_path, marks=[2.0], eval_only=[2.0])
        _run_step(cb, DummyModel(), total_minutes=3.0, tmp_path=tmp_path)
        row = json.loads((tmp_path / "marks.json").read_text().splitlines()[0])
        assert row["eval_only"] is True
        assert row["claimed"] is True


class TestResume:
    def test_resume_drops_marks_at_or_below_starting_minutes(self, tmp_path):
        cb = _callback(tmp_path, marks=[2.0, 5.0])
        cb.on_train_begin(_args(), _state(total_minutes=3.0), TrainerControl())
        model = DummyModel()
        _run_step(cb, model, total_minutes=3.5, tmp_path=tmp_path)
        assert model.saved == []
        _run_step(cb, model, total_minutes=6.0, tmp_path=tmp_path)
        assert len(model.saved) == 1

    def test_fresh_start_keeps_all_marks(self, tmp_path):
        cb = _callback(tmp_path, marks=[2.0, 5.0])
        cb.on_train_begin(_args(), _state(total_minutes=0.0), TrainerControl())
        model = DummyModel()
        _run_step(cb, model, total_minutes=2.5, tmp_path=tmp_path)
        assert len(model.saved) == 1


class TestFSDPStateDict:
    def test_gathered_state_dict_is_cast_to_bf16_and_passed_to_save(self, tmp_path):
        cb = _callback(tmp_path, marks=[2.0])
        model = DummyModel()
        gathered = {"w": torch.ones(2, dtype=torch.float32), "step": torch.tensor(3)}

        class DummyAccelerator:
            def get_state_dict(self, m):
                return gathered

            def unwrap_model(self, m):
                return m

            def wait_for_everyone(self):
                pass

        cb.set_trainer(SimpleNamespace(accelerator=DummyAccelerator()))
        _run_step(cb, model, total_minutes=3.0, tmp_path=tmp_path)

        saved_sd = model.saved[0]["state_dict"]
        assert saved_sd["w"].dtype == torch.bfloat16
        assert saved_sd["step"].dtype == torch.int64
