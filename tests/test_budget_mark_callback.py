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
        import os
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as fh:
            json.dump({"torch_dtype": "float32"}, fh)
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
    control = TrainerControl()
    with patch("tuning.training.callback_utils.MODELS_DIR", str(tmp_path)):
        cb.on_step_end(args or _args(), _state(total_minutes), control, model=model)
    return control


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


class TestStopAfterLastMark:
    def test_training_stops_once_the_last_mark_is_banked(self, tmp_path):
        """Nothing past the largest mark is ever evaluated; training on is wasted GPU time."""
        cb = _callback(tmp_path, marks=[2.0, 5.0])
        model = DummyModel()
        assert _run_step(cb, model, total_minutes=3.0, tmp_path=tmp_path).should_training_stop is False
        assert _run_step(cb, model, total_minutes=6.0, tmp_path=tmp_path).should_training_stop is True
        assert len(model.saved) == 2

    def test_resumed_past_every_mark_stops_at_once(self, tmp_path):
        cb = _callback(tmp_path, marks=[2.0])
        cb.on_train_begin(_args(), _state(total_minutes=3.0), TrainerControl())
        control = _run_step(cb, DummyModel(), total_minutes=3.5, tmp_path=tmp_path)
        assert control.should_training_stop is True


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


class TestLiveDispatch:
    def _pipeline_args(self):
        return SimpleNamespace(
            live_dispatch=True,
            verl_num_gpus=8,
            verl_gpu_type="h100",
            verl_config="tuning/verl/configs/qwen3_14b_grpo.yaml",
            wandb_project="longcot",
            qos=None,
        )

    def _callback(self, tmp_path, mark, budget_rows):
        from tuning.training.budget_marks import BudgetMarkCallback

        return BudgetMarkCallback(
            model_name="qwen3-8B",
            tokenizer=DummyTokenizer(),
            target_total_minutes=[mark],
            metadata_path=str(tmp_path / "marks.json"),
            pipeline_args=self._pipeline_args(),
            budget_rows=budget_rows,
        )

    def _dispatches(self, tmp_path, mark, budget_rows):
        cb = self._callback(tmp_path, mark, budget_rows)
        with patch("tuning.training.pipeline.orchestrator."
                   "submit_verl_worker_for_metadata") as submit:
            _run_step(cb, DummyModel(), total_minutes=mark + 1, tmp_path=tmp_path)
        return [call.kwargs for call in submit.call_args_list]

    def test_shared_mark_dispatches_one_worker_to_its_largest_row(self, tmp_path):
        """7680 is 50% of 15360 and 25% of 30720: one run to 30720 banking at 15360."""
        calls = self._dispatches(tmp_path, mark=7680, budget_rows=[15360, 30720, 61440])
        assert len(calls) == 1
        assert calls[0]["budget_minutes"] == 30720
        assert calls[0]["bank_at"] == [15360]
        assert calls[0]["sft_total_minutes"] == 7680

    def test_mark_serving_three_rows_banks_every_smaller_row(self, tmp_path):
        calls = self._dispatches(tmp_path, mark=15360, budget_rows=[15360, 30720, 61440])
        assert len(calls) == 1
        assert calls[0]["budget_minutes"] == 61440
        assert calls[0]["bank_at"] == [30720]

    def test_off_grid_rows_are_not_run_to(self, tmp_path):
        """3840 is a quarter of 15360 only; it must not run on to 61440."""
        calls = self._dispatches(tmp_path, mark=3840, budget_rows=[15360, 30720, 61440])
        assert len(calls) == 1
        assert calls[0]["budget_minutes"] == 15360
        assert calls[0]["bank_at"] == []

    def test_terminal_mark_dispatches_nothing(self, tmp_path):
        calls = self._dispatches(tmp_path, mark=61440, budget_rows=[15360, 30720, 61440])
        assert calls == []


def test_design_rows_served_by_a_mark():
    from tuning.training.budget_marks import rows_served_by

    rows = [15360, 30720, 61440]
    assert rows_served_by(3840, rows) == [15360]
    assert rows_served_by(7680, rows) == [15360, 30720]
    assert rows_served_by(11520, rows) == [15360]
    assert rows_served_by(15360, rows) == [30720, 61440]
    assert rows_served_by(23040, rows) == [30720]
    assert rows_served_by(30720, rows) == [61440]
    assert rows_served_by(46080, rows) == [61440]
    assert rows_served_by(61440, rows) == []


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
        config = json.loads(open(model.saved[0]["path"] + "/config.json").read())
        assert config["torch_dtype"] == "bfloat16"


class TestResumeSharesTheMetadataFile:
    def test_metadata_path_is_keyed_by_the_wandb_run(self, tmp_path):
        """Every ladder leg of one run appends to the same file, so a resumed leg
        can see which marks are already banked."""
        from tuning.training.budget_marks import budget_marks_metadata_path

        path = budget_marks_metadata_path("qwen3-8B", "ab12cd34")
        assert path.endswith("qwen3-8B_budget-marks_ab12cd34.json")

    def test_already_banked_marks_do_not_refire_after_resume(self, tmp_path):
        """Resuming from a checkpoint saved before a banked mark replays that
        stretch of training; the mark must not bank (or dispatch) twice."""
        cb = _callback(tmp_path, marks=[4.0, 8.0, 12.0])
        (tmp_path / "marks.json").write_text(json.dumps({
            "checkpoint_path": "/ckpt/8m", "threshold_type": "total_minutes",
            "threshold_value": 8.0, "sft_wandb_run_id": "ab12cd34",
        }) + "\n")
        cb.on_train_begin(_args(), _state(total_minutes=6.0), TrainerControl())
        assert cb._engine.target_total_minutes == [12.0]

    def test_rl_rows_never_hide_an_sft_mark(self, tmp_path):
        """An RL worker banking at a cell budget writes the same threshold value
        an SFT mark can carry; only SFT-banked rows count."""
        cb = _callback(tmp_path, marks=[8.0, 12.0])
        (tmp_path / "marks.json").write_text(json.dumps({
            "checkpoint_path": "/ckpt/rl-8m", "threshold_type": "total_minutes",
            "threshold_value": 8.0, "rl_wandb_run_id": "rlrun",
        }) + "\n")
        cb.on_train_begin(_args(), _state(total_minutes=6.0), TrainerControl())
        assert cb._engine.target_total_minutes == [8.0, 12.0]


class TestSavedDtype:
    def test_bank_declares_the_dtype_it_wrote(self, tmp_path):
        """fp32 masters are cast to bf16 for disk; config.json must say so or
        the serving stack picks its precision from the master dtype."""
        from tuning.training.callback_utils import record_saved_dtype

        checkpoint = tmp_path / "ckpt"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps({"torch_dtype": "float32", "x": 1}))
        record_saved_dtype(str(checkpoint), {"w": torch.ones(2, dtype=torch.bfloat16), "i": torch.tensor(1)})
        config = json.loads((checkpoint / "config.json").read_text())
        assert config == {"torch_dtype": "bfloat16", "x": 1}
