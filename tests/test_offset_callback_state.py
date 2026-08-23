# ABOUTME: Tests that OffsetAwareWandbCallback survives HuggingFace's checkpoint rebuild.
# ABOUTME: Trainer reconstructs stateful callbacks via type(cb)(**state()["args"]).

from types import SimpleNamespace
from unittest.mock import patch

from transformers import TrainerControl, TrainerState

from tuning.training.callback_utils import OffsetAwareWandbCallback


def _hf_rebuild(state_dict):
    """Mirror transformers.Trainer._load_callback_state's reconstruction."""
    rebuilt = OffsetAwareWandbCallback(**state_dict["args"])
    for attribute, value in state_dict["attributes"].items():
        setattr(rebuilt, attribute, value)
    rebuilt._wandb = None
    return rebuilt


def _resume(callback, stored_state, gpu_minute_multiplier):
    state = TrainerState(
        global_step=8,
        stateful_callbacks={"OffsetAwareWandbCallback": stored_state},
    )
    callback.on_train_begin(
        SimpleNamespace(gpu_minute_multiplier=gpu_minute_multiplier),
        state,
        TrainerControl(),
    )
    return state


def _fire_step(callback, state, seconds):
    with patch(
        "tuning.training.callback_utils.time.perf_counter",
        side_effect=[0.0, seconds],
    ):
        callback.on_step_begin(SimpleNamespace(), state, TrainerControl())
        callback.on_step_end(SimpleNamespace(), state, TrainerControl())


def test_rebuild_restores_gpu_minute_multiplier_from_training_args():
    original = OffsetAwareWandbCallback(initial_global_step=8, time_multiplier=2.0)
    original.total_seconds = 1200.0
    stored_state = original.state()

    rebuilt = _hf_rebuild(stored_state)
    assert rebuilt.time_multiplier == 1.0, "the rebuild is what drops the multiplier"

    _resume(rebuilt, stored_state, gpu_minute_multiplier=2.0)

    assert rebuilt.time_multiplier == 2.0
    assert rebuilt.total_seconds == 1200.0
    assert rebuilt.step_offset == 8


def test_resumed_steps_bank_gpu_minutes_not_wall_clock():
    """A checkpoint written before the multiplier existed still resumes at N GPUs."""
    legacy_state = {
        "args": {"initial_global_step": 0},
        "attributes": {"total_seconds": 1200.0},
    }
    rebuilt = _hf_rebuild(legacy_state)

    state = _resume(rebuilt, legacy_state, gpu_minute_multiplier=2.0)
    _fire_step(rebuilt, state, seconds=30.0)

    assert rebuilt.total_seconds == 1260.0
