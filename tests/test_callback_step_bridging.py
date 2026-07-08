# ABOUTME: Tests training-time state plus step/time fields on W&B eval rows.
# ABOUTME: Covers resume, missing checkpoint state, and chained-run offsets.

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Mock heavy imports before importing callback modules
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from transformers import TrainerControl, TrainerState
from transformers.trainer_callback import ExportableState

from tuning.training.callback_utils import (
    OffsetAwareWandbCallback,
    load_total_seconds_from_checkpoint,
    save_trainer_state,
)
from tuning.training.config_training import PassAtKConfig, PerplexityConfig
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.training.perplexity_callback import PerplexityStoppingCallback


class _FakeEval:
    def __init__(self):
        self._n_samples = 1
        self.stopping_k = 1

    @property
    def id(self):
        return "ifeval"

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def label_prefix(self):
        return "p@1"

    def get_test_messages(self):
        return [[{"role": "user", "content": "P"}]]

    def get_test_prompts(self):
        return ["P"]

    def score_responses(self, results, tokenizer):
        return {"pass_at_1": 0.0}

    def stopping_metric(self):
        return "pass_at_1"

    def wandb_metrics(self, scores):
        return {"eval/pass_at_1": scores["pass_at_1"]}


class _FakeTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *args):
        self.rows.append(args)


def _make_passk_callback(initial_global_step=0):
    tokenizer = MagicMock()
    tokenizer.chat_template = "test_template"
    tokenizer.apply_chat_template.return_value = "formatted_prompt"
    tokenizer.save_pretrained = MagicMock()

    config = PassAtKConfig(
        target_pass_at_k=[0.95],
        enabled=True,
        use_persistent_vllm=False,
        num_inference_gpus=1,
        initial_global_step=initial_global_step,
    )
    return PassAtKStoppingCallback(
        config=config,
        tokenizer=tokenizer,
        model_name="test-model",
        base_model_hf="test/model",
        primary_eval=_FakeEval(),
        monitor_evals=[_FakeEval()],
    )


def _make_passk_callback_with_minutes(target_total_minutes, initial_global_step=0):
    tokenizer = MagicMock()
    tokenizer.chat_template = "test_template"
    tokenizer.apply_chat_template.return_value = "formatted_prompt"
    tokenizer.save_pretrained = MagicMock()

    config = PassAtKConfig(
        target_pass_at_k=[0.95],
        enabled=True,
        use_persistent_vllm=False,
        num_inference_gpus=1,
        initial_global_step=initial_global_step,
        target_total_minutes=target_total_minutes,
    )
    return PassAtKStoppingCallback(
        config=config,
        tokenizer=tokenizer,
        model_name="test-model",
        base_model_hf="test/model",
        primary_eval=_FakeEval(),
    )


def _patch_eval_runs(callback, scores, results):
    def run_with_results(model, eval_strategy):
        return (scores, results)
    callback._run_eval_with_results = MagicMock(side_effect=run_with_results)
    callback._run_eval = MagicMock(side_effect=lambda m, e: scores)


def _make_perplexity_callback(initial_global_step=0):
    tokenizer = MagicMock()
    config = PerplexityConfig(
        perplexity_thresholds=[1.0],
        num_samples=1,
        enabled=True,
        initial_global_step=initial_global_step,
    )
    test_dataset = [{"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]}]
    return PerplexityStoppingCallback(
        config=config,
        test_dataset=test_dataset,
        tokenizer=tokenizer,
        model_name="test-model",
    )


# ---------------------------------------------------------------------------
# PassAtK step bridging
# ---------------------------------------------------------------------------


class TestPassAtKStepBridging:
    def test_on_evaluate_injects_total_global_step_when_offset_set(self):
        callback = _make_passk_callback(initial_global_step=100)
        scores = {"pass_at_1": 0.42, "num_prompts_evaluated": 1, "avg_response_length_tokens": 8.0}
        results = [{"prompt": "P", "responses": ["r"]}]
        _patch_eval_runs(callback, scores, results)

        args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
        state = TrainerState(stateful_callbacks={
            "OffsetAwareWandbCallback": OffsetAwareWandbCallback(
                initial_total_seconds=120.0,
            ).state(),
        })
        state.global_step = 5
        control = TrainerControl()

        with patch("tuning.training.passk.logging.wandb.Table", side_effect=_FakeTable), \
             patch("tuning.training.passk.logging.wandb.log") as mock_log:
            callback.on_evaluate(args, state, control, model=MagicMock())

        payloads = [call.args[0] for call in mock_log.call_args_list]
        assert payloads, "wandb.log must have been called at least once"
        for payload in payloads:
            assert "train/total_global_step" in payload, (
                f"Expected train/total_global_step in every log dict, missing from: {payload}"
            )
            assert payload["train/total_global_step"] == 105
            assert payload["train/total_minutes"] == 2.0

    def test_on_evaluate_includes_total_global_step_when_offset_zero(self):
        callback = _make_passk_callback(initial_global_step=0)
        scores = {"pass_at_1": 0.42, "num_prompts_evaluated": 1}
        results = [{"prompt": "P", "responses": ["r"]}]
        _patch_eval_runs(callback, scores, results)

        args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
        state = TrainerState()
        state.global_step = 5
        control = TrainerControl()

        with patch("tuning.training.passk.logging.wandb.Table", side_effect=_FakeTable), \
             patch("tuning.training.passk.logging.wandb.log") as mock_log:
            callback.on_evaluate(args, state, control, model=MagicMock())

        payloads = [call.args[0] for call in mock_log.call_args_list]
        assert payloads
        for payload in payloads:
            assert payload["train/total_global_step"] == 5
            assert payload["train/total_minutes"] == 0.0

    def test_raw_generation_table_log_includes_total_step_when_offset_set(self):
        callback = _make_passk_callback(initial_global_step=100)
        scores = {"pass_at_1": 0.42, "num_prompts_evaluated": 1}
        results = [{"prompt": "P", "responses": ["r"]}]
        _patch_eval_runs(callback, scores, results)

        args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
        state = TrainerState()
        state.global_step = 5
        control = TrainerControl()

        with patch("tuning.training.passk.logging.wandb.Table", side_effect=_FakeTable), \
             patch("tuning.training.passk.logging.wandb.log") as mock_log:
            callback.on_evaluate(args, state, control, model=MagicMock())

        table_payloads = [
            call.args[0]
            for call in mock_log.call_args_list
            if any(k.startswith("raw_generations/") for k in call.args[0])
        ]
        assert table_payloads, "Expected at least one raw-generation table log"
        for payload in table_payloads:
            assert payload.get("train/total_global_step") == 105
            assert payload.get("train/total_minutes") == 0.0


class TestChainedRunTargetFiltering:
    """A forked/resumed run inherits its parent's clock; budget targets already
    spent upstream must be dropped before the baseline eval so they don't emit a
    zero-progress (step-0) or duplicate (resume) checkpoint."""

    def _state_with_minutes(self, minutes):
        return TrainerState(stateful_callbacks={
            "OffsetAwareWandbCallback": OffsetAwareWandbCallback(
                initial_total_seconds=minutes * 60.0,
            ).state(),
        })

    def test_forked_run_drops_targets_at_or_below_inherited_clock(self):
        callback = _make_passk_callback_with_minutes([240.0, 480.0])
        callback.on_evaluate = MagicMock()
        state = self._state_with_minutes(240.0)
        state.global_step = 0

        callback.on_train_begin(SimpleNamespace(), state, TrainerControl(),
                                model=MagicMock())

        assert callback._decision_engine.target_total_minutes == [480.0]

    def test_sft_run_keeps_all_targets_including_zero_budget(self):
        callback = _make_passk_callback_with_minutes([0.0, 60.0, 240.0])
        callback.on_evaluate = MagicMock()
        state = TrainerState()  # no timing state -> starting clock 0
        state.global_step = 0

        callback.on_train_begin(SimpleNamespace(), state, TrainerControl(),
                                model=MagicMock())

        assert callback._decision_engine.target_total_minutes == [0.0, 60.0, 240.0]

    def test_resumed_run_drops_already_saved_target(self):
        callback = _make_passk_callback_with_minutes([240.0])
        callback.on_evaluate = MagicMock()
        state = self._state_with_minutes(943.0)
        state.global_step = 448

        callback.on_train_begin(SimpleNamespace(), state, TrainerControl(),
                                model=MagicMock())

        assert callback._decision_engine.target_total_minutes == []


class TestMinuteCrossingRankSynchronization:
    """Every rank must reach the same total_minutes crossing decision.

    total_minutes accumulates per-rank from wall-clock time, so the ranks' clocks
    drift apart. If one rank crosses a target a step before its peer, that rank
    enters eval while the other keeps training -> the two ranks issue different
    NCCL collectives and the job deadlocks. on_step_end broadcasts rank 0's clock
    so the crossing (and the target consumption) is identical on every rank.
    """

    def _run_on_step_end(self, callback, local_minutes, rank, rank0_minutes):
        control = TrainerControl()

        def fake_broadcast(payload, src=0):
            # rank 0 is the source and keeps its value; peers receive rank 0's.
            if rank != src:
                payload[0] = rank0_minutes

        with patch(
            "tuning.training.passk.callback.get_total_minutes_from_state",
            return_value=local_minutes,
        ), patch("torch.distributed.is_initialized", return_value=True), \
                patch("torch.distributed.get_world_size", return_value=2), \
                patch("torch.distributed.get_rank", return_value=rank), \
                patch("torch.distributed.broadcast_object_list",
                      side_effect=fake_broadcast):
            callback.on_step_end(SimpleNamespace(), SimpleNamespace(), control)
        return control

    def test_peer_clock_ahead_does_not_trigger_solo_eval(self):
        """rank 1's clock crossed 960 but rank 0's has not: neither evaluates."""
        cb0 = _make_passk_callback_with_minutes([960.0])
        control0 = self._run_on_step_end(cb0, 959.0, rank=0, rank0_minutes=959.0)

        cb1 = _make_passk_callback_with_minutes([960.0])
        control1 = self._run_on_step_end(cb1, 961.0, rank=1, rank0_minutes=959.0)

        assert control0.should_evaluate is False
        assert control1.should_evaluate is False
        assert cb0._decision_engine.target_total_minutes == [960.0]
        assert cb1._decision_engine.target_total_minutes == [960.0]

    def test_peer_clock_behind_still_follows_rank0_crossing(self):
        """rank 0's clock crossed 960 but rank 1's has not: both evaluate."""
        cb0 = _make_passk_callback_with_minutes([960.0])
        control0 = self._run_on_step_end(cb0, 961.0, rank=0, rank0_minutes=961.0)

        cb1 = _make_passk_callback_with_minutes([960.0])
        control1 = self._run_on_step_end(cb1, 959.0, rank=1, rank0_minutes=961.0)

        assert control0.should_evaluate is True
        assert control1.should_evaluate is True
        assert cb0._decision_engine.target_total_minutes == []
        assert cb1._decision_engine.target_total_minutes == []


# ---------------------------------------------------------------------------
# Perplexity step bridging
# ---------------------------------------------------------------------------


class TestPerplexityStepBridging:
    def test_on_evaluate_injects_total_global_step_when_offset_set(self):
        callback = _make_perplexity_callback(initial_global_step=100)
        callback.evaluate_perplexity = MagicMock(return_value=2.0)

        args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
        state = TrainerState(stateful_callbacks={
            "OffsetAwareWandbCallback": OffsetAwareWandbCallback(
                initial_total_seconds=120.0,
            ).state(),
        })
        state.global_step = 5
        control = TrainerControl()

        with patch("tuning.training.perplexity_callback.wandb.log") as mock_log:
            callback.on_evaluate(args, state, control, model=MagicMock())

        assert mock_log.call_args_list, "wandb.log must have been called"
        for call in mock_log.call_args_list:
            payload = call.args[0]
            assert "train/total_global_step" in payload
            assert payload["train/total_global_step"] == 105
            assert payload["train/total_minutes"] == 2.0

    def test_on_evaluate_includes_total_global_step_when_offset_zero(self):
        callback = _make_perplexity_callback(initial_global_step=0)
        callback.evaluate_perplexity = MagicMock(return_value=2.0)

        args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
        state = TrainerState()
        state.global_step = 5
        control = TrainerControl()

        with patch("tuning.training.perplexity_callback.wandb.log") as mock_log:
            callback.on_evaluate(args, state, control, model=MagicMock())

        for call in mock_log.call_args_list:
            payload = call.args[0]
            assert payload["train/total_global_step"] == 5
            assert payload["train/total_minutes"] == 0.0


# ---------------------------------------------------------------------------
# OffsetAwareWandbCallback state and step bridging
# ---------------------------------------------------------------------------


def _fire_on_log(cb, global_step, logs=None):
    """Invoke on_log without calling the real W&B integration."""
    state = TrainerState()
    state.global_step = global_step
    args = SimpleNamespace()
    control = TrainerControl()
    captured_logs = logs if logs is not None else {"train/loss": 0.5}
    with patch("transformers.integrations.WandbCallback.on_log"):
        cb.on_log(args, state, control, logs=captured_logs)
    return captured_logs, state


def _fire_step(cb, state, start, end):
    args = SimpleNamespace()
    control = TrainerControl()
    with patch(
        "tuning.training.callback_utils.time.perf_counter",
        side_effect=[start, end],
    ):
        cb.on_step_begin(args, state, control)
        cb.on_step_end(args, state, control)


class TestOffsetAwareWandbCallback:
    def test_exports_only_total_seconds(self):
        cb = OffsetAwareWandbCallback(
            initial_global_step=1000,
            initial_total_seconds=12.5,
        )

        assert isinstance(cb, ExportableState)
        assert cb.state() == {
            "args": {"initial_global_step": 1000},
            "attributes": {"total_seconds": 12.5},
        }
        assert "step_start" not in cb.state()["attributes"]

    def test_restored_state_continues_total_minutes(self):
        original = OffsetAwareWandbCallback(
            initial_global_step=1000,
            initial_total_seconds=12.0,
        )
        restored = OffsetAwareWandbCallback.from_state(original.state())
        state = TrainerState(global_step=20)

        _fire_step(restored, state, start=100.0, end=103.0)
        logs, _ = _fire_on_log(restored, global_step=20)

        assert logs["total_minutes"] == 15.0 / 60.0
        assert logs["total_global_step"] == 1020

    def test_completed_steps_add_elapsed_time(self):
        cb = OffsetAwareWandbCallback(initial_total_seconds=5.0)
        state = TrainerState(global_step=1)

        _fire_step(cb, state, start=10.0, end=12.5)

        assert cb.total_seconds == 7.5
        assert state.stateful_callbacks["OffsetAwareWandbCallback"] == cb.state()

    def test_time_multiplier_scales_elapsed_time(self):
        cb = OffsetAwareWandbCallback(initial_total_seconds=5.0, time_multiplier=2.0)
        state = TrainerState(global_step=1)

        _fire_step(cb, state, start=10.0, end=12.5)

        assert cb.total_seconds == 10.0

    def test_unfinished_step_is_not_counted_or_persisted(self):
        cb = OffsetAwareWandbCallback(initial_total_seconds=5.0)
        state = TrainerState(global_step=1)
        with patch("tuning.training.callback_utils.time.perf_counter", return_value=10.0):
            cb.on_step_begin(SimpleNamespace(), state, TrainerControl())

        logs, _ = _fire_on_log(cb, global_step=1)

        assert cb.total_seconds == 5.0
        assert logs["total_minutes"] == 5.0 / 60.0

    def test_missing_callback_state_starts_at_zero_and_becomes_saveable(self):
        cb = OffsetAwareWandbCallback(initial_global_step=1000)
        cb._wandb = None
        state = TrainerState(global_step=20, stateful_callbacks={})

        with pytest.warns(RuntimeWarning, match="resumed TrainerState"):
            cb.on_train_begin(SimpleNamespace(), state, TrainerControl())

        assert cb.total_seconds == 0.0
        assert state.stateful_callbacks["OffsetAwareWandbCallback"] == cb.state()

    def test_migrates_training_time_callback_state_on_resume(self):
        saved_state = {
            "args": {"initial_global_step": 1000},
            "attributes": {"total_seconds": 45.0},
        }
        cb = OffsetAwareWandbCallback(initial_global_step=0)
        cb._wandb = None
        state = TrainerState(
            global_step=20,
            stateful_callbacks={"TrainingTimeCallback": saved_state},
        )

        cb.on_train_begin(SimpleNamespace(), state, TrainerControl())

        assert cb.total_seconds == 45.0
        assert cb.step_offset == 1000
        assert "TrainingTimeCallback" not in state.stateful_callbacks
        assert state.stateful_callbacks["OffsetAwareWandbCallback"] == cb.state()

    def test_migrates_old_offset_cumulative_seconds_on_resume(self):
        saved_state = {
            "args": {"initial_global_step": 1000},
            "attributes": {"_cumulative_train_seconds": 30.0},
        }
        cb = OffsetAwareWandbCallback(initial_global_step=1000)
        cb._wandb = None
        state = TrainerState(
            global_step=20,
            stateful_callbacks={"OffsetAwareWandbCallback": saved_state},
        )

        cb.on_train_begin(SimpleNamespace(), state, TrainerControl())

        assert cb.total_seconds == 30.0
        assert state.stateful_callbacks["OffsetAwareWandbCallback"] == cb.state()

    def test_injects_total_global_step_and_total_minutes(self):
        cb = OffsetAwareWandbCallback(
            initial_global_step=500,
            initial_total_seconds=120.0,
        )
        logs, _ = _fire_on_log(cb, global_step=50)

        assert logs["total_global_step"] == 550
        assert logs["total_minutes"] == 2.0

    def test_no_crash_when_logs_is_none(self):
        cb = OffsetAwareWandbCallback(initial_global_step=100)
        state = TrainerState(global_step=5)
        with patch("transformers.integrations.WandbCallback.on_log"):
            cb.on_log(SimpleNamespace(), state, TrainerControl(), logs=None)


class TestTrainingTimeCheckpointLoading:
    def test_save_requires_trainer_state(self, tmp_path):
        with pytest.raises(TypeError, match="TrainerState"):
            save_trainer_state(SimpleNamespace(), str(tmp_path))

    def test_loads_total_seconds(self, tmp_path):
        state_path = tmp_path / "trainer_state.json"
        state_path.write_text(json.dumps({
            "stateful_callbacks": {
                "OffsetAwareWandbCallback": {
                    "args": {"initial_global_step": 100},
                    "attributes": {"total_seconds": 321.5},
                }
            }
        }))

        assert load_total_seconds_from_checkpoint(str(tmp_path)) == 321.5

    def test_loads_training_time_callback_state(self, tmp_path):
        (tmp_path / "trainer_state.json").write_text(json.dumps({
            "stateful_callbacks": {
                "TrainingTimeCallback": {
                    "args": {"initial_global_step": 100},
                    "attributes": {"total_seconds": 222.0},
                }
            }
        }))

        assert load_total_seconds_from_checkpoint(str(tmp_path)) == 222.0

    def test_missing_trainer_state_starts_at_zero(self, tmp_path):
        with pytest.warns(RuntimeWarning, match="starting train/total_minutes at 0"):
            total_seconds = load_total_seconds_from_checkpoint(str(tmp_path))

        assert total_seconds == 0.0

    def test_missing_timing_state_starts_at_zero(self, tmp_path):
        (tmp_path / "trainer_state.json").write_text(json.dumps({
            "global_step": 20,
            "stateful_callbacks": {},
        }))

        with pytest.warns(RuntimeWarning, match="No timing callback state"):
            total_seconds = load_total_seconds_from_checkpoint(str(tmp_path))

        assert total_seconds == 0.0

    def test_invalid_total_seconds_starts_at_zero(self, tmp_path):
        (tmp_path / "trainer_state.json").write_text(json.dumps({
            "stateful_callbacks": {
                "OffsetAwareWandbCallback": {
                    "args": {"initial_global_step": 0},
                    "attributes": {"total_seconds": "invalid"},
                }
            }
        }))

        with pytest.warns(RuntimeWarning, match="No valid total_seconds"):
            total_seconds = load_total_seconds_from_checkpoint(str(tmp_path))

        assert total_seconds == 0.0
