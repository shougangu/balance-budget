# ABOUTME: Tests that CompletionsIntervalCallback gates trainer.log_completions
# ABOUTME: to fire only every N training steps.

from types import SimpleNamespace

from transformers import TrainerControl, TrainerState


class TestCompletionsIntervalCallback:
    def _make_callback_and_trainer(self, interval):
        from tuning.training.callback_utils import CompletionsIntervalCallback

        trainer = SimpleNamespace(log_completions=True)
        cb = CompletionsIntervalCallback(trainer=trainer, interval=interval)
        return cb, trainer

    def _fire_on_log(self, cb, global_step):
        state = TrainerState()
        state.global_step = global_step
        args = SimpleNamespace()
        control = TrainerControl()
        cb.on_log(args, state, control)

    def test_enables_at_interval_steps(self):
        cb, trainer = self._make_callback_and_trainer(interval=10)
        for step in [0, 10, 20, 50, 100]:
            self._fire_on_log(cb, step)
            assert trainer.log_completions is True, f"Expected True at step {step}"

    def test_disables_between_interval_steps(self):
        cb, trainer = self._make_callback_and_trainer(interval=10)
        for step in [1, 3, 7, 11, 19, 99]:
            self._fire_on_log(cb, step)
            assert trainer.log_completions is False, f"Expected False at step {step}"

    def test_interval_1_always_enables(self):
        cb, trainer = self._make_callback_and_trainer(interval=1)
        for step in range(10):
            self._fire_on_log(cb, step)
            assert trainer.log_completions is True, f"Expected True at step {step}"
