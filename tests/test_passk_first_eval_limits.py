# ABOUTME: Tests that the baseline (step-0) pass@k eval lowers max_tokens and prompt
# ABOUTME: count, since the pre-training base model never emits EOS and runs to the cap.

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Mock heavy imports before importing callback modules
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from transformers import TrainerControl, TrainerState

from tuning.training.config_training import PassAtKConfig
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.training.passk.callback import (
    FIRST_EVAL_MAX_PROMPTS,
    FIRST_EVAL_MAX_TOKENS,
)


class _FakeEval:
    def __init__(self):
        self._n_samples = 8
        self.stopping_k = 1
        self.prompt_limit = None

    @property
    def id(self):
        return "math500"

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
        return {"pass_at_1": 0.0, "num_prompts_evaluated": 1,
                "avg_response_length_tokens": 8.0}

    def stopping_metric(self):
        return "pass_at_1"

    def wandb_metrics(self, scores):
        return {"eval/pass_at_1": scores["pass_at_1"]}


def _make_callback(initial_global_step=0):
    tokenizer = MagicMock()
    tokenizer.chat_template = "test_template"
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


def test_first_eval_limits_active_sets_and_restores():
    callback = _make_callback()
    full_max_tokens = callback.max_tokens
    strategies = [callback.primary_eval, *callback.monitor_evals]

    with callback._first_eval_limits(active=True):
        assert callback.max_tokens == FIRST_EVAL_MAX_TOKENS
        assert callback._runner_config.max_tokens == FIRST_EVAL_MAX_TOKENS
        for strategy in strategies:
            assert strategy.prompt_limit == FIRST_EVAL_MAX_PROMPTS

    assert callback.max_tokens == full_max_tokens
    assert callback._runner_config.max_tokens == full_max_tokens
    for strategy in strategies:
        assert strategy.prompt_limit is None


def test_first_eval_limits_inactive_is_noop():
    callback = _make_callback()
    full_max_tokens = callback.max_tokens

    with callback._first_eval_limits(active=False):
        assert callback.max_tokens == full_max_tokens
        assert callback.primary_eval.prompt_limit is None


def _observe_on_evaluate(callback, state):
    observed = []

    def fake_run(model, eval_strategy):
        observed.append((callback.max_tokens, eval_strategy.prompt_limit))
        return ({"pass_at_1": 0.0, "num_prompts_evaluated": 1,
                 "avg_response_length_tokens": 8.0},
                [{"prompt": "P", "responses": ["r"]}])

    callback._run_eval_with_results = MagicMock(side_effect=fake_run)
    args = SimpleNamespace(per_device_train_batch_size=2,
                           gradient_accumulation_steps=1, world_size=1)
    with patch("tuning.training.passk.callback.log_eval_metrics"):
        callback.on_evaluate(args, state, TrainerControl(), model=MagicMock())
    return observed


def test_on_evaluate_caps_baseline_step_zero():
    callback = _make_callback()
    state = TrainerState(global_step=0)
    # Baseline eval (untrained base model): primary + monitor both capped.
    assert _observe_on_evaluate(callback, state) == [
        (FIRST_EVAL_MAX_TOKENS, FIRST_EVAL_MAX_PROMPTS),
        (FIRST_EVAL_MAX_TOKENS, FIRST_EVAL_MAX_PROMPTS),
    ]


def test_on_evaluate_full_budget_after_training_starts():
    callback = _make_callback()
    full_max_tokens = callback.max_tokens
    state = TrainerState(global_step=64)
    # Model has trained: full budget, no cap.
    assert _observe_on_evaluate(callback, state) == [
        (full_max_tokens, None),
        (full_max_tokens, None),
    ]


def test_on_evaluate_full_budget_when_resumed_from_checkpoint():
    # Chained/resumed run: step_offset > 0 means the model is already trained,
    # so even the first eval of this process must not be capped.
    callback = _make_callback(initial_global_step=100)
    full_max_tokens = callback.max_tokens
    state = TrainerState(global_step=0)
    assert _observe_on_evaluate(callback, state) == [
        (full_max_tokens, None),
        (full_max_tokens, None),
    ]
