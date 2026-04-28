# ABOUTME: Tests that PassAtK and Perplexity callbacks inject train/total_global_step
# ABOUTME: into their direct wandb.log dicts when initial_global_step is set.

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
        state = TrainerState()
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


# ---------------------------------------------------------------------------
# Perplexity step bridging
# ---------------------------------------------------------------------------


class TestPerplexityStepBridging:
    def test_on_evaluate_injects_total_global_step_when_offset_set(self):
        callback = _make_perplexity_callback(initial_global_step=100)
        callback.evaluate_perplexity = MagicMock(return_value=2.0)

        args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
        state = TrainerState()
        state.global_step = 5
        control = TrainerControl()

        with patch("tuning.training.perplexity_callback.wandb.log") as mock_log:
            callback.on_evaluate(args, state, control, model=MagicMock())

        assert mock_log.call_args_list, "wandb.log must have been called"
        for call in mock_log.call_args_list:
            payload = call.args[0]
            assert "train/total_global_step" in payload
            assert payload["train/total_global_step"] == 105

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
