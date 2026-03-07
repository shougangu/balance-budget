"""Tests for per-eval-step raw generation logging to W&B tables."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Mock heavy imports before importing callback module
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from transformers import TrainerControl, TrainerState

from tuning.training.config_training import PassAtKConfig
from tuning.training.passk_callback import PassAtKStoppingCallback


class _BaseEval:
    def __init__(self, metric_name="pass_at_1"):
        self._metric_name = metric_name
        self._n_samples = 2
        self.stopping_k = 1

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def label_prefix(self):
        return "p@1"

    def get_test_messages(self):
        return [[{"role": "user", "content": "Prompt A"}]]

    def get_test_prompts(self):
        return ["Prompt A"]

    def score_responses(self, results, tokenizer):
        return {self._metric_name: 0.0}

    def stopping_metric(self):
        return self._metric_name

    def wandb_metrics(self, scores):
        return {f"eval/{self._metric_name}": scores[self._metric_name]}


class IFEvalStrategy(_BaseEval):
    pass


class GSM8KEvalStrategy(_BaseEval):
    @property
    def label_prefix(self):
        return "gsm8k-p@1"


class FakeTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *args):
        self.rows.append(args)


def _make_callback(monitor_evals=None):
    tokenizer = MagicMock()
    tokenizer.chat_template = "test_template"
    tokenizer.apply_chat_template.return_value = "formatted_prompt"
    tokenizer.save_pretrained = MagicMock()

    config = PassAtKConfig(
        target_pass_at_k=[0.95],
        enabled=True,
        use_persistent_vllm=False,
        num_inference_gpus=1,
    )
    return PassAtKStoppingCallback(
        config=config,
        tokenizer=tokenizer,
        model_name="test-model",
        base_model_hf="test/model",
        primary_eval=IFEvalStrategy(),
        monitor_evals=monitor_evals or [],
    )


def _patch_eval_runs(callback, primary_payload, monitor_payload=None):
    monitor_payload = monitor_payload or primary_payload

    def run_with_results(model, eval_strategy):
        if isinstance(eval_strategy, IFEvalStrategy):
            return primary_payload
        return monitor_payload

    callback._run_eval_with_results = MagicMock(side_effect=run_with_results)
    callback._run_eval = MagicMock(side_effect=lambda m, e: run_with_results(m, e)[0])


def test_on_evaluate_logs_primary_raw_generation_table():
    callback = _make_callback()
    primary_scores = {
        "pass_at_1": 0.42,
        "num_prompts_evaluated": 1,
        "avg_response_length_tokens": 8.0,
    }
    primary_results = [{"prompt": "Prompt A", "responses": ["resp1", "resp2"]}]
    _patch_eval_runs(callback, (primary_scores, primary_results))

    args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
    state = TrainerState()
    state.global_step = 5
    control = TrainerControl()

    with patch("tuning.training.passk_callback.wandb.Table", side_effect=FakeTable), \
         patch("tuning.training.passk_callback.wandb.log") as mock_log:
        callback.on_evaluate(args, state, control, model=MagicMock())

    payloads = [call.args[0] for call in mock_log.call_args_list]
    table_payloads = [
        d for d in payloads if any(k.startswith("eval/raw_generations/") for k in d.keys())
    ]
    assert len(table_payloads) == 1

    key = next(k for k in table_payloads[0].keys() if k.startswith("eval/raw_generations/"))
    assert key == "eval/raw_generations/ifeval/step_5"

    table = table_payloads[0][key]
    assert table.columns == [
        "global_step",
        "eval_name",
        "prompt_index",
        "prompt",
        "responses",
        "num_responses",
        "stopping_metric_name",
        "stopping_metric_value",
        "thresholds_remaining",
        "timestamp_utc",
    ]
    assert len(table.rows) == 1
    row = table.rows[0]
    assert row[0] == 5
    assert row[1] == "ifeval"
    assert row[2] == 0
    assert row[3] == "Prompt A"
    assert json.loads(row[4]) == ["resp1", "resp2"]
    assert row[5] == 2
    assert row[6] == "pass_at_1"
    assert row[7] == 0.42


def test_on_evaluate_logs_monitor_raw_generation_table():
    callback = _make_callback(monitor_evals=[GSM8KEvalStrategy()])

    primary_scores = {"pass_at_1": 0.31, "num_prompts_evaluated": 1}
    primary_results = [{"prompt": "Prompt A", "responses": ["primary"]}]
    monitor_scores = {"pass_at_1": 0.62, "num_prompts_evaluated": 1}
    monitor_results = [{"prompt": "Prompt A", "responses": ["monitor"]}]
    _patch_eval_runs(callback, (primary_scores, primary_results), (monitor_scores, monitor_results))

    args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
    state = TrainerState()
    state.global_step = 7
    control = TrainerControl()

    with patch("tuning.training.passk_callback.wandb.Table", side_effect=FakeTable), \
         patch("tuning.training.passk_callback.wandb.log") as mock_log:
        callback.on_evaluate(args, state, control, model=MagicMock())

    payloads = [call.args[0] for call in mock_log.call_args_list]
    table_keys = []
    for payload in payloads:
        table_keys.extend([k for k in payload.keys() if k.startswith("eval/raw_generations/")])

    assert "eval/raw_generations/ifeval/step_7" in table_keys
    assert "eval/raw_generations/gsm8k/step_7" in table_keys


def test_table_logging_failure_is_non_fatal():
    callback = _make_callback()
    primary_scores = {"pass_at_1": 0.5, "num_prompts_evaluated": 1}
    primary_results = [{"prompt": "Prompt A", "responses": ["resp"]}]
    _patch_eval_runs(callback, (primary_scores, primary_results))

    args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=1, world_size=1)
    state = TrainerState()
    state.global_step = 9
    control = TrainerControl()

    with patch("tuning.training.passk_callback.wandb.Table", side_effect=RuntimeError("table boom")) as mock_table, \
         patch("tuning.training.passk_callback.wandb.log") as mock_log:
        callback.on_evaluate(args, state, control, model=MagicMock())

    assert mock_table.called
    metric_logs = [d for d in (c.args[0] for c in mock_log.call_args_list) if "eval/pass_at_1" in d]
    assert len(metric_logs) == 1
