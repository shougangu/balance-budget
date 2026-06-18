# ABOUTME: Tests for DeepSeek judge parsing, aggregation, and callback orchestration.

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from transformers import TrainerState

from tuning.training.config_training import PassAtKConfig
from tuning.training.passk.callback import PassAtKStoppingCallback
from tuning.training.passk.judge import (
    aggregate_quality,
    build_judge_messages,
    parse_score,
)


class _FakeEval:
    def __init__(self):
        self._n_samples = 2
        self.stopping_k = 1

    @property
    def id(self):
        return "ifbench"

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def label_prefix(self):
        return "ifbench-p@1"

    def get_test_messages(self):
        return [[{"role": "user", "content": "Prompt A"}]]

    def get_test_prompts(self):
        return ["Prompt A"]

    def score_responses(self, results, tokenizer):
        return {"pass_at_1": 0.0}

    def stopping_metric(self):
        return "pass_at_1"

    def wandb_metrics(self, scores):
        return {"eval/ifbench_pass_at_1": scores["pass_at_1"]}


def _make_callback():
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
        primary_eval=_FakeEval(),
        monitor_evals=[],
    )


def test_build_judge_messages_contains_payload_and_json_instruction():
    messages = build_judge_messages("Write a haiku", "Old pond")

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Write a haiku" in messages[1]["content"]
    assert "Old pond" in messages[1]["content"]
    assert "JSON" in messages[0]["content"] + messages[1]["content"]
    assert "Score" in messages[0]["content"] + messages[1]["content"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"Score":"8"}', 8),
        ('{"Score":8}', 8),
        ('12', 10),
        ('0', 1),
        ("garbage", None),
    ],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


def test_aggregate_quality_excludes_failures_and_conditions_on_correctness():
    metrics = aggregate_quality(
        [8, None, 4, 10],
        [True, True, False, None],
        conditioned=True,
    )

    assert metrics["quality"] == pytest.approx((8 + 4 + 10) / 3)
    assert metrics["quality_when_correct"] == 8
    assert metrics["quality_when_incorrect"] == 4
    assert metrics["quality_n_judged"] == 3
    assert metrics["quality_n_correct"] == 1
    assert metrics["quality_n_incorrect"] == 1
    assert metrics["quality_n_judge_failures"] == 1


def test_callback_submits_and_logs_judge_job():
    callback = _make_callback()
    callback._judge_samples_per_prompt = 0
    class StubJudge:
        def __init__(self):
            self.pairs = None

        def score_pairs(self, pairs):
            self.pairs = list(pairs)
            return [8, 4, None]

    stub = StubJudge()
    callback._judge = stub
    callback._run_eval_with_results = MagicMock(return_value=(
        {"pass_at_1": 0.5, "num_prompts_evaluated": 2},
        [
            {
                "prompt": "P1",
                "responses": ["R1", "R2"],
                "per_response_correct": [True, False],
            },
            {
                "prompt": "P2",
                "responses": ["R3"],
                "per_response_correct": [True],
            },
        ],
    ))

    state = TrainerState()
    state.global_step = 12

    with patch("tuning.training.passk.callback.log_eval_metrics"), \
         patch("tuning.training.passk.callback.log_judge_quality") as mock_log:
        callback._eval_and_log(MagicMock(), _FakeEval(), state, is_primary=True)

    assert stub.pairs == [("P1", "R1"), ("P1", "R2"), ("P2", "R3")]
    payload = mock_log.call_args.kwargs
    assert payload["eval_id"] == "ifbench"
    assert payload["global_step"] == 12
    assert payload["metrics"]["quality"] == pytest.approx(6.0)
    assert payload["metrics"]["quality_when_correct"] == 8
    assert payload["metrics"]["quality_when_incorrect"] == 4
    assert payload["metrics"]["quality_n_judge_failures"] == 1
