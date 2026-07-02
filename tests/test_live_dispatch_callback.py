# ABOUTME: Tests for --live-dispatch GRPO worker submission from the SFT pass@k callback.
# ABOUTME: Verifies the callback submits one worker per saved checkpoint, gated on flags and rank.

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Mock heavy imports before importing the callback module.
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from transformers import TrainerControl, TrainerState

from tuning.training.config_training import PassAtKConfig
from tuning.training.passk_callback import PassAtKStoppingCallback


class _Eval:
    def __init__(self, metric_name="pass_at_1"):
        self._metric_name = metric_name
        self.stopping_k = 1

    @property
    def id(self):
        return "ifeval"

    @property
    def n_samples(self):
        return 1

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


def _make_callback(pipeline_args=None):
    tokenizer = MagicMock()
    tokenizer.chat_template = "test_template"
    tokenizer.apply_chat_template.return_value = "formatted_prompt"
    config = PassAtKConfig(
        target_pass_at_k=[0.95],
        enabled=True,
        use_persistent_vllm=False,
        num_inference_gpus=1,
    )
    callback = PassAtKStoppingCallback(
        config=config,
        tokenizer=tokenizer,
        model_name="test-model",
        base_model_hf="test/model",
        primary_eval=_Eval(),
        pipeline_args=pipeline_args,
    )
    callback.metadata_path = "/tmp/meta.jsonl"
    return callback


def _grpo_args(**overrides):
    base = dict(
        live_dispatch=True,
        post_training_method="grpo",
        grpo_num_gpus=1,
        sbatch_script="tuning/slurm/unified_early_pipeline.sh",
        long=False,
        short=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _drive_on_evaluate(callback, eval_only=False, metadata_type=None,
                       metadata_value=None):
    """Run on_evaluate with one forced checkpoint decision, return the submit mock."""
    decision = SimpleNamespace(label="0.3", advances_state=True, eval_only=eval_only,
                               metadata_type=metadata_type,
                               metadata_value=metadata_value)
    callback._eval_and_log = MagicMock(return_value={"pass_at_1": 0.5})
    callback._decision_engine.decide = MagicMock(return_value=[decision])
    callback._save_decision_checkpoint = MagicMock(return_value="/models/cp_live")

    args = SimpleNamespace(per_device_train_batch_size=2,
                           gradient_accumulation_steps=1, world_size=1)
    state = TrainerState()
    state.global_step = 5
    control = TrainerControl()

    with patch(
        "tuning.training.pipeline.orchestrator.submit_post_training_worker_for_metadata"
    ) as submit:
        callback.on_evaluate(args, state, control, model=MagicMock())
    return submit


def test_no_live_dispatch_without_pipeline_args():
    callback = _make_callback(pipeline_args=None)
    submit = _drive_on_evaluate(callback)
    submit.assert_not_called()


def test_no_live_dispatch_when_flag_disabled():
    callback = _make_callback(pipeline_args=_grpo_args(live_dispatch=False))
    submit = _drive_on_evaluate(callback)
    submit.assert_not_called()


def test_no_live_dispatch_for_non_grpo_method():
    callback = _make_callback(pipeline_args=_grpo_args(post_training_method="dpo"))
    submit = _drive_on_evaluate(callback)
    submit.assert_not_called()


def test_live_dispatch_submits_after_save():
    args = _grpo_args()
    callback = _make_callback(pipeline_args=args)
    submit = _drive_on_evaluate(callback)
    submit.assert_called_once_with(args, "/tmp/meta.jsonl", sft_total_minutes=None,
                                   checkpoint_path="/models/cp_live")


def test_live_dispatch_forwards_total_minutes_target():
    args = _grpo_args()
    callback = _make_callback(pipeline_args=args)
    submit = _drive_on_evaluate(callback, metadata_type="total_minutes",
                                metadata_value=240.0)
    submit.assert_called_once_with(args, "/tmp/meta.jsonl", sft_total_minutes=240.0,
                                   checkpoint_path="/models/cp_live")


def test_no_live_dispatch_for_eval_only_checkpoint():
    callback = _make_callback(pipeline_args=_grpo_args())
    submit = _drive_on_evaluate(callback, eval_only=True)
    submit.assert_not_called()
