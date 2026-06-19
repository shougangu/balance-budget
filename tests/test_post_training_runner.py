# ABOUTME: Verifies run_post_training claims a checkpoint, builds method-specific
# ABOUTME: configs, calls the right train_model_*, and marks completion.

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())


CHECKPOINT_ROW = {
    "checkpoint_path": "/models/cp_1",
    "data_points_seen": 1024,
    "threshold_value": 0.3,
    "threshold_type": "pass_at_1",
    "global_step": 64,
}


def _write_meta(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_args(metadata_file, **overrides):
    base = dict(
        model="llama3-3B",
        wandb_project="test",
        tags=[],
        dataset="gsm8k",
        train_size=4096,
        sft_data_size=None, dpo_data_size=None, grpo_data_size=None,
        task_name="gsm8k",
        monitor_evals=[],
        max_seq_length=1024,
        seed=42, eval_seed=None,
        simple_template=False,
        sft_enable_passk=False, sft_enable_ppl=False,
        dpo_enable_passk=False, dpo_enable_ppl=False,
        grpo_enable_passk=False,
        grpo_enable_judge=False,
        grpo_judge_model="deepseek-v4-flash",
        grpo_judge_base_url="https://api.deepseek.com",
        grpo_judge_api_key_env="DEEPSEEK_API_KEY",
        grpo_judge_samples_per_prompt=1,
        grpo_judge_concurrency=16,
        grpo_judge_timeout=60.0,
        grpo_judge_max_retries=3,
        grpo_judge_max_tokens=64,
        dpo_learning_rate=5e-6, dpo_num_epochs=2, dpo_eval_steps=64,
        dpo_batch_size=2, dpo_grad_accum=2,
        grpo_learning_rate=1e-5, grpo_num_epochs=1, grpo_eval_steps=64,
        grpo_batch_size=2, grpo_grad_accum=2, grpo_num_generations=2,
        grpo_num_iterations=1,
        grpo_max_completion_length=256, grpo_max_prompt_length=256,
        grpo_beta=0.0, grpo_temperature=1.0,
        grpo_warmup_ratio=0.05, grpo_lr_scheduler_type="constant",
        grpo_loss_type="dapo", grpo_scale_rewards="group",
        grpo_gpu_util=None,
        grpo_vllm_importance_sampling=True,
        grpo_vllm_sleep_mode=True,
        grpo_upcast_lm_head_fp32=False,
        grpo_precision="auto",
        grpo_use_liger_kernel=False,
        grpo_zero_variance_filter=True,
        grpo_zero_variance_filter_epsilon=0.0,
        grpo_vllm_mode="colocate",
        grpo_vllm_server_host="127.0.0.1",
        grpo_vllm_server_port=8000,
        grpo_vllm_group_port=51216,
        grpo_vllm_server_timeout=300.0,
        grpo_lora_target_modules=None, grpo_lora_layers_fraction=1.0,
        gradient_checkpointing=True,
        metadata_file=[str(metadata_file)],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestRunPostTrainingDpo:
    def test_exits_42_when_nothing_to_claim(self, tmp_path):
        from tuning.training.pipeline.stages import run_post_training
        f = tmp_path / "meta.jsonl"
        _write_meta(f, [{**CHECKPOINT_ROW, "completed": True}])
        args = _make_args(f)
        with pytest.raises(SystemExit) as exc:
            run_post_training(args, "dpo")
        assert exc.value.code == 42

    def test_skips_when_no_data_budget(self, tmp_path, monkeypatch):
        from tuning.training.pipeline import stages
        f = tmp_path / "meta.jsonl"
        _write_meta(f, [{**CHECKPOINT_ROW, "data_points_seen": 99999}])
        args = _make_args(f, train_size=1000)

        monkeypatch.setattr(stages, "_init_seeds",
                            lambda *a, **k: pytest.fail("budget guard did not skip"))

        stages.run_post_training(args, "dpo")
        rows = [json.loads(line) for line in open(f)]
        assert rows[0]["completed"] is True


class TestRunPostTrainingGrpo:
    def test_grpo_uses_grpo_dispatch(self, tmp_path, monkeypatch):
        from tuning.training.pipeline import stages
        f = tmp_path / "meta.jsonl"
        _write_meta(f, [CHECKPOINT_ROW])
        args = _make_args(f)

        captured = {}

        def fake_dispatch(method, configs, passk_config, primary_eval,
                          monitor_evals, ppl_config, checkpoint):
            captured["method"] = method
            captured["checkpoint_path"] = checkpoint["checkpoint_path"]

        monkeypatch.setattr(stages, "_init_seeds", lambda *a, **k: None)
        monkeypatch.setattr(stages, "set_chat_template", lambda *a, **k: None)
        monkeypatch.setattr(stages, "_build_post_training_configs",
                            lambda *a, **k: MagicMock(
                                run_config=MagicMock(model_name="m"),
                                gpu_util=0.6,
                            ))
        monkeypatch.setattr(stages, "_train_dispatch", fake_dispatch)

        with patch("tuning.training.pipeline.stages.wandb") as wandb_mock:
            wandb_mock.init.return_value.id = "run-test"
            wandb_mock.init.return_value.__enter__ = MagicMock()
            wandb_mock.init.return_value.__exit__ = MagicMock()
            stages.run_post_training(args, "grpo")

        assert captured["method"] == "grpo"
        assert captured["checkpoint_path"] == CHECKPOINT_ROW["checkpoint_path"]
        rows = [json.loads(line) for line in open(f)]
        assert rows[0]["completed"] is True


def test_build_grpo_config_forwards_vllm_sleep_mode():
    from tuning.training.pipeline.stages import _build_post_training_configs

    args = _make_args("/tmp/unused", grpo_vllm_sleep_mode=False)
    configs = _build_post_training_configs(
        args, "grpo", dict(CHECKPOINT_ROW), train_size=512,
    )

    assert configs.training_args.vllm_enable_sleep_mode is False


class TestBuildPostTrainingConfigsContinueFlag:
    """_build_post_training_configs reads checkpoint['continue'] and writes it to
    training_args.resume_from_checkpoint for both DPO and GRPO. Absent ⇒ False."""

    def _build(self, method, checkpoint):
        from tuning.training.pipeline.stages import _build_post_training_configs
        args = _make_args("/tmp/unused")
        configs = _build_post_training_configs(args, method, checkpoint, train_size=512)
        return configs.training_args.resume_from_checkpoint

    def test_grpo_continue_true(self):
        assert self._build("grpo", {**CHECKPOINT_ROW, "continue": True}) is True

    def test_grpo_continue_false(self):
        assert self._build("grpo", {**CHECKPOINT_ROW, "continue": False}) is False

    def test_grpo_continue_absent_defaults_false(self):
        assert self._build("grpo", dict(CHECKPOINT_ROW)) is False

    def test_dpo_continue_true(self):
        assert self._build("dpo", {**CHECKPOINT_ROW, "continue": True}) is True

    def test_dpo_continue_absent_defaults_false(self):
        assert self._build("dpo", dict(CHECKPOINT_ROW)) is False


def test_build_grpo_config_precision_auto_keeps_model_dtype_none():
    from tuning.training.pipeline.stages import _build_post_training_configs

    args = _make_args("/tmp/unused", grpo_precision="auto")
    configs = _build_post_training_configs(
        args, "grpo", dict(CHECKPOINT_ROW), train_size=512,
    )

    assert configs.model_load_config.dtype is None
    assert configs.training_args.precision == "auto"


def test_build_grpo_config_precision_fp16_maps_model_and_hf_args():
    from tuning.training.pipeline.stages import _build_post_training_configs

    args = _make_args("/tmp/unused", grpo_precision="fp16")
    configs = _build_post_training_configs(
        args, "grpo", dict(CHECKPOINT_ROW), train_size=512,
    )
    hf_args = configs.training_args.to_hf_args(output_dir="/tmp/test")

    assert configs.model_load_config.dtype is torch.float16
    assert hf_args["fp16"] is True
    assert hf_args["bf16"] is False


def test_build_grpo_config_precision_bf16_maps_model_and_hf_args():
    from tuning.training.pipeline.stages import _build_post_training_configs

    args = _make_args("/tmp/unused", grpo_precision="bf16")
    configs = _build_post_training_configs(
        args, "grpo", dict(CHECKPOINT_ROW), train_size=512,
    )
    hf_args = configs.training_args.to_hf_args(output_dir="/tmp/test")

    assert configs.model_load_config.dtype is torch.bfloat16
    assert hf_args["fp16"] is False
    assert hf_args["bf16"] is True
