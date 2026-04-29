# ABOUTME: Tests for DDP eval support in PassAtKStoppingCallback.
# ABOUTME: CPU-only; mocks vllm, unsloth, and torch.distributed.

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from tuning.training.config_training import PassAtKConfig
from tuning.training.passk.callback import PassAtKStoppingCallback


class _FakeEval:
    """Minimal EvalStrategy stand-in."""
    def __init__(self):
        self._n_samples = 1
        self.stopping_k = 1

    @property
    def id(self): return "test"
    @property
    def n_samples(self): return self._n_samples
    @property
    def label_prefix(self): return "p@1"
    def get_test_messages(self):
        return [[{"role": "user", "content": f"Prompt {i}"}] for i in range(8)]
    def get_test_prompts(self):
        return [f"Prompt {i}" for i in range(8)]
    def score_responses(self, results, tokenizer):
        return {"pass_at_1": 0.5}
    def stopping_metric(self):
        return "pass_at_1"
    def wandb_metrics(self, scores):
        return {"eval/pass_at_1": scores["pass_at_1"]}


def _make_callback(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    config = PassAtKConfig(
        target_pass_at_k=[0.5],
        temperature=0.5,
        max_tokens=128,
        enabled=True,
        use_persistent_vllm=False,
        vllm_gpu_memory_utilization=0.4,
        num_inference_gpus=1,
    )
    tokenizer = SimpleNamespace(chat_template="dummy",
                                apply_chat_template=lambda *a, **kw: "Prompt 0")
    return PassAtKStoppingCallback(
        config=config,
        tokenizer=tokenizer,
        model_name="qwen2-2B",
        base_model_hf="Qwen/Qwen2-2B",
        primary_eval=_FakeEval(),
        monitor_evals=[],
    )


def test_is_rank_zero_no_dist(monkeypatch):
    """Without torch.distributed initialized, every process is rank 0."""
    cb = _make_callback(monkeypatch)
    with patch("torch.distributed.is_initialized", return_value=False):
        assert cb._is_rank_zero() is True


def test_is_rank_zero_under_ddp(monkeypatch):
    """Under DDP, only rank 0 returns True."""
    cb = _make_callback(monkeypatch)
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=0):
        assert cb._is_rank_zero() is True
    with patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.get_rank", return_value=1):
        assert cb._is_rank_zero() is False


def test_default_accelerator_is_none(monkeypatch):
    cb = _make_callback(monkeypatch)
    assert cb._accelerator is None


def test_accelerator_can_be_assigned_directly(monkeypatch):
    """train_model_grpo assigns trainer.accelerator to cb._accelerator directly."""
    cb = _make_callback(monkeypatch)
    fake_accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    cb._accelerator = fake_accelerator
    assert cb._accelerator is fake_accelerator


def test_save_sweetspot_unwraps_when_accelerator_provided(tmp_path, monkeypatch):
    """With accelerator, save_pretrained is called on the unwrapped PEFT model (no unsloth merge)."""
    from tuning.training.callback_utils import save_sweetspot_checkpoint

    underlying = MagicMock(name="peft_model")
    wrapped = MagicMock(name="ddp_model")
    accelerator = SimpleNamespace(unwrap_model=MagicMock(return_value=underlying))
    tokenizer = MagicMock()

    state = SimpleNamespace(global_step=42, log_history=[])
    args = SimpleNamespace(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        world_size=4,
        output_dir=str(tmp_path),
        to_dict=lambda: {"per_device_train_batch_size": 8},
    )

    save_sweetspot_checkpoint(
        model=wrapped,
        tokenizer=tokenizer,
        model_name="qwen2-2B",
        threshold_label="p@1-0.5",
        state=state,
        args=args,
        metadata_path=str(tmp_path / "meta.jsonl"),
        extra_metadata={"threshold_type": "pass_at_1", "threshold_value": 0.5},
        accelerator=accelerator,
    )

    accelerator.unwrap_model.assert_called_once_with(wrapped)
    underlying.save_pretrained.assert_called_once()
    underlying.save_pretrained_merged.assert_not_called()
    wrapped.save_pretrained.assert_not_called()


def test_save_sweetspot_no_unwrap_when_accelerator_none(tmp_path):
    """Without accelerator (SFT/DPO callers), legacy merged save is used."""
    from tuning.training.callback_utils import save_sweetspot_checkpoint

    model = MagicMock(name="unsloth_model")
    tokenizer = MagicMock()

    state = SimpleNamespace(global_step=10, log_history=[])
    args = SimpleNamespace(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        world_size=1,
        output_dir=str(tmp_path),
        to_dict=lambda: {"per_device_train_batch_size": 8},
    )

    save_sweetspot_checkpoint(
        model=model,
        tokenizer=tokenizer,
        model_name="qwen2-2B",
        threshold_label="p@1-0.5",
        state=state,
        args=args,
        metadata_path=str(tmp_path / "meta.jsonl"),
        extra_metadata={"threshold_type": "pass_at_1", "threshold_value": 0.5},
    )

    model.save_pretrained_merged.assert_called_once()
    model.save_pretrained.assert_not_called()
