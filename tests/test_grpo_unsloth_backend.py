# ABOUTME: Tests the opt-in Unsloth GRPO backend: config defaults/validation, CLI
# ABOUTME: flags, stages wiring, version normalization, and bootstrap import order.

import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

# Stub unsloth before importing pipeline modules (the real import needs a GPU).
if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub

from tuning.training.config_training import GRPOTrainingConfig
from tuning.training.pipeline.cli import (
    _parse_args, bootstrap_worker,
)
from tuning.training.pipeline.stages import _build_post_training_configs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHECKPOINT = {
    "checkpoint_path": "/models/cp",
    "data_points_seen": 1024,
    "threshold_value": 0.3,
    "threshold_type": "pass_at_1",
    "global_step": 64,
}


def _cli(*extra):
    return _parse_args(["--model", "qwen2-3B", "--wandb-project", "test", *extra])


# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------

class TestBackendCli:
    def test_backend_defaults_to_hf(self):
        assert _cli().grpo_backend == "hf"

    def test_backend_unsloth_parses(self):
        assert _cli("--grpo-backend", "unsloth").grpo_backend == "unsloth"

    def test_standby_defaults_on(self):
        assert _cli().grpo_unsloth_standby is True

    def test_standby_can_disable(self):
        assert _cli("--no-grpo-unsloth-standby").grpo_unsloth_standby is False

    def test_num_chunks_default_auto(self):
        assert _cli().grpo_unsloth_num_chunks == -1

    def test_num_chunks_positive(self):
        assert _cli("--grpo-unsloth-num-chunks", "16").grpo_unsloth_num_chunks == 16


    def test_mini_batch_default_none(self):
        assert _cli().grpo_unsloth_mini_batch is None

    def test_mini_batch_positive(self):
        assert _cli("--grpo-unsloth-mini-batch", "4").grpo_unsloth_mini_batch == 4


class TestUnslothCliRejections:
    def test_server_mode_rejected(self):
        with pytest.raises(SystemExit):
            _cli("--grpo-backend", "unsloth", "--grpo-vllm-mode", "server")

    def test_liger_kernel_rejected(self):
        with pytest.raises(SystemExit):
            _cli("--grpo-backend", "unsloth", "--grpo-use-liger-kernel")

    def test_colocate_with_gradient_checkpointing_ok(self):
        assert _cli("--grpo-backend", "unsloth").grpo_backend == "unsloth"

    def test_hf_backend_allows_server_mode(self):
        # The Unsloth restrictions must not constrain the default HF backend.
        args = _cli("--grpo-backend", "hf", "--grpo-vllm-mode", "server",
                    "--grpo-num-gpus", "2")
        assert args.grpo_vllm_mode == "server"


# ---------------------------------------------------------------------------
# Config defaults / validation / serialization
# ---------------------------------------------------------------------------

class TestConfigUnslothDefaults:
    def test_defaults(self):
        c = GRPOTrainingConfig()
        assert c.grpo_backend == "hf"
        assert c.unsloth_standby is True
        assert c.unsloth_num_chunks == -1
        assert c.unsloth_grpo_mini_batch is None

    def test_hf_pops_backend_only_fields(self):
        d = GRPOTrainingConfig().to_hf_args(output_dir="/tmp/x")
        assert "grpo_backend" not in d
        assert "unsloth_standby" not in d
        assert "unsloth_num_chunks" not in d
        assert "unsloth_grpo_mini_batch" not in d
        assert "vllm_max_model_length" not in d

    def test_unsloth_keeps_chunk_fields(self):
        c = GRPOTrainingConfig(grpo_backend="unsloth", unsloth_num_chunks=16,
                               unsloth_grpo_mini_batch=4)
        d = c.to_hf_args(output_dir="/tmp/x")
        assert d["unsloth_num_chunks"] == 16
        assert d["unsloth_grpo_mini_batch"] == 4
        # backend/standby are consumed by grpo_training, not GRPOConfig.
        assert "grpo_backend" not in d
        assert "unsloth_standby" not in d


# ---------------------------------------------------------------------------
# stages: config build + GPU utilisation defaults
# ---------------------------------------------------------------------------

class TestStagesUnslothWiring:
    def test_wires_unsloth_fields(self):
        args = _cli("--grpo-backend", "unsloth",
                    "--grpo-unsloth-num-chunks", "16",
                    "--grpo-unsloth-mini-batch", "4")
        configs = _build_post_training_configs(args, "grpo", dict(CHECKPOINT),
                                               train_size=512)
        ta = configs.training_args
        assert ta.grpo_backend == "unsloth"
        assert ta.unsloth_standby is True
        assert ta.unsloth_num_chunks == 16
        assert ta.unsloth_grpo_mini_batch == 4

    def test_standby_defaults_gpu_util_to_090(self):
        args = _cli("--grpo-backend", "unsloth")
        configs = _build_post_training_configs(args, "grpo", dict(CHECKPOINT),
                                               train_size=512)
        assert configs.gpu_util == 0.90
        assert configs.training_args.vllm_gpu_memory_utilization == 0.90

    def test_no_standby_keeps_model_gpu_map(self):
        args = _cli("--grpo-backend", "unsloth", "--no-grpo-unsloth-standby")
        configs = _build_post_training_configs(args, "grpo", dict(CHECKPOINT),
                                               train_size=512)
        # MODEL_TO_GPU_3["qwen2-3B"] == 0.45
        assert configs.gpu_util == 0.45

    def test_explicit_gpu_util_overrides_unsloth_default(self):
        args = _cli("--grpo-backend", "unsloth", "--grpo-gpu-util", "0.8")
        configs = _build_post_training_configs(args, "grpo", dict(CHECKPOINT),
                                               train_size=512)
        assert configs.gpu_util == 0.8

    def test_hf_backend_uses_model_gpu_map(self):
        args = _cli()  # hf backend by default
        configs = _build_post_training_configs(args, "grpo", dict(CHECKPOINT),
                                               train_size=512)
        assert configs.gpu_util == 0.45


# ---------------------------------------------------------------------------
# bootstrap_worker: Standby env + import order (no real heavy imports)
# ---------------------------------------------------------------------------

class TestBootstrapWorker:
    def _args(self, **kw):
        base = dict(run_sft=False, run_dpo=False, run_grpo=True, run_all=False,
                    grpo_backend="hf", grpo_unsloth_standby=True)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_grpo_unsloth_sets_standby_before_import(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_VLLM_STANDBY", raising=False)
        env_at_import = {}

        def importer(name):
            env_at_import[name] = os.environ.get("UNSLOTH_VLLM_STANDBY")

        bootstrap_worker(self._args(grpo_backend="unsloth"), importer=importer)
        assert env_at_import == {"unsloth": "1"}

    def test_grpo_unsloth_no_standby_unsets_env(self, monkeypatch):
        monkeypatch.setenv("UNSLOTH_VLLM_STANDBY", "1")
        seen = []
        bootstrap_worker(
            self._args(grpo_backend="unsloth", grpo_unsloth_standby=False),
            importer=seen.append,
        )
        assert seen == ["unsloth"]
        assert "UNSLOTH_VLLM_STANDBY" not in os.environ

    def test_grpo_hf_does_not_import_unsloth(self, monkeypatch):
        monkeypatch.delenv("UNSLOTH_VLLM_STANDBY", raising=False)
        seen = []
        bootstrap_worker(self._args(grpo_backend="hf"), importer=seen.append)
        assert seen == []
        assert "UNSLOTH_VLLM_STANDBY" not in os.environ

    def test_grpo_hf_keeps_torchrun_visibility(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES_ALL", raising=False)

        bootstrap_worker(self._args(grpo_backend="hf"), importer=lambda name: None)

        assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"
        assert os.environ["CUDA_VISIBLE_DEVICES_ALL"] == "4,7"
        assert os.environ["LOCAL_RANK"] == "1"

    def test_grpo_unsloth_isolates_torchrun_rank(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "1")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES_ALL", raising=False)
        seen = []

        bootstrap_worker(
            self._args(grpo_backend="unsloth"),
            importer=seen.append,
        )

        assert seen == ["unsloth"]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"
        assert os.environ["CUDA_VISIBLE_DEVICES_ALL"] == "4,7"
        assert os.environ["LOCAL_RANK"] == "0"

    def test_sft_imports_unsloth(self):
        seen = []
        bootstrap_worker(
            SimpleNamespace(run_sft=True, run_dpo=False, run_grpo=False, run_all=False),
            importer=seen.append,
        )
        assert seen == ["unsloth"]

    def test_orchestrator_mode_is_noop(self):
        seen = []
        bootstrap_worker(
            SimpleNamespace(run_sft=False, run_dpo=False, run_grpo=False, run_all=True),
            importer=seen.append,
        )
        assert seen == []


# ---------------------------------------------------------------------------
# Subprocess tests: isolate the real trl/vllm import in grpo_training.
# ---------------------------------------------------------------------------

def _run_subprocess(script):
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )


_IMPORT_ORDER_SCRIPT = """
import os, sys, types, importlib.abc, importlib.machinery

events = []


class Recorder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name in ("unsloth", "trl") and name not in sys.modules:
            events.append((name, os.environ.get("UNSLOTH_VLLM_STANDBY")))
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        pass


sys.meta_path.insert(0, Recorder())
os.environ.pop("UNSLOTH_VLLM_STANDBY", None)

from tuning.training.pipeline.cli import bootstrap_worker

args = types.SimpleNamespace(
    run_sft=False, run_dpo=False, run_grpo=True, run_all=False,
    grpo_backend="unsloth", grpo_unsloth_standby=True,
)
bootstrap_worker(args)
import trl  # noqa: F401  -- recorded by the finder, must come after unsloth

names = [name for name, _ in events]
assert names.index("unsloth") < names.index("trl"), events
assert events[0] == ("unsloth", "1"), events  # Standby set before unsloth imports
print("IMPORT_ORDER_OK")
"""


def test_standby_set_before_unsloth_and_unsloth_before_trl():
    result = _run_subprocess(_IMPORT_ORDER_SCRIPT)
    assert "IMPORT_ORDER_OK" in result.stdout, (result.stdout, result.stderr)


_ENGINE_REUSE_SCRIPT = """
from types import SimpleNamespace
import tuning.training.grpo_training as g

engine = object()
model = SimpleNamespace(vllm_engine=engine)
ok_gen = SimpleNamespace(llm=engine, unsloth_fast_inference_lora=True)
assert g._validate_unsloth_engine_reuse(
    model, SimpleNamespace(vllm_generation=ok_gen)
) is engine

second_gen = SimpleNamespace(llm=object(), unsloth_fast_inference_lora=True)
try:
    g._validate_unsloth_engine_reuse(model, SimpleNamespace(vllm_generation=second_gen))
except RuntimeError as exc:
    assert "second" in str(exc).lower(), exc
else:
    raise AssertionError("expected RuntimeError for a second engine")

no_lora_gen = SimpleNamespace(llm=engine, unsloth_fast_inference_lora=False)
try:
    g._validate_unsloth_engine_reuse(model, SimpleNamespace(vllm_generation=no_lora_gen))
except RuntimeError as exc:
    assert "full-model" in str(exc).lower(), exc
else:
    raise AssertionError("expected RuntimeError when LoRA sync is not patched")

print("ENGINE_REUSE_OK")
"""


def test_unsloth_reuses_the_model_owned_engine():
    result = _run_subprocess(_ENGINE_REUSE_SCRIPT)
    assert "ENGINE_REUSE_OK" in result.stdout, (result.stdout, result.stderr)


def test_unsloth_loader_forwards_training_and_vllm_settings(monkeypatch):
    from tuning.training.model_utils import load_unsloth_model_with_lora

    monkeypatch.setenv("LOCAL_RANK", "0")
    calls = {}
    fake_model = object()
    fake_tokenizer = object()

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["load"] = kwargs
            return fake_model, fake_tokenizer

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["peft"] = {"model": model, **kwargs}
            return "peft-model"

    monkeypatch.setattr(
        sys.modules["unsloth"],
        "FastLanguageModel",
        FakeFastLanguageModel,
        raising=False,
    )
    model_load_config = SimpleNamespace(
        max_seq_length=1024, dtype=None, load_in_4bit=True,
    )
    lora_config = SimpleNamespace(
        r=32,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config={},
    )

    model, tokenizer = load_unsloth_model_with_lora(
        "/models/test",
        model_load_config,
        lora_config,
        layers_to_transform=[20, 21],
        fast_inference=True,
        gpu_memory_utilization=0.9,
        standby=True,
        max_seq_length=3072,
    )

    assert (model, tokenizer) == ("peft-model", fake_tokenizer)
    assert calls["load"] == {
        "model_name": "/models/test",
        "max_seq_length": 3072,
        "dtype": None,
        "load_in_4bit": True,
        "fast_inference": True,
        "device_map": {"": "cuda:0"},
        "gpu_memory_utilization": 0.9,
        "max_lora_rank": 32,
        "disable_log_stats": False,
        "unsloth_vllm_standby": True,
    }
    assert calls["peft"]["model"] is fake_model
    assert calls["peft"]["layers_to_transform"] == [20, 21]
    assert calls["peft"]["use_gradient_checkpointing"] == "unsloth"


def test_final_save_does_not_silently_fall_back_to_adapter_only(tmp_path):
    from tuning.training.model_utils import save_trained_model

    adapter_only_model = SimpleNamespace(save_pretrained=lambda path: None)
    with pytest.raises(TypeError, match="merged model"):
        save_trained_model(
            adapter_only_model,
            tokenizer=SimpleNamespace(),
            trainer=SimpleNamespace(),
            output_dir=str(tmp_path),
        )
