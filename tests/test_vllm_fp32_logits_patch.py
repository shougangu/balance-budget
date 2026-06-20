# ABOUTME: Tests vLLM class-level fp32 logits and dtype-constructor patches.
# ABOUTME: Uses fake vLLM/TRL modules so no GPU vLLM import is required.

import sys
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def _install_fake_vllm(monkeypatch):
    module_name = "vllm.model_executor.layers.logits_processor"

    class FakeLogitsProcessor:
        def __init__(self, org_vocab_size):
            self.org_vocab_size = org_vocab_size
            self.gathered_dtype = None

        def _gather_logits(self, logits):
            self.gathered_dtype = logits.dtype
            return logits

        def _get_logits(self, hidden_states, lm_head, embedding_bias):
            raise AssertionError("original _get_logits should be patched")

    packages = {
        "vllm": ModuleType("vllm"),
        "vllm.model_executor": ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": ModuleType("vllm.model_executor.layers"),
        module_name: ModuleType(module_name),
    }
    packages[module_name].LogitsProcessor = FakeLogitsProcessor
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)
    return FakeLogitsProcessor


def test_vllm_fp32_logits_patch_outputs_fp32_and_trims_vocab(monkeypatch):
    from tuning.training.model_utils import install_vllm_fp32_logits_patch

    FakeLogitsProcessor = _install_fake_vllm(monkeypatch)
    assert install_vllm_fp32_logits_patch() is True

    hidden_states = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    lm_head = nn.Embedding(7, 4).to(dtype=torch.bfloat16)
    bias = torch.randn(7, dtype=torch.bfloat16)
    processor = FakeLogitsProcessor(org_vocab_size=5)

    logits = processor._get_logits(hidden_states, lm_head, bias)

    expected = F.linear(hidden_states.float(), lm_head.weight.float(), bias.float())
    assert logits.dtype == torch.float32
    assert processor.gathered_dtype == torch.float32
    assert logits.shape == (2, 3, 5)
    assert torch.allclose(logits, expected[..., :5])


def test_vllm_fp32_logits_patch_handles_fp16_without_bias(monkeypatch):
    from tuning.training.model_utils import install_vllm_fp32_logits_patch

    FakeLogitsProcessor = _install_fake_vllm(monkeypatch)
    install_vllm_fp32_logits_patch()

    hidden_states = torch.randn(2, 4, dtype=torch.float16)
    lm_head = nn.Linear(4, 6, bias=False).to(dtype=torch.float16)
    processor = FakeLogitsProcessor(org_vocab_size=6)

    logits = processor._get_logits(hidden_states, lm_head, None)

    assert logits.dtype == torch.float32
    assert logits.shape == (2, 6)
    assert torch.allclose(logits, F.linear(hidden_states.float(), lm_head.weight.float()))


def test_vllm_fp32_logits_patch_is_idempotent(monkeypatch):
    from tuning.training.model_utils import install_vllm_fp32_logits_patch

    FakeLogitsProcessor = _install_fake_vllm(monkeypatch)
    assert install_vllm_fp32_logits_patch() is True
    patched = FakeLogitsProcessor._get_logits

    assert install_vllm_fp32_logits_patch() is False
    assert FakeLogitsProcessor._get_logits is patched


def _install_fake_trl_vllm_generation(monkeypatch):
    calls = []

    def fake_llm(*args, **kwargs):
        calls.append((args, kwargs))
        return {"args": args, "kwargs": kwargs}

    trl = ModuleType("trl")
    extras = ModuleType("trl.extras")
    vllm_generation = ModuleType("trl.extras.vllm_generation")
    vllm_generation.LLM = fake_llm
    extras.vllm_generation = vllm_generation
    trl.extras = extras

    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.extras", extras)
    monkeypatch.setitem(sys.modules, "trl.extras.vllm_generation", vllm_generation)
    return vllm_generation, calls


def test_trl_vllm_dtype_patch_forces_constructor_dtype(monkeypatch):
    from tuning.training.model_utils import install_trl_vllm_dtype_patch

    vllm_generation, calls = _install_fake_trl_vllm_generation(monkeypatch)

    assert install_trl_vllm_dtype_patch("float16") is True
    result = vllm_generation.LLM("model", dtype="auto", other=True)

    assert result["kwargs"]["dtype"] == "float16"
    assert calls == [(("model",), {"dtype": "float16", "other": True})]


def test_trl_vllm_dtype_patch_is_idempotent(monkeypatch):
    from tuning.training.model_utils import install_trl_vllm_dtype_patch

    vllm_generation, calls = _install_fake_trl_vllm_generation(monkeypatch)

    assert install_trl_vllm_dtype_patch("bfloat16") is True
    patched = vllm_generation.LLM
    assert install_trl_vllm_dtype_patch("bfloat16") is False
    assert vllm_generation.LLM is patched

    vllm_generation.LLM("model")
    assert calls == [(("model",), {"dtype": "bfloat16"})]


def test_sync_colocated_vllm_chat_template_sets_missing_template():
    from tuning.training.model_utils import sync_colocated_vllm_chat_template

    vllm_generation = SimpleNamespace(chat_template=None)
    trainer = SimpleNamespace(vllm_mode="colocate", vllm_generation=vllm_generation)
    tokenizer = SimpleNamespace(chat_template="checkpoint-template")

    assert sync_colocated_vllm_chat_template(trainer, tokenizer) is True
    assert vllm_generation.chat_template == "checkpoint-template"


def test_sync_colocated_vllm_chat_template_skips_server_mode():
    from tuning.training.model_utils import sync_colocated_vllm_chat_template

    vllm_generation = SimpleNamespace(chat_template=None)
    trainer = SimpleNamespace(vllm_mode="server", vllm_generation=vllm_generation)
    tokenizer = SimpleNamespace(chat_template="checkpoint-template")

    assert sync_colocated_vllm_chat_template(trainer, tokenizer) is False
    assert vllm_generation.chat_template is None


def test_sync_colocated_vllm_chat_template_preserves_trl_override():
    from tuning.training.model_utils import sync_colocated_vllm_chat_template

    vllm_generation = SimpleNamespace(chat_template="trl-template")
    trainer = SimpleNamespace(vllm_mode="colocate", vllm_generation=vllm_generation)
    tokenizer = SimpleNamespace(chat_template="checkpoint-template")

    assert sync_colocated_vllm_chat_template(trainer, tokenizer) is False
    assert vllm_generation.chat_template == "trl-template"
