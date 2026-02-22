import sys
import types

import pytest

import tuning.config
from tuning.utils.utils import (
    STOP_TOKENS,
    RESPONSE_DELIMITERS,
    chat_template_func,
    get_response_delimiters,
    get_stop_tokens,
)


@pytest.fixture(autouse=True)
def restore_global():
    original = tuning.config.DEFAULT_CHAT_TEMPLATE
    yield
    tuning.config.DEFAULT_CHAT_TEMPLATE = original


def test_get_stop_tokens_reads_global_not_import_snapshot():
    tuning.config.DEFAULT_CHAT_TEMPLATE = "llama-3.1"
    assert get_stop_tokens() == STOP_TOKENS["llama-3.1"]


def test_get_stop_tokens_chatml():
    tuning.config.DEFAULT_CHAT_TEMPLATE = "chatml"
    assert get_stop_tokens() == STOP_TOKENS["chatml"]


def test_get_stop_tokens_llama():
    tuning.config.DEFAULT_CHAT_TEMPLATE = "llama-3.1"
    assert get_stop_tokens() == STOP_TOKENS["llama-3.1"]


def test_get_response_delimiters_reads_global_not_import_snapshot():
    tuning.config.DEFAULT_CHAT_TEMPLATE = "llama-3.1"
    assert get_response_delimiters() == RESPONSE_DELIMITERS["llama-3.1"]


def test_get_response_delimiters_chatml():
    tuning.config.DEFAULT_CHAT_TEMPLATE = "chatml"
    assert get_response_delimiters() == RESPONSE_DELIMITERS["chatml"]


def test_get_response_delimiters_llama():
    tuning.config.DEFAULT_CHAT_TEMPLATE = "llama-3.1"
    assert get_response_delimiters() == RESPONSE_DELIMITERS["llama-3.1"]


def _make_fake_unsloth(monkeypatch, calls):
    def fake_get_chat_template(tokenizer, chat_template, mapping, map_eos_token):
        calls.append(chat_template)
        tokenizer.chat_template = f"applied:{chat_template}"
        return tokenizer

    fake_unsloth = types.ModuleType("unsloth")
    fake_chat_templates = types.ModuleType("unsloth.chat_templates")
    fake_chat_templates.get_chat_template = fake_get_chat_template
    fake_unsloth.chat_templates = fake_chat_templates
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.chat_templates", fake_chat_templates)


def test_chat_template_func_reads_global(monkeypatch):
    calls = []
    _make_fake_unsloth(monkeypatch, calls)
    tuning.config.DEFAULT_CHAT_TEMPLATE = "chatml"
    tokenizer = types.SimpleNamespace()
    chat_template_func(tokenizer)
    assert calls[-1] == "chatml"


def test_chat_template_func_reads_updated_global(monkeypatch):
    calls = []
    _make_fake_unsloth(monkeypatch, calls)
    tuning.config.DEFAULT_CHAT_TEMPLATE = "llama-3.1"
    tokenizer = types.SimpleNamespace()
    chat_template_func(tokenizer)
    assert calls[-1] == "llama-3.1"
