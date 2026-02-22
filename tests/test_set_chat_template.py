import pytest

import tuning.config
from tuning.config import set_chat_template


@pytest.fixture(autouse=True)
def restore_global():
    original = tuning.config.DEFAULT_CHAT_TEMPLATE
    yield
    tuning.config.DEFAULT_CHAT_TEMPLATE = original


def test_set_chat_template_llama_sets_global():
    set_chat_template("llama3-8B")
    assert tuning.config.DEFAULT_CHAT_TEMPLATE == "llama-3.1"


def test_set_chat_template_qwen_sets_global():
    set_chat_template("qwen2-7B")
    assert tuning.config.DEFAULT_CHAT_TEMPLATE == "chatml"


def test_set_chat_template_returns_value():
    result = set_chat_template("llama3-8B")
    assert result == "llama-3.1"


def test_set_chat_template_updates_existing():
    set_chat_template("llama3-8B")
    assert tuning.config.DEFAULT_CHAT_TEMPLATE == "llama-3.1"
    set_chat_template("qwen2-7B")
    assert tuning.config.DEFAULT_CHAT_TEMPLATE == "chatml"
