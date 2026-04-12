import pytest

import tuning.config
from tuning.config import set_chat_template


@pytest.fixture(autouse=True)
def restore_global():
    original = tuning.config.DEFAULT_CHAT_TEMPLATE
    original_base = tuning.config._BASE_CHAT_TEMPLATE
    yield
    tuning.config.DEFAULT_CHAT_TEMPLATE = original
    tuning.config._BASE_CHAT_TEMPLATE = original_base


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


def test_set_chat_template_simple_sets_global_to_simple():
    set_chat_template("llama3-8B", simple=True)
    assert tuning.config.DEFAULT_CHAT_TEMPLATE == "simple"


def test_set_chat_template_simple_stores_base_template():
    set_chat_template("llama3-8B", simple=True)
    assert tuning.config._BASE_CHAT_TEMPLATE == "llama-3.1"


def test_set_chat_template_simple_false_no_base_template():
    set_chat_template("llama3-8B", simple=False)
    assert tuning.config.DEFAULT_CHAT_TEMPLATE == "llama-3.1"
    assert tuning.config._BASE_CHAT_TEMPLATE is None


def test_set_chat_template_simple_qwen_stores_chatml():
    set_chat_template("qwen2-7B", simple=True)
    assert tuning.config.DEFAULT_CHAT_TEMPLATE == "simple"
    assert tuning.config._BASE_CHAT_TEMPLATE == "chatml"
