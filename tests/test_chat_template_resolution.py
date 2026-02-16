import warnings

from tuning.config import (
    DEFAULT_CHAT_TEMPLATE,
    MODEL_CHAT_TEMPLATE_MAP,
    resolve_chat_template,
)


def test_resolve_chat_template_uses_override():
    assert resolve_chat_template("llama3-8B", override="mistral") == "mistral"


def test_resolve_chat_template_uses_model_map():
    assert resolve_chat_template("llama3-8B") == MODEL_CHAT_TEMPLATE_MAP["llama3-8B"]
    assert resolve_chat_template("qwen2-7B") == MODEL_CHAT_TEMPLATE_MAP["qwen2-7B"]


def test_resolve_chat_template_supports_run_name_prefix():
    assert resolve_chat_template("llama3-8B_sft-tuluif-500_pt-tuluif-500") == MODEL_CHAT_TEMPLATE_MAP["llama3-8B"]


def test_resolve_chat_template_falls_back_to_default():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_chat_template("unknown-model") == DEFAULT_CHAT_TEMPLATE
        assert len(caught) == 1
        assert "No chat template mapping found for model 'unknown-model'" in str(caught[0].message)


def test_resolve_chat_template_known_model_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_chat_template("llama3-8B") == MODEL_CHAT_TEMPLATE_MAP["llama3-8B"]
        assert len(caught) == 0
