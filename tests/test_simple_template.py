# ABOUTME: Tests for the simple chat template Jinja2 rendering and stop tokens.
# ABOUTME: Verifies training/inference modes, system prompt stripping, and ShareGPT format.

import pytest
from unittest.mock import patch, MagicMock
from jinja2 import BaseLoader, Environment

import tuning.config
from tuning.utils.utils import SIMPLE_TEMPLATE, STOP_TOKENS, get_stop_tokens


BOS = "<bos>"
EOS = "<eos>"


def render(messages, add_generation_prompt=False):
    env = Environment(loader=BaseLoader())
    template = env.from_string(SIMPLE_TEMPLATE)
    return template.render(
        messages=messages,
        bos_token=BOS,
        eos_token=EOS,
        add_generation_prompt=add_generation_prompt,
    )


class TestSimpleTemplateRendering:
    def test_training_mode_gsm8k(self):
        messages = [
            {"role": "user", "content": "Question: What is 2+2?\nAnswer:"},
            {"role": "assistant", "content": "The answer is 4.\n\n#### 4"},
        ]
        result = render(messages, add_generation_prompt=False)
        assert result == f"{BOS}Question: What is 2+2?\nAnswer:The answer is 4.\n\n#### 4{EOS}"

    def test_inference_mode_gsm8k(self):
        messages = [
            {"role": "user", "content": "Question: What is 2+2?\nAnswer:"},
        ]
        result = render(messages, add_generation_prompt=True)
        assert result == f"{BOS}Question: What is 2+2?\nAnswer:"

    def test_system_message_stripped(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        result = render(messages, add_generation_prompt=False)
        assert "helpful assistant" not in result
        assert result == f"{BOS}Hello{EOS}"

    def test_sharegpt_format(self):
        messages = [
            {"from": "human", "value": "Question: 1+1?\nAnswer:"},
            {"from": "gpt", "value": "2"},
        ]
        result = render(messages, add_generation_prompt=False)
        assert result == f"{BOS}Question: 1+1?\nAnswer:2{EOS}"

    def test_sharegpt_system_stripped(self):
        messages = [
            {"from": "system", "value": "System prompt"},
            {"from": "human", "value": "Hello"},
        ]
        result = render(messages, add_generation_prompt=False)
        assert "System prompt" not in result
        assert result == f"{BOS}Hello{EOS}"

    def test_ifeval_no_system_prompt(self):
        messages = [
            {"role": "user", "content": "Write a poem about cats."},
        ]
        result = render(messages, add_generation_prompt=True)
        assert result == f"{BOS}Write a poem about cats."


class TestSimpleStopTokens:
    def test_simple_stop_tokens_exist(self):
        assert "simple" in STOP_TOKENS

    def test_simple_stop_tokens_cover_both_families(self):
        tokens = STOP_TOKENS["simple"]
        assert "<|end_of_text|>" in tokens
        assert "</s>" in tokens

    def test_get_stop_tokens_returns_simple(self):
        original = tuning.config.DEFAULT_CHAT_TEMPLATE
        try:
            tuning.config.DEFAULT_CHAT_TEMPLATE = "simple"
            result = get_stop_tokens()
            assert "<|end_of_text|>" in result
            assert "</s>" in result
        finally:
            tuning.config.DEFAULT_CHAT_TEMPLATE = original


class TestChatTemplateFuncSimpleMode:
    @pytest.fixture(autouse=True)
    def restore_globals(self):
        original = tuning.config.DEFAULT_CHAT_TEMPLATE
        original_base = tuning.config._BASE_CHAT_TEMPLATE
        yield
        tuning.config.DEFAULT_CHAT_TEMPLATE = original
        tuning.config._BASE_CHAT_TEMPLATE = original_base

    def test_simple_mode_overrides_tokenizer_chat_template(self):
        """When DEFAULT_CHAT_TEMPLATE is 'simple', chat_template_func should
        set tokenizer.chat_template to SIMPLE_TEMPLATE."""
        import sys

        tuning.config.DEFAULT_CHAT_TEMPLATE = "simple"
        tuning.config._BASE_CHAT_TEMPLATE = "chatml"

        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "original"

        def fake_get_chat_template(tokenizer, chat_template, mapping, map_eos_token):
            assert chat_template == "chatml", "Should use _BASE_CHAT_TEMPLATE for unsloth setup"
            return tokenizer

        # Mock the unsloth module hierarchy so the local import inside
        # chat_template_func succeeds without a GPU.
        mock_chat_templates = MagicMock()
        mock_chat_templates.get_chat_template = fake_get_chat_template
        mock_unsloth = MagicMock()
        mock_unsloth.chat_templates = mock_chat_templates

        with patch.dict(sys.modules, {
            "unsloth": mock_unsloth,
            "unsloth.chat_templates": mock_chat_templates,
        }):
            from tuning.utils.utils import chat_template_func
            result = chat_template_func(mock_tokenizer)

        assert result.chat_template == SIMPLE_TEMPLATE


class TestSFTSkipsResponseMasking:
    def test_simple_template_has_no_response_delimiters(self):
        """get_response_delimiters() should raise for 'simple', confirming
        that sft_training must skip train_on_responses_only."""
        from tuning.utils.utils import get_response_delimiters
        original = tuning.config.DEFAULT_CHAT_TEMPLATE
        try:
            tuning.config.DEFAULT_CHAT_TEMPLATE = "simple"
            with pytest.raises(ValueError, match="No response delimiters"):
                get_response_delimiters()
        finally:
            tuning.config.DEFAULT_CHAT_TEMPLATE = original
