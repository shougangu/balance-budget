# ABOUTME: Tests for the simple chat template Jinja2 rendering and stop tokens.
# ABOUTME: Verifies training/inference modes, system prompt stripping, and ShareGPT format.

import pytest
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
