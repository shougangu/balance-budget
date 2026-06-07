# ABOUTME: Tests client-side chat rendering for GRPO's external vLLM server.
# ABOUTME: Verifies template use, sampling passthrough, and text-only safeguards.

from unittest.mock import MagicMock

import pytest

from tuning.training.server_rollouts import install_client_rendered_chat


class FakeTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return f"rendered:{messages[-1]['content']}"


def test_client_rendered_chat_uses_generate_and_preserves_sampling_params():
    client = MagicMock()
    client.generate.return_value = {"completion_ids": [[1, 2, 3]]}
    tokenizer = FakeTokenizer()
    messages = [[{"role": "user", "content": "2+2?"}]]

    install_client_rendered_chat(client, tokenizer)
    result = client.chat(
        messages=messages,
        n=8,
        repetition_penalty=1.1,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.05,
        max_tokens=128,
        logprobs=1,
        truncate_prompt_tokens=512,
        structured_outputs_regex=r"\d+",
        generation_kwargs={"seed": 7},
        chat_template_kwargs={"custom_flag": True},
    )

    assert result == {"completion_ids": [[1, 2, 3]]}
    client.generate.assert_called_once_with(
        prompts=["rendered:2+2?"],
        n=8,
        repetition_penalty=1.1,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.05,
        max_tokens=128,
        logprobs=1,
        truncate_prompt_tokens=512,
        structured_outputs_regex=r"\d+",
        generation_kwargs={"seed": 7},
    )
    _, template_kwargs = tokenizer.calls[0]
    assert template_kwargs["tokenize"] is False
    assert template_kwargs["add_generation_prompt"] is True
    assert template_kwargs["continue_final_message"] is False
    assert template_kwargs["custom_flag"] is True


def test_client_rendered_chat_forwards_explicit_template_and_tools():
    client = MagicMock()
    tokenizer = FakeTokenizer()
    tools = [{"type": "function", "function": {"name": "solve"}}]

    install_client_rendered_chat(client, tokenizer)
    client.chat(
        messages=[[{"role": "user", "content": "solve"}]],
        tools=tools,
        chat_template="checkpoint-template",
    )

    _, template_kwargs = tokenizer.calls[0]
    assert template_kwargs["tools"] == tools
    assert template_kwargs["chat_template"] == "checkpoint-template"


def test_client_rendered_chat_rejects_images_before_generation():
    client = MagicMock()
    tokenizer = FakeTokenizer()
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image_pil", "image_pil": object()},
            {"type": "text", "text": "describe"},
        ],
    }]]

    install_client_rendered_chat(client, tokenizer)

    with pytest.raises(NotImplementedError, match="text-only"):
        client.chat(messages=messages)
    client.generate.assert_not_called()


def test_install_client_rendered_chat_is_idempotent():
    client = MagicMock()
    tokenizer = FakeTokenizer()

    install_client_rendered_chat(client, tokenizer)
    installed_chat = client.chat
    install_client_rendered_chat(client, tokenizer)

    assert client.chat is installed_chat
