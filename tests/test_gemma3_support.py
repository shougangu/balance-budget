# ABOUTME: Verifies gemma3-* model entries are registered across CLI GPU maps,
# ABOUTME: HF model paths, chat-template resolution, stop tokens, and response delimiters.

import warnings

import pytest
from jinja2 import BaseLoader, Environment

from tuning.config import (
    HF_MODEL_MAP,
    MODEL_CHAT_TEMPLATE_MAP,
    resolve_chat_template,
)
from tuning.training.pipeline.cli import (
    MODEL_TO_GPU_1,
    MODEL_TO_GPU_2,
    MODEL_TO_GPU_3,
)
from tuning.utils.utils import GEMMA_3_CHAT_TEMPLATE, RESPONSE_DELIMITERS, STOP_TOKENS


GEMMA3_MODELS = ["gemma3-1B", "gemma3-4B", "gemma3-12B"]
BOS = "<bos>"


def render_gemma3(messages, add_generation_prompt=False):
    env = Environment(loader=BaseLoader())
    template = env.from_string(GEMMA_3_CHAT_TEMPLATE)
    return template.render(
        messages=messages,
        bos_token=BOS,
        add_generation_prompt=add_generation_prompt,
    )


@pytest.mark.parametrize("model", GEMMA3_MODELS)
def test_gemma3_in_cli_gpu_maps(model):
    assert model in MODEL_TO_GPU_1, f"{model} missing from MODEL_TO_GPU_1 (SFT)"
    assert model in MODEL_TO_GPU_2, f"{model} missing from MODEL_TO_GPU_2 (DPO)"
    assert model in MODEL_TO_GPU_3, f"{model} missing from MODEL_TO_GPU_3 (GRPO)"
    assert 0.0 < MODEL_TO_GPU_1[model] <= 1.0
    assert 0.0 < MODEL_TO_GPU_2[model] <= 1.0
    assert 0.0 < MODEL_TO_GPU_3[model] <= 1.0


@pytest.mark.parametrize("model", GEMMA3_MODELS)
def test_gemma3_in_hf_model_map(model):
    assert model in HF_MODEL_MAP, f"{model} missing from HF_MODEL_MAP"
    path = HF_MODEL_MAP[model]
    assert "gemma" in path.lower()
    assert "bnb" not in path.lower(), f"{model} maps to a quantized checkpoint: {path}"
    assert "4bit" not in path.lower(), f"{model} maps to a quantized checkpoint: {path}"
    assert "8bit" not in path.lower(), f"{model} maps to a quantized checkpoint: {path}"


@pytest.mark.parametrize("model", GEMMA3_MODELS)
def test_gemma3_chat_template_resolution(model):
    assert MODEL_CHAT_TEMPLATE_MAP[model] == "gemma-3"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_chat_template(model) == "gemma-3"
        assert len(caught) == 0, f"Unexpected warning for {model}: {[str(w.message) for w in caught]}"


def test_gemma3_chat_template_resolution_with_run_name_suffix():
    assert resolve_chat_template("gemma3-4B_sft-tuluif-500_pt-tuluif-500") == "gemma-3"


def test_gemma3_chat_template_renders_system_as_first_user_prefix():
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "  What is 2+2?  "},
        {"role": "assistant", "content": " 4 "},
    ]

    rendered = render_gemma3(messages)

    assert rendered == (
        "<bos><start_of_turn>user\n"
        "You are concise.\n\n"
        "What is 2+2?<end_of_turn>\n"
        "<start_of_turn>model\n"
        "4<end_of_turn>\n"
    )


def test_gemma3_chat_template_adds_generation_prompt():
    messages = [
        {"role": "system", "content": "You are concise."},
        {"role": "user", "content": "  Say hi.  "},
    ]

    rendered = render_gemma3(messages, add_generation_prompt=True)

    assert rendered == (
        "<bos><start_of_turn>user\n"
        "You are concise.\n\n"
        "Say hi.<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def test_gemma3_chat_template_supports_sharegpt_messages():
    messages = [
        {"from": "system", "value": "You are concise."},
        {"from": "human", "value": "  Hi  "},
        {"from": "gpt", "value": " Hello "},
    ]

    rendered = render_gemma3(messages)

    assert rendered == (
        "<bos><start_of_turn>user\n"
        "You are concise.\n\n"
        "Hi<end_of_turn>\n"
        "<start_of_turn>model\n"
        "Hello<end_of_turn>\n"
    )


def test_gemma3_stop_tokens_registered():
    assert "gemma-3" in STOP_TOKENS
    tokens = STOP_TOKENS["gemma-3"]
    assert "<end_of_turn>" in tokens


def test_gemma3_response_delimiters_registered():
    assert "gemma-3" in RESPONSE_DELIMITERS
    delims = RESPONSE_DELIMITERS["gemma-3"]
    assert delims["instruction_part"] == "<start_of_turn>user\n"
    assert delims["response_part"] == "<start_of_turn>model\n"


def test_gemma3_get_stop_tokens_runtime_lookup():
    import tuning.config
    from tuning.utils.utils import get_stop_tokens

    original = tuning.config.DEFAULT_CHAT_TEMPLATE
    try:
        tuning.config.DEFAULT_CHAT_TEMPLATE = "gemma-3"
        assert "<end_of_turn>" in get_stop_tokens()
    finally:
        tuning.config.DEFAULT_CHAT_TEMPLATE = original


def test_gemma3_get_response_delimiters_runtime_lookup():
    import tuning.config
    from tuning.utils.utils import get_response_delimiters

    original = tuning.config.DEFAULT_CHAT_TEMPLATE
    try:
        tuning.config.DEFAULT_CHAT_TEMPLATE = "gemma-3"
        delims = get_response_delimiters()
        assert delims["response_part"] == "<start_of_turn>model\n"
    finally:
        tuning.config.DEFAULT_CHAT_TEMPLATE = original
