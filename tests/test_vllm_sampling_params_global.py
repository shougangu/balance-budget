import pytest

import tuning.config
from tuning.utils.utils import STOP_TOKENS


@pytest.fixture(autouse=True)
def restore_global():
    original = tuning.config.DEFAULT_CHAT_TEMPLATE
    yield
    tuning.config.DEFAULT_CHAT_TEMPLATE = original


def test_vllm_sampling_params_no_chat_template_field():
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert not hasattr(config, "chat_template")


def test_vllm_sampling_params_stop_tokens_from_global():
    tuning.config.DEFAULT_CHAT_TEMPLATE = "llama-3.1"
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert config.stop == STOP_TOKENS["llama-3.1"]
