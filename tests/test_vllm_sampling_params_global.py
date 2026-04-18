import pytest

import tuning.config
from tuning.utils.utils import STOP_TOKENS


@pytest.fixture(autouse=True)
def restore_global():
    original = tuning.config.DEFAULT_CHAT_TEMPLATE
    yield
    tuning.config.DEFAULT_CHAT_TEMPLATE = original


@pytest.fixture(autouse=True)
def restore_seed_globals():
    orig_seed = tuning.config.DEFAULT_SEED
    orig_eval = tuning.config.DEFAULT_EVAL_SEED
    yield
    tuning.config.DEFAULT_SEED = orig_seed
    tuning.config.DEFAULT_EVAL_SEED = orig_eval


def test_vllm_sampling_params_no_chat_template_field():
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert not hasattr(config, "chat_template")


def test_vllm_sampling_params_stop_tokens_from_global():
    tuning.config.DEFAULT_CHAT_TEMPLATE = "llama-3.1"
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert config.stop == STOP_TOKENS["llama-3.1"]


def test_vllm_sampling_params_seed_resolves_from_global():
    """When seed is not set, it resolves from the global eval seed."""
    tuning.config.set_seed(7)
    tuning.config.DEFAULT_EVAL_SEED = None  # falls back to DEFAULT_SEED
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert config.seed == 7


def test_vllm_sampling_params_seed_resolves_eval_seed_override():
    """When DEFAULT_EVAL_SEED is set, it takes priority."""
    tuning.config.set_seed(42)
    tuning.config.set_eval_seed(99)
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert config.seed == 99


def test_vllm_sampling_params_seed_explicit_overrides_global():
    """When seed is passed explicitly, global is ignored."""
    tuning.config.set_seed(42)
    tuning.config.set_eval_seed(99)
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig(seed=7)
    assert config.seed == 7


def test_vllm_sampling_params_seed_roundtrips_through_model_dump():
    tuning.config.set_seed(42)
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig(seed=7)
    assert config.model_dump()["seed"] == 7
