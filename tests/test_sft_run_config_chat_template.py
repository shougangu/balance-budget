from tuning.training.config_training import SFTRunConfig
from tuning.config import MODEL_CHAT_TEMPLATE_MAP


def test_sft_run_config_auto_sets_chat_template_from_model():
    cfg = SFTRunConfig(model_name="llama3-8B")
    assert cfg.chat_template == MODEL_CHAT_TEMPLATE_MAP["llama3-8B"]


def test_sft_run_config_keeps_explicit_chat_template_override():
    cfg = SFTRunConfig(model_name="llama3-8B", chat_template="mistral")
    assert cfg.chat_template == "mistral"
