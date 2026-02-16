from tuning.training.config_training import PTRunConfig
from tuning.config import MODEL_CHAT_TEMPLATE_MAP


def test_pft_run_config_auto_sets_chat_template_from_model():
    cfg = PTRunConfig(model_name="qwen2-7B")
    assert cfg.chat_template == MODEL_CHAT_TEMPLATE_MAP["qwen2-7B"]


def test_pft_run_config_keeps_explicit_chat_template_override():
    cfg = PTRunConfig(model_name="qwen2-7B", chat_template="mistral")
    assert cfg.chat_template == "mistral"
