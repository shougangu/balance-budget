# ABOUTME: Tests that PassAtK callback can reuse an external vLLM engine (e.g. from GRPOTrainer).
# ABOUTME: Verifies LoRA save is skipped and external engine is used directly for inference.

import sys
from unittest.mock import MagicMock

# Mock heavy imports before importing callback module
sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from tuning.training.config_training import PassAtKConfig
from tuning.training.passk_callback import PassAtKStoppingCallback


class _FakeEval:
    """Minimal EvalStrategy stand-in for testing."""
    def __init__(self):
        self._n_samples = 2
        self.stopping_k = 1

    @property
    def id(self):
        return "test"

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def label_prefix(self):
        return "p@1"

    def get_test_messages(self):
        return [[{"role": "user", "content": "Prompt A"}]]

    def get_test_prompts(self):
        return ["Prompt A"]

    def score_responses(self, results, tokenizer):
        return {"pass_at_1": 0.5}

    def stopping_metric(self):
        return "pass_at_1"

    def wandb_metrics(self, scores):
        return {f"eval/pass_at_1": scores["pass_at_1"]}


def _make_callback():
    tokenizer = MagicMock()
    tokenizer.chat_template = "test_template"
    tokenizer.apply_chat_template.return_value = "formatted_prompt"

    config = PassAtKConfig(
        target_pass_at_k=[0.95],
        enabled=True,
        use_persistent_vllm=False,
        num_inference_gpus=1,
    )
    return PassAtKStoppingCallback(
        config=config,
        tokenizer=tokenizer,
        model_name="test-model",
        base_model_hf="test/model",
        primary_eval=_FakeEval(),
    )


def _mock_vllm_engine():
    """Create a mock vLLM engine that returns valid chat outputs."""
    resp1 = MagicMock()
    resp1.text = "response1"
    resp2 = MagicMock()
    resp2.text = "response2"
    output = MagicMock()
    output.outputs = [resp1, resp2]

    llm = MagicMock()
    llm.chat.return_value = [output]
    return llm


def test_set_trainer_vllm_stores_reference():
    callback = _make_callback()
    mock_llm = _mock_vllm_engine()

    callback.set_trainer_vllm(mock_llm)

    assert callback._external_vllm is mock_llm


def test_external_vllm_defaults_to_none():
    callback = _make_callback()
    assert callback._external_vllm is None


def test_external_vllm_skips_lora_save_and_uses_engine():
    callback = _make_callback()
    mock_llm = _mock_vllm_engine()
    callback.set_trainer_vllm(mock_llm)

    mock_model = MagicMock()
    eval_strategy = _FakeEval()

    scores, results = callback._run_eval_with_results(mock_model, eval_strategy)

    # Should NOT have saved LoRA adapter
    mock_model.save_pretrained.assert_not_called()

    # Should have used the external vLLM
    mock_llm.chat.assert_called_once()

    # Should still return valid scores
    assert scores == {"pass_at_1": 0.5}
    assert len(results) == 1
    assert results[0]["prompt"] == "Prompt A"
