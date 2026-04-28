# ABOUTME: Tests for VLLMRunner strategy — selection, fallback, and per-runner behavior.
# ABOUTME: vLLM is mocked; we test the dispatch shape, not real generation.

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("vllm", MagicMock())
sys.modules.setdefault("vllm.lora.request", MagicMock())
sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from tuning.training.passk.runners import RunnerConfig, VLLMRunner


def test_runner_config_defaults_are_explicit():
    cfg = RunnerConfig(
        base_model_hf="m",
        vllm_gpu_memory_utilization=0.6,
        lora_max_rank=32,
        chat_template="t",
        temperature=0.5,
        max_tokens=256,
        available_gpus=["0"],
        num_inference_gpus=1,
    )
    assert cfg.base_model_hf == "m"
    assert cfg.vllm_gpu_memory_utilization == 0.6


def test_base_runner_is_abstract():
    cfg = RunnerConfig(
        base_model_hf="m", vllm_gpu_memory_utilization=0.6, lora_max_rank=32,
        chat_template="t", temperature=0.5, max_tokens=256,
        available_gpus=["0"], num_inference_gpus=1,
    )
    with __import__("pytest").raises(NotImplementedError):
        VLLMRunner(cfg).run(model=None, eval_strategy=None, adapter_path=None)


def _make_eval_strategy(n=1):
    es = MagicMock()
    es.n_samples = n
    es.get_test_messages.return_value = [[{"role": "user", "content": "hi"}]]
    es.get_test_prompts.return_value = ["hi"]
    return es


def _make_config():
    return RunnerConfig(
        base_model_hf="m", vllm_gpu_memory_utilization=0.6, lora_max_rank=32,
        chat_template="t", temperature=0.5, max_tokens=256,
        available_gpus=["0"], num_inference_gpus=1,
    )


def test_external_runner_uses_provided_llm_and_skips_lora():
    from tuning.training.passk.runners import ExternalVLLMRunner

    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    llm = MagicMock()
    llm.chat.return_value = [fake_output]

    runner = ExternalVLLMRunner(_make_config(), llm=llm)
    out = runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
                     adapter_path=None)

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    args, kwargs = llm.chat.call_args
    assert kwargs["lora_request"] is None


def test_ephemeral_runner_creates_and_destroys_llm(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    fake_llm.chat.return_value = [fake_output]

    monkeypatch.setattr(runners_mod, "_make_llm", lambda cfg: fake_llm)
    cleanup_calls = []
    monkeypatch.setattr(runners_mod, "_cleanup_llm",
                        lambda llm: cleanup_calls.append(llm))

    model = MagicMock()
    model.parameters.return_value = iter([MagicMock(device="cuda:0")])

    runner = runners_mod.EphemeralVLLMRunner(_make_config())
    out = runner.run(model=model, eval_strategy=_make_eval_strategy(),
                     adapter_path="/tmp/adapter")

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    assert cleanup_calls == [fake_llm]
    model.cpu.assert_called_once()
    model.to.assert_called_once()
    model.train.assert_called_once()


def test_persistent_runner_keeps_llm_alive(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    fake_llm.chat.return_value = [fake_output]

    make_calls = []
    monkeypatch.setattr(runners_mod, "_make_llm",
                        lambda cfg: (make_calls.append(cfg), fake_llm)[1])

    runner = runners_mod.PersistentVLLMRunner(_make_config())
    runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
               adapter_path="/tmp/a")
    runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
               adapter_path="/tmp/a")

    assert len(make_calls) == 1


def test_data_parallel_runner_offloads_and_dispatches(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    captured = {}

    def fake_dp(eval_strategy, adapter_path, config):
        captured["eval"] = eval_strategy
        captured["adapter"] = adapter_path
        captured["num_gpus"] = config.num_inference_gpus
        return [{"prompt": "hi", "responses": ["ok"]}]

    monkeypatch.setattr(runners_mod, "_run_data_parallel", fake_dp)

    cfg = RunnerConfig(
        base_model_hf="m", vllm_gpu_memory_utilization=0.6, lora_max_rank=32,
        chat_template="t", temperature=0.5, max_tokens=256,
        available_gpus=["0", "1"], num_inference_gpus=2,
    )
    model = MagicMock()
    model.parameters.return_value = iter([MagicMock(device="cuda:0")])

    runner = runners_mod.DataParallelVLLMRunner(cfg)
    out = runner.run(model=model, eval_strategy=_make_eval_strategy(),
                     adapter_path="/tmp/a")

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    assert captured["adapter"] == "/tmp/a"
    assert captured["num_gpus"] == 2
    model.cpu.assert_called_once()
    model.to.assert_called_once()
