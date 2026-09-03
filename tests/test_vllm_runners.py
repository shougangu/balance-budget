# ABOUTME: Tests for VLLMRunner strategy — selection, fallback, and per-runner behavior.
# ABOUTME: vLLM is mocked; we test the dispatch shape, not real generation.

import sys
from types import SimpleNamespace
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
        VLLMRunner(cfg).run(model=None, eval_strategy=None, checkpoint_path=None)


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
                     checkpoint_path=None)

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    args, kwargs = llm.chat.call_args
    assert kwargs["lora_request"] is None


def test_external_runner_syncs_sleeping_trainer_vllm():
    from tuning.training.passk.runners import ExternalVLLMRunner

    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    llm = MagicMock()
    llm.chat.return_value = [fake_output]
    vllm_generation = SimpleNamespace(
        enable_sleep_mode=True,
        sync_weights=MagicMock(),
    )
    trainer = SimpleNamespace(_last_loaded_step=128)

    runner = ExternalVLLMRunner(
        _make_config(),
        llm=llm,
        vllm_generation=vllm_generation,
        trainer=trainer,
    )
    out = runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
                     checkpoint_path=None)

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    vllm_generation.sync_weights.assert_called_once_with()
    llm.wake_up.assert_called_once_with(tags=["kv_cache"])
    llm.sleep.assert_called_once_with(level=2)
    assert trainer._last_loaded_step == -1


def test_server_runner_syncs_trainer_weights_before_generate(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    monkeypatch.setattr(runners_mod.dist, "is_initialized", lambda: False)

    call_order = []
    vllm_generation = SimpleNamespace(
        sync_weights=MagicMock(side_effect=lambda: call_order.append("sync")),
    )
    client = MagicMock()

    def fake_generate(**kwargs):
        call_order.append("generate")
        return {"completion_ids": [[1], [2]]}

    client.generate.side_effect = fake_generate

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = (
        lambda messages, **kwargs: f"rendered:{messages[0]['content']}"
    )
    tokenizer.batch_decode.return_value = ["ok0", "ok1"]

    runner = runners_mod.ServerVLLMRunner(
        _make_config(),
        client=client,
        tokenizer=tokenizer,
        vllm_generation=vllm_generation,
    )
    out = runner.run(
        model=MagicMock(), eval_strategy=_make_eval_strategy(n=2), checkpoint_path=None,
    )

    assert call_order == ["sync", "generate"]
    assert out == [{"prompt": "hi", "responses": ["ok0", "ok1"]}]
    assert client.generate.call_args.kwargs["prompts"] == ["rendered:hi"]
    vllm_generation.sync_weights.assert_called_once_with()


def test_server_runner_requires_sync_weights_when_generation_is_provided(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    monkeypatch.setattr(runners_mod.dist, "is_initialized", lambda: False)

    client = MagicMock()
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "rendered"
    runner = runners_mod.ServerVLLMRunner(
        _make_config(),
        client=client,
        tokenizer=tokenizer,
        vllm_generation=SimpleNamespace(),
    )

    with __import__("pytest").raises(RuntimeError, match="without sync_weights"):
        runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(), checkpoint_path=None)

    client.generate.assert_not_called()


def test_ephemeral_runner_creates_and_destroys_llm(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    fake_llm.chat.return_value = [fake_output]

    monkeypatch.setattr(runners_mod, "_make_llm", lambda cfg, **kwargs: fake_llm)
    cleanup_calls = []
    monkeypatch.setattr(runners_mod, "_cleanup_llm",
                        lambda llm: cleanup_calls.append(llm))

    model = MagicMock()
    model.parameters.return_value = iter([MagicMock(device="cuda:0")])

    optimizer = MagicMock()
    offloaded = []
    monkeypatch.setattr(
        runners_mod, "_offload_paged_optimizer_state",
        lambda value: offloaded.append(value),
    )

    runner = runners_mod.EphemeralVLLMRunner(_make_config())
    out = runner.run(model=model, eval_strategy=_make_eval_strategy(),
                     checkpoint_path="/tmp/adapter", optimizer=optimizer)

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    assert cleanup_calls == [fake_llm]
    assert offloaded == [optimizer]
    model.cpu.assert_called_once()
    model.to.assert_called_once()
    model.train.assert_called_once()


def test_paged_optimizer_state_is_prefetched_to_cpu_with_telemetry(
    monkeypatch, capsys,
):
    from tuning.training.passk import runners as runners_mod

    gib = 1024**3
    page_manager = SimpleNamespace(
        paged_tensors=[
            SimpleNamespace(nbytes=2 * gib),
            SimpleNamespace(nbytes=3 * gib),
        ],
        prefetch_all=MagicMock(),
    )
    paged_optimizer = SimpleNamespace(is_paged=True, page_mng=page_manager)
    accelerate_wrapper = SimpleNamespace(optimizer=paged_optimizer)

    prefetched = []
    monkeypatch.setattr(runners_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runners_mod.torch.cuda, "synchronize", MagicMock())
    monkeypatch.setattr(runners_mod.torch.cuda, "empty_cache", MagicMock())
    monkeypatch.setattr(
        runners_mod,
        "_prefetch_managed_tensors_to_cpu",
        lambda tensors: prefetched.extend(tensors),
    )
    monkeypatch.setattr(
        runners_mod.torch.cuda, "mem_get_info",
        MagicMock(side_effect=[(20 * gib, 80 * gib), (25 * gib, 80 * gib)]),
    )

    assert runners_mod._offload_paged_optimizer_state(accelerate_wrapper) is True

    assert prefetched == page_manager.paged_tensors
    page_manager.prefetch_all.assert_not_called()
    assert runners_mod.torch.cuda.synchronize.call_count == 2
    output = capsys.readouterr().out
    assert "5.00 GiB managed" in output
    assert "GPU free 20.00 -> 25.00 GiB" in output


def test_managed_pages_use_cuda_host_location_v2(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    calls = []

    class FakeDriver:
        def cuMemPrefetchAsync_v2(self, pointer, nbytes, location, flags, stream):
            calls.append((pointer, nbytes, location.type, location.id, flags, stream))
            return 0

    monkeypatch.setattr(runners_mod, "_load_cuda_driver", lambda: FakeDriver())
    tensors = [
        SimpleNamespace(
            nbytes=4096, is_paged=True, data_ptr=lambda: 0x1000,
        ),
        SimpleNamespace(
            nbytes=8192, is_paged=True, data_ptr=lambda: 0x2000,
        ),
    ]

    runners_mod._prefetch_managed_tensors_to_cpu(tensors)

    assert calls == [
        (0x2000, 8192, runners_mod._CU_MEM_LOCATION_TYPE_HOST, 0, 0, None),
        (0x1000, 4096, runners_mod._CU_MEM_LOCATION_TYPE_HOST, 0, 0, None),
    ]


def test_managed_pages_fall_back_to_legacy_cpu_destination(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    calls = []

    class FakeDriver:
        def cuMemPrefetchAsync(self, pointer, nbytes, device, stream):
            calls.append((pointer, nbytes, device, stream))
            return 0

    monkeypatch.setattr(runners_mod, "_load_cuda_driver", lambda: FakeDriver())
    tensor = SimpleNamespace(
        nbytes=4096, is_paged=True, data_ptr=lambda: 0x1000,
    )

    runners_mod._prefetch_managed_tensors_to_cpu([tensor])

    assert calls == [(0x1000, 4096, runners_mod._CU_DEVICE_CPU, None)]


def test_nonpaged_optimizer_state_is_not_offloaded(monkeypatch):
    from tuning.training.passk import runners as runners_mod

    prefetch_all = MagicMock()
    optimizer = SimpleNamespace(
        is_paged=False,
        page_mng=SimpleNamespace(prefetch_all=prefetch_all),
    )

    assert runners_mod._offload_paged_optimizer_state(optimizer) is False
    prefetch_all.assert_not_called()


def test_persistent_runner_keeps_llm_alive(monkeypatch, tmp_path):
    from tuning.training.passk import runners as runners_mod

    fake_llm = MagicMock()
    fake_output = MagicMock()
    fake_output.outputs = [MagicMock(text="ok")]
    fake_llm.chat.return_value = [fake_output]

    make_calls = []
    monkeypatch.setattr(runners_mod, "_make_llm",
                        lambda cfg, **kwargs: (make_calls.append(cfg), fake_llm)[1])

    (tmp_path / "adapter_config.json").write_text("{}")
    runner = runners_mod.PersistentVLLMRunner(_make_config())
    runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
               checkpoint_path=str(tmp_path))
    runner.run(model=MagicMock(), eval_strategy=_make_eval_strategy(),
               checkpoint_path=str(tmp_path))

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
    optimizer = MagicMock()
    offloaded = []
    monkeypatch.setattr(
        runners_mod, "_offload_paged_optimizer_state",
        lambda value: offloaded.append(value),
    )

    cfg = RunnerConfig(
        base_model_hf="m", vllm_gpu_memory_utilization=0.6, lora_max_rank=32,
        chat_template="t", temperature=0.5, max_tokens=256,
        available_gpus=["0", "1"], num_inference_gpus=2,
    )
    model = MagicMock()
    model.parameters.return_value = iter([MagicMock(device="cuda:0")])

    runner = runners_mod.DataParallelVLLMRunner(cfg)
    out = runner.run(model=model, eval_strategy=_make_eval_strategy(),
                     checkpoint_path="/tmp/a", optimizer=optimizer)

    assert out == [{"prompt": "hi", "responses": ["ok"]}]
    assert captured["adapter"] == "/tmp/a"
    assert captured["num_gpus"] == 2
    assert offloaded == [optimizer]
    model.cpu.assert_called_once()
    model.to.assert_called_once()


def test_callback_falls_back_from_persistent_to_ephemeral(monkeypatch):
    """If the persistent runner raises on its first run, the callback should swap
    in an EphemeralVLLMRunner and retry — without re-running inference twice."""
    sys.modules.setdefault("torch", MagicMock())

    from tuning.training.passk import callback as cb_mod
    from tuning.training.passk import runners as runners_mod
    from tuning.training.config_training import PassAtKConfig

    persistent_run_calls = []
    ephemeral_run_calls = []

    class FakePersistent(runners_mod.PersistentVLLMRunner):
        def run(self, model, eval_strategy, adapter_path, optimizer=None):
            persistent_run_calls.append((adapter_path, optimizer))
            raise RuntimeError("persistent failed")

    class FakeEphemeral(runners_mod.EphemeralVLLMRunner):
        def run(self, model, eval_strategy, adapter_path, optimizer=None):
            ephemeral_run_calls.append((adapter_path, optimizer))
            return [{"prompt": "hi", "responses": ["ok"]}]

    monkeypatch.setattr(cb_mod, "PersistentVLLMRunner", FakePersistent)
    monkeypatch.setattr(cb_mod, "EphemeralVLLMRunner", FakeEphemeral)

    eval_strategy = _make_eval_strategy()
    eval_strategy.score_responses.return_value = {"pass_at_1": 0.5}

    tokenizer = MagicMock()
    tokenizer.chat_template = "t"

    config = PassAtKConfig(
        target_pass_at_k=[0.95],
        use_persistent_vllm=True,
        num_inference_gpus=1,
        enabled=True,
    )

    callback = cb_mod.PassAtKStoppingCallback(
        config=config, tokenizer=tokenizer, model_name="m",
        base_model_hf="m", primary_eval=eval_strategy, monitor_evals=[],
    )
    monkeypatch.setattr(callback, "_save_checkpoint_if_needed",
                        lambda model, adapter_dir: "/tmp/a")

    optimizer = MagicMock()
    scores, results = callback._run_eval_with_results(
        MagicMock(), eval_strategy, optimizer=optimizer,
    )

    assert scores == {"pass_at_1": 0.5}
    assert len(persistent_run_calls) == 1
    assert len(ephemeral_run_calls) == 1
    assert persistent_run_calls[0] == ("/tmp/a", optimizer)
    assert ephemeral_run_calls[0] == ("/tmp/a", optimizer)
    assert isinstance(callback._runner, FakeEphemeral)


def _outputs_for(texts_per_prompt):
    """Build fake vLLM outputs: one entry per prompt, each with its completions."""
    return [SimpleNamespace(outputs=[SimpleNamespace(text=t) for t in texts])
            for texts in texts_per_prompt]


def test_format_outputs_n_samples_override_takes_one_completion():
    es = _make_eval_strategy(n=4)
    es.get_test_prompts.return_value = ["p1", "p2"]
    outputs = _outputs_for([["a1", "a2"], ["b1", "b2"]])

    formatted = VLLMRunner._format_outputs(outputs, es, n_samples=1)

    assert formatted == [{"prompt": "p1", "responses": ["a1"]},
                         {"prompt": "p2", "responses": ["b1"]}]


def test_merge_passes_concatenates_responses_per_prompt_in_seed_order():
    seed_a = [{"prompt": "p1", "responses": ["a1"]},
              {"prompt": "p2", "responses": ["b1"]}]
    seed_b = [{"prompt": "p1", "responses": ["a2"]},
              {"prompt": "p2", "responses": ["b2"]}]

    merged = VLLMRunner.merge_passes([seed_a, seed_b])

    assert merged == [{"prompt": "p1", "responses": ["a1", "a2"]},
                      {"prompt": "p2", "responses": ["b1", "b2"]}]


def test_merge_passes_is_identity_for_a_single_pass():
    single = [{"prompt": "p1", "responses": ["a1"]}]
    assert VLLMRunner.merge_passes([single]) == single


def test_merge_passes_rejects_disagreeing_prompt_sets():
    import pytest
    a = [{"prompt": "p1", "responses": ["a1"]}]
    b = [{"prompt": "p2", "responses": ["b1"]}]
    with pytest.raises(ValueError):
        VLLMRunner.merge_passes([a, b])
