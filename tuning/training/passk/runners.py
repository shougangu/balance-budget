# ABOUTME: VLLMRunner strategy — one runner per inference mode (External / Persistent /
# ABOUTME: Ephemeral / DataParallel). Shared offload context + inference shape live here.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from collections import defaultdict


@dataclass
class RunnerConfig:
    base_model_hf: str
    vllm_gpu_memory_utilization: float
    lora_max_rank: int
    chat_template: str
    temperature: float
    max_tokens: int
    available_gpus: List[str]
    num_inference_gpus: int


class VLLMRunner:
    """Base class. Subclasses override `run`; `cleanup` is optional."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._lora_request_id = 0

    def run(self, model, eval_strategy, adapter_path: Optional[str]) -> List[Dict]:
        raise NotImplementedError

    def cleanup(self) -> None:
        return None

    def _next_lora_request(self, adapter_path: Optional[str]):
        if adapter_path is None:
            return None
        from vllm.lora.request import LoRARequest
        self._lora_request_id += 1
        return LoRARequest(
            lora_name=f"adapter_{self._lora_request_id}",
            lora_int_id=self._lora_request_id,
            lora_path=adapter_path,
        )

    def _run_inference(self, llm, eval_strategy, adapter_path: Optional[str]) -> List[Dict]:
        from vllm import SamplingParams
        from tuning.inference.config_inference import VLLMSamplingParamsConfig

        inference_config = VLLMSamplingParamsConfig(
            n=eval_strategy.n_samples,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        sampling_params = SamplingParams(**inference_config.model_dump())
        lora_request = self._next_lora_request(adapter_path)

        test_messages = eval_strategy.get_test_messages()
        outputs = llm.chat(
            test_messages,
            sampling_params,
            chat_template=self.config.chat_template,
            lora_request=lora_request,
        )
        return self._format_outputs(outputs, eval_strategy)

    @staticmethod
    def _format_outputs(outputs, eval_strategy) -> List[Dict]:
        n_samples = eval_strategy.n_samples
        if n_samples == 1:
            responses = [output.outputs[0].text for output in outputs]
        else:
            responses = [[r.text for r in output.outputs] for output in outputs]
        test_prompts = eval_strategy.get_test_prompts()
        grouped = defaultdict(list)
        for prompt, resp in zip(test_prompts, responses):
            if isinstance(resp, list):
                grouped[prompt].extend(resp)
            else:
                grouped[prompt].append(resp)
        return [{"prompt": p, "responses": resps} for p, resps in grouped.items()]

    @contextmanager
    def _with_model_offloaded(self, model):
        original_device = next(model.parameters()).device
        model.cpu()
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            model.to(original_device)
            model.train()


def _trainer_vllm_sleep_mode_enabled(vllm_generation) -> bool:
    return bool(
        vllm_generation is not None
        and getattr(vllm_generation, "enable_sleep_mode", False)
    )


def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def trainer_vllm_awake_for_passk(llm, vllm_generation):
    """Wake TRL's colocated vLLM for direct pass@k calls, then sleep it."""
    if not _trainer_vllm_sleep_mode_enabled(vllm_generation):
        yield
        return

    _empty_cuda_cache()
    llm.wake_up(tags=["weights"])
    llm.wake_up(tags=["kv_cache"])
    try:
        yield
    finally:
        llm.sleep(level=2)


class ExternalVLLMRunner(VLLMRunner):
    """Uses an externally-provided LLM (e.g. the trainer's own vLLM). No adapter save."""

    def __init__(self, config: RunnerConfig, llm, vllm_generation=None):
        super().__init__(config)
        self._llm = llm
        self._vllm_generation = vllm_generation

    def run(self, model, eval_strategy, adapter_path):
        with trainer_vllm_awake_for_passk(self._llm, self._vllm_generation):
            return self._run_inference(self._llm, eval_strategy, adapter_path=None)


class ServerVLLMRunner(VLLMRunner):
    """Uses TRL's VLLMClient to talk to a trl-vllm-serve HTTP endpoint.

    Single-rank: DDP callers must invoke from rank 0 and broadcast results. The
    server holds the merged base+LoRA weights from the trainer's last sync_weights()
    call, so no adapter save is required.
    """

    def __init__(self, config: RunnerConfig, client, tokenizer):
        super().__init__(config)
        self._client = client
        self._tokenizer = tokenizer

    def run(self, model, eval_strategy, adapter_path):
        if self._client is None:
            raise RuntimeError(
                "ServerVLLMRunner.run() called without a VLLMClient; this should "
                "only happen on rank 0 under DDP."
            )
        import tuning.config as tuning_config

        messages = eval_strategy.get_test_messages()
        prompts = [
            self._tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True,
                chat_template=self.config.chat_template,
            )
            for m in messages
        ]
        n = eval_strategy.n_samples
        # VLLMClient.chat raises on custom chat_template; pre-render and use generate().
        response = self._client.generate(
            prompts=prompts,
            n=n,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            generation_kwargs={"seed": tuning_config.get_eval_seed()},
        )
        completion_ids = response["completion_ids"]  # flat [P*n]
        texts = self._tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        test_prompts = eval_strategy.get_test_prompts()
        grouped = defaultdict(list)
        for i, prompt in enumerate(test_prompts):
            grouped[prompt].extend(texts[i * n:(i + 1) * n])
        return [{"prompt": p, "responses": resps} for p, resps in grouped.items()]


def _make_llm(config: RunnerConfig):
    """Construct a vLLM LLM with our standard LoRA settings.

    enforce_eager=True is required: CUDA-graph capture is incompatible with dynamic
    LoRA adapter swapping.
    """
    from vllm import LLM
    return LLM(
        model=config.base_model_hf,
        enable_lora=True,
        max_lora_rank=config.lora_max_rank,
        max_loras=1,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
    )


def _cleanup_llm(llm):
    """Tear down an ephemeral LLM and free GPU memory."""
    from vllm.distributed.parallel_state import destroy_model_parallel
    from tuning.utils.gpu import cleanup_gpu

    llm.llm_engine.engine_core.shutdown()
    destroy_model_parallel()
    del llm
    cleanup_gpu()


class EphemeralVLLMRunner(VLLMRunner):
    """Creates a fresh vLLM engine per call; offloads training model to CPU."""

    def run(self, model, eval_strategy, adapter_path):
        with self._with_model_offloaded(model):
            llm = _make_llm(self.config)
            try:
                return self._run_inference(llm, eval_strategy, adapter_path)
            finally:
                _cleanup_llm(llm)


class PersistentVLLMRunner(VLLMRunner):
    """Keeps a persistent vLLM engine across calls; swaps LoRA adapters."""

    def __init__(self, config: RunnerConfig):
        super().__init__(config)
        self._llm = None

    def run(self, model, eval_strategy, adapter_path):
        if self._llm is None:
            self._llm = _make_llm(self.config)
        return self._run_inference(self._llm, eval_strategy, adapter_path)

    def cleanup(self):
        if self._llm is None:
            return
        try:
            llm_engine = getattr(self._llm, "llm_engine", None)
            if llm_engine is not None:
                executor = getattr(llm_engine, "model_executor", None)
                if executor is not None:
                    executor.shutdown()
        finally:
            self._llm = None
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            from tuning.utils.gpu import cleanup_gpu
            cleanup_dist_env_and_memory(shutdown_ray=False)
            cleanup_gpu()


def _run_data_parallel(eval_strategy, adapter_path: str, config: RunnerConfig) -> List[Dict]:
    """Spawn N subprocess workers, partition prompts, merge results.

    Lives at module level so closures over `self` aren't accidentally captured.
    """
    import multiprocessing as mp

    from tuning.training.passk.data_parallel import (
        partition_prompts, _data_parallel_worker,
    )
    from tuning.utils.utils import get_stop_tokens
    import tuning.config as tuning_config

    all_messages = eval_strategy.get_test_messages()
    all_prompts = eval_strategy.get_test_prompts()

    available_gpus = config.available_gpus
    num_gpus = config.num_inference_gpus
    if len(available_gpus) < num_gpus:
        print(f"[VLLMRunner] WARNING: requested {num_gpus} inference GPUs but only "
              f"{len(available_gpus)} available ({available_gpus}). "
              f"Using {len(available_gpus)}.")
        num_gpus = len(available_gpus)

    message_chunks = partition_prompts(all_messages, num_gpus)
    prompt_chunks = partition_prompts(all_prompts, num_gpus)
    actual_num_workers = len(message_chunks)

    print(f"[VLLMRunner] Data-parallel: {len(all_messages)} prompts across "
          f"{actual_num_workers} GPUs")
    for i, chunk in enumerate(message_chunks):
        print(f"[VLLMRunner]   Worker {i} → CUDA device {available_gpus[i]}: "
              f"{len(chunk)} prompts")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    stop_tokens = get_stop_tokens()
    eval_seed = tuning_config.get_eval_seed()

    processes = []
    for i in range(actual_num_workers):
        p = ctx.Process(
            target=_data_parallel_worker,
            args=(
                i, available_gpus[i], message_chunks[i], config.base_model_hf,
                adapter_path, eval_strategy.n_samples, config.temperature,
                config.max_tokens, config.chat_template, config.lora_max_rank,
                config.vllm_gpu_memory_utilization, result_queue,
                stop_tokens, eval_seed,
            ),
        )
        p.start()
        processes.append(p)

    results_by_worker = {}
    for _ in range(actual_num_workers):
        worker_id, serialized, error = result_queue.get()
        if error is not None:
            for p in processes:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(f"[VLLMRunner] Worker {worker_id} failed:\n{error}")
        results_by_worker[worker_id] = serialized

    for p in processes:
        p.join(timeout=30)

    merged = []
    for worker_id in range(actual_num_workers):
        chunk_texts = results_by_worker[worker_id]
        chunk_prompts = prompt_chunks[worker_id]
        for prompt, response_texts in zip(chunk_prompts, chunk_texts):
            merged.append({"prompt": prompt, "responses": response_texts})

    grouped = defaultdict(list)
    for item in merged:
        grouped[item["prompt"]].extend(item["responses"])
    return [{"prompt": p, "responses": resps} for p, resps in grouped.items()]


class DataParallelVLLMRunner(VLLMRunner):
    """Spawns N subprocess vLLM workers; offloads training model to CPU."""

    def run(self, model, eval_strategy, adapter_path):
        if adapter_path is None:
            raise ValueError("DataParallelVLLMRunner requires an adapter_path "
                             "(no External-mode equivalent).")
        with self._with_model_offloaded(model):
            return _run_data_parallel(eval_strategy, adapter_path, self.config)
