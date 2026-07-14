# ABOUTME: VLLMRunner strategy — one runner per inference mode (External / Persistent /
# ABOUTME: Ephemeral / DataParallel). Shared offload context + inference shape live here.

from contextlib import contextmanager
from dataclasses import dataclass
import ctypes as ct
from functools import lru_cache
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from collections import defaultdict

from tuning.training.config_training import DEFAULT_VLLM_MAX_MODEL_LEN
from tuning.training.model_utils import is_adapter_checkpoint


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
    max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN


def _resolve_checkpoint(base_model_hf: str, checkpoint_path: Optional[str]):
    """Map an eval checkpoint dir to (engine model path, LoRA adapter path).

    Adapter checkpoints are served as base model + attached adapter; full-model
    checkpoints (no adapter_config.json) are served directly with no LoRA.
    """
    if checkpoint_path is None:
        return base_model_hf, None
    if is_adapter_checkpoint(checkpoint_path):
        return base_model_hf, checkpoint_path
    return checkpoint_path, None


class VLLMRunner:
    """Base class. Subclasses override `run`; `cleanup` is optional."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._lora_request_id = 0

    def run(
        self, model, eval_strategy, checkpoint_path: Optional[str], optimizer=None,
    ) -> List[Dict]:
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


def _unwrap_paged_optimizer(optimizer):
    """Find a bitsandbytes paged optimizer through Accelerate wrappers."""
    seen = set()
    current = optimizer
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        page_manager = getattr(current, "page_mng", None)
        if (
            getattr(current, "is_paged", False)
            and page_manager is not None
            and callable(getattr(page_manager, "prefetch_all", None))
        ):
            return current
        current = getattr(current, "optimizer", None)
    return None


class _CUMemLocation(ct.Structure):
    """ctypes mirror of CUDA's two-int CUmemLocation structure."""

    _fields_ = [("type", ct.c_int), ("id", ct.c_int)]


_CU_MEM_LOCATION_TYPE_HOST = 0x2
_CU_DEVICE_CPU = -1


@lru_cache(maxsize=1)
def _load_cuda_driver():
    """Load the small subset of the CUDA driver API used for UVM prefetch."""
    try:
        driver = ct.CDLL("libcuda.so.1")
    except OSError as exc:
        raise RuntimeError(
            "Cannot offload the paged optimizer because libcuda.so.1 could not "
            "be loaded"
        ) from exc

    driver.cuGetErrorString.argtypes = [ct.c_int, ct.POINTER(ct.c_char_p)]
    driver.cuGetErrorString.restype = ct.c_int

    prefetch_v2 = getattr(driver, "cuMemPrefetchAsync_v2", None)
    if prefetch_v2 is not None:
        prefetch_v2.argtypes = [
            ct.c_uint64,
            ct.c_size_t,
            _CUMemLocation,
            ct.c_uint,
            ct.c_void_p,
        ]
        prefetch_v2.restype = ct.c_int
    else:
        # CUDA < 12.2 exposes only the legacy API, where -1 is the documented
        # CPU destination. Keep this fallback so paged eval is not CUDA-13-only.
        prefetch_legacy = driver.cuMemPrefetchAsync
        prefetch_legacy.argtypes = [
            ct.c_uint64,
            ct.c_size_t,
            ct.c_int,
            ct.c_void_p,
        ]
        prefetch_legacy.restype = ct.c_int

    return driver


def _cuda_error_string(driver, result: int) -> str:
    message = ct.c_char_p()
    if driver.cuGetErrorString(result, ct.byref(message)) == 0 and message.value:
        return message.value.decode("utf-8", errors="replace")
    return f"CUDA driver error {result}"


def _prefetch_managed_tensors_to_cpu(tensors) -> None:
    """Prefetch bitsandbytes CUDA-managed allocations to host memory.

    bitsandbytes prefetch_all(to_cpu=True) passes device -1 through a native
    wrapper that first queries it as a GPU ordinal. On CUDA 13 that wrapper also
    builds a device-type location with id -1. Both paths abort natively with
    invalid device ordinal instead of raising in Python.

    Calling the CUDA driver v2 API directly lets us describe the destination as
    host memory. The legacy driver fallback accepts CU_DEVICE_CPU (-1).
    """
    if not tensors:
        return

    driver = _load_cuda_driver()
    prefetch_v2 = getattr(driver, "cuMemPrefetchAsync_v2", None)
    host = _CUMemLocation(type=_CU_MEM_LOCATION_TYPE_HOST, id=0)

    # Match GlobalPageManager.prefetch_all's reverse order: tensors needed first
    # by the next optimizer step are migrated last and are less likely to churn.
    for tensor in reversed(tensors):
        nbytes = int(getattr(tensor, "nbytes", 0))
        if nbytes == 0:
            continue
        if not getattr(tensor, "is_paged", False):
            raise RuntimeError(
                "bitsandbytes page manager contained a non-paged tensor"
            )

        pointer = int(tensor.data_ptr())
        if prefetch_v2 is not None:
            result = prefetch_v2(pointer, nbytes, host, 0, None)
        else:
            result = driver.cuMemPrefetchAsync(
                pointer, nbytes, _CU_DEVICE_CPU, None,
            )
        if result != 0:
            raise RuntimeError(
                "CUDA failed to prefetch paged optimizer state to host: "
                f"{_cuda_error_string(driver, result)}"
            )


def _offload_paged_optimizer_state(optimizer) -> bool:
    """Migrate initialized bitsandbytes optimizer pages to CPU for vLLM eval.

    The next bitsandbytes optimizer step prefetches each state tensor back to its
    CUDA device as it is used, so eagerly restoring every page after eval would
    only increase peak memory before training resumes.
    """
    paged_optimizer = _unwrap_paged_optimizer(optimizer)
    if paged_optimizer is None:
        if optimizer is not None:
            print("[VLLMRunner] Optimizer is not paged; state remains on its current device")
        return False

    page_manager = paged_optimizer.page_mng
    paged_tensors = list(getattr(page_manager, "paged_tensors", ()))
    paged_bytes = sum(
        int(getattr(tensor, "nbytes", 0))
        for tensor in {id(tensor): tensor for tensor in paged_tensors}.values()
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free_before, total = torch.cuda.mem_get_info()
    else:
        free_before = total = None

    # Avoid bitsandbytes' native CPU=-1 path, which aborts on CUDA 13.
    _prefetch_managed_tensors_to_cpu(paged_tensors)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free_after, _ = torch.cuda.mem_get_info()
        print(
            "[VLLMRunner] Paged optimizer state prefetched to CPU: "
            f"{paged_bytes / 1024**3:.2f} GiB managed, "
            f"GPU free {free_before / 1024**3:.2f} -> "
            f"{free_after / 1024**3:.2f} GiB "
            f"of {total / 1024**3:.2f} GiB"
        )
    else:
        print(
            "[VLLMRunner] Paged optimizer state prefetched to CPU: "
            f"{paged_bytes / 1024**3:.2f} GiB managed"
        )
    return True


def _trainer_vllm_sleep_mode_enabled(vllm_generation) -> bool:
    return bool(
        vllm_generation is not None
        and getattr(vllm_generation, "enable_sleep_mode", False)
    )


def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _invalidate_trainer_vllm_step_cache(trainer):
    if trainer is not None and hasattr(trainer, "_last_loaded_step"):
        trainer._last_loaded_step = -1


@contextmanager
def trainer_vllm_awake_for_passk(llm, vllm_generation, trainer=None):
    """Refresh TRL's colocated vLLM for direct pass@k calls, then sleep it."""
    if vllm_generation is None: # ie, no colocated gpu that must be synced
        try:
            yield
        finally:
            _invalidate_trainer_vllm_step_cache(trainer)
        return

    sync_weights = getattr(vllm_generation, "sync_weights", None)
    if not callable(sync_weights):
        raise RuntimeError(
            "Pass@k eval is reusing a trainer vLLM object without sync_weights(). "
            "Sleeping colocated vLLM engines must be refreshed through the trainer "
            "generation path before direct eval generation."
        )

    sleep_mode_enabled = _trainer_vllm_sleep_mode_enabled(vllm_generation)
    if sleep_mode_enabled:
        _empty_cuda_cache()
    try:
        sync_weights()
        if sleep_mode_enabled:
            llm.wake_up(tags=["kv_cache"])
        yield
    finally:
        if sleep_mode_enabled:
            llm.sleep(level=2)
        _invalidate_trainer_vllm_step_cache(trainer)


class ExternalVLLMRunner(VLLMRunner):
    """Uses an externally-provided LLM (e.g. the trainer's own vLLM). No adapter save."""

    def __init__(self, config: RunnerConfig, llm, vllm_generation=None, trainer=None):
        super().__init__(config)
        self._llm = llm
        self._vllm_generation = vllm_generation
        self._trainer = trainer

    @contextmanager
    def awake_for_passk(self):
        with trainer_vllm_awake_for_passk(
            self._llm, self._vllm_generation, trainer=self._trainer,
        ):
            yield self._llm

    def run(self, model, eval_strategy, checkpoint_path, optimizer=None):
        with self.awake_for_passk() as llm:
            return self._run_inference(llm, eval_strategy, adapter_path=None)


class ServerVLLMRunner(VLLMRunner):
    """Uses TRL's VLLMClient to talk to a trl-vllm-serve HTTP endpoint.

    Single-rank: DDP callers must invoke from rank 0 and broadcast results. The
    server holds the merged base+LoRA weights from the trainer's sync_weights()
    call, so no adapter save is required.
    """

    def __init__(
        self, config: RunnerConfig, client, tokenizer, vllm_generation=None,
        trainer=None,
    ):
        super().__init__(config)
        self._client = client
        self._tokenizer = tokenizer
        self._vllm_generation = vllm_generation
        self._trainer = trainer

    def sync_weights(self):
        if self._vllm_generation is None:
            return

        sync_weights = getattr(self._vllm_generation, "sync_weights", None)
        if not callable(sync_weights):
            raise RuntimeError(
                "Pass@k eval is reusing a trainer vLLM server without "
                "sync_weights(). Server-mode eval must refresh the HTTP server "
                "through TRL's VLLMGeneration before direct generation."
            )
        sync_weights()
        _invalidate_trainer_vllm_step_cache(self._trainer)

    def run(self, model, eval_strategy, checkpoint_path, optimizer=None):
        if self._client is None:
            raise RuntimeError(
                "ServerVLLMRunner.run() called without a VLLMClient; this should "
                "only happen on rank 0 under DDP."
            )
        import tuning.config as tuning_config

        if not (dist.is_initialized() and dist.get_world_size() > 1):
            self.sync_weights()

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


def _make_llm(config: RunnerConfig, model_path: Optional[str] = None, enable_lora: bool = True):
    """Construct a vLLM LLM serving model_path (default: the base model).

    enforce_eager=True is required: CUDA-graph capture is incompatible with dynamic
    LoRA adapter swapping. Full-model checkpoints disable LoRA support entirely.
    """
    from vllm import LLM
    lora_kwargs = {}
    if enable_lora:
        lora_kwargs = {
            "enable_lora": True,
            "max_lora_rank": config.lora_max_rank,
            "max_loras": 1,
        }
    return LLM(
        model=model_path or config.base_model_hf,
        gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        max_model_len=config.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        **lora_kwargs,
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

    def run(self, model, eval_strategy, checkpoint_path, optimizer=None):
        model_path, lora_path = _resolve_checkpoint(
            self.config.base_model_hf, checkpoint_path)
        with self._with_model_offloaded(model):
            _offload_paged_optimizer_state(optimizer)
            llm = _make_llm(
                self.config, model_path=model_path, enable_lora=lora_path is not None)
            try:
                return self._run_inference(llm, eval_strategy, lora_path)
            finally:
                _cleanup_llm(llm)


class PersistentVLLMRunner(VLLMRunner):
    """Keeps a persistent vLLM engine across calls; swaps LoRA adapters."""

    def __init__(self, config: RunnerConfig):
        super().__init__(config)
        self._llm = None

    def run(self, model, eval_strategy, checkpoint_path, optimizer=None):
        if checkpoint_path is not None and not is_adapter_checkpoint(checkpoint_path):
            raise RuntimeError(
                "Persistent vLLM only swaps LoRA adapters and cannot serve the "
                f"full-model checkpoint {checkpoint_path}; use ephemeral mode "
                "(use_persistent_vllm=False) for full fine-tuning."
            )
        if self._llm is None:
            self._llm = _make_llm(self.config)
        return self._run_inference(self._llm, eval_strategy, checkpoint_path)

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


def _run_data_parallel(eval_strategy, checkpoint_path: str, config: RunnerConfig) -> List[Dict]:
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
    model_path, lora_path = _resolve_checkpoint(config.base_model_hf, checkpoint_path)

    processes = []
    for i in range(actual_num_workers):
        p = ctx.Process(
            target=_data_parallel_worker,
            args=(
                i, available_gpus[i], message_chunks[i], model_path,
                lora_path, eval_strategy.n_samples, config.temperature,
                config.max_tokens, config.chat_template, config.lora_max_rank,
                config.vllm_gpu_memory_utilization, config.max_model_len, result_queue,
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

    def run(self, model, eval_strategy, checkpoint_path, optimizer=None):
        if checkpoint_path is None:
            raise ValueError("DataParallelVLLMRunner requires a checkpoint_path "
                             "(no External-mode equivalent).")
        with self._with_model_offloaded(model):
            _offload_paged_optimizer_state(optimizer)
            return _run_data_parallel(eval_strategy, checkpoint_path, self.config)
