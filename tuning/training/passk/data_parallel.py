# ABOUTME: Helpers for data-parallel vLLM inference across multiple GPUs.
# ABOUTME: _data_parallel_worker is a subprocess entry point — keep top-level (no closures).

import os
from typing import List

import torch


def partition_prompts(messages: List, num_chunks: int) -> List[List]:
    """Split a list of messages into num_chunks roughly-equal chunks.

    If num_chunks > len(messages), only len(messages) chunks are returned (1 item each).
    """
    n = len(messages)
    num_chunks = min(num_chunks, n)
    chunks = []
    base_size = n // num_chunks
    remainder = n % num_chunks
    start = 0
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        chunks.append(messages[start:start + size])
        start += size
    return chunks


def _data_parallel_worker(worker_id, cuda_device, messages_chunk, model_path, lora_path,
                          n_samples, temperature, max_tokens, chat_template,
                          lora_max_rank, gpu_memory_utilization, max_model_len, result_queue,
                          stop_tokens=None, seed=None):
    """Worker function for data-parallel vLLM inference. Runs in a subprocess.

    Each worker pins itself to a single GPU, creates an ephemeral vLLM engine,
    runs inference on its chunk of prompts, and returns serialized outputs.
    lora_path attaches an adapter on top of model_path; None serves model_path as-is
    (full-model checkpoints).

    Args:
        worker_id: Logical worker index (0, 1, 2...) used for result ordering.
        cuda_device: The actual CUDA device string (e.g. "3") from SLURM allocation.
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

        from vllm import LLM, SamplingParams

        lora_kwargs = {}
        if lora_path is not None:
            lora_kwargs = {
                "enable_lora": True,
                "max_lora_rank": lora_max_rank,
                "max_loras": 1,
            }
        llm = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            **lora_kwargs,
        )

        from tuning.inference.config_inference import VLLMSamplingParamsConfig
        inference_config = VLLMSamplingParamsConfig(
            n=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_tokens or [],
            seed=seed,
        )
        sampling_params = SamplingParams(**inference_config.model_dump())

        lora_request = None
        if lora_path is not None:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(
                lora_name=f"adapter_worker{worker_id}",
                lora_int_id=1,
                lora_path=lora_path,
            )

        outputs = llm.chat(
            messages_chunk,
            sampling_params,
            chat_template=chat_template,
            lora_request=lora_request,
        )

        serialized = []
        for output in outputs:
            texts = [resp.text for resp in output.outputs]
            serialized.append(texts)

        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        result_queue.put((worker_id, serialized, None))
    except Exception:
        import traceback
        result_queue.put((worker_id, None, traceback.format_exc()))
