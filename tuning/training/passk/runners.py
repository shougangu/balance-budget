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


class ExternalVLLMRunner(VLLMRunner):
    """Uses an externally-provided LLM (e.g. the trainer's own vLLM). No adapter save."""

    def __init__(self, config: RunnerConfig, llm):
        super().__init__(config)
        self._llm = llm

    def run(self, model, eval_strategy, adapter_path):
        return self._run_inference(self._llm, eval_strategy, adapter_path=None)
