import torch
import numpy as np
import wandb
import tempfile
import shutil
import os
import json
import datetime
from typing import List, Dict
from pathlib import Path
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from collections import defaultdict
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from instruction_following_eval import evaluation_lib
from tuning.data.test_dataset import get_ifeval_test_dataset
from tuning.config import MODELS_DIR, MODELS_METADATA_DIR
from tuning.inference.config_inference import VLLMSamplingParamsConfig
from tuning.utils.gpu import cleanup_gpu
from tuning.training.callback_utils import save_sweetspot_checkpoint, compute_data_points_seen

BASE_DIR = Path('/home/shougan/projects/aip-fredashi/shougan/balance-budget')
IFEVAL_INPUT_PATH = BASE_DIR / "instruction_following_eval/data/input_data.jsonl"

def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k: probability that at least one of k samples is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def evaluate_single_response(inp: evaluation_lib.InputExample, response: str, strict: bool = True) -> bool:
    """Evaluate a single response using the pre-built IFEval functions."""
    prompt_to_response = {inp.prompt: response}
    if strict:
        result = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    else:
        result = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
    return result.follow_all_instructions


class PassAtKStoppingCallback(TrainerCallback):
    """
    Save checkpoints at pass@k sweetspots for downstream runs.

    Implements the "Fork Strategy": training continues through all thresholds,
    saving checkpoints at each sweetspot without stopping. The final threshold
    in the list will stop training.

    Supports two vLLM modes for inference during training:
    - Persistent mode (default): Keeps vLLM engine alive with base model loaded,
      swaps LoRA adapters each eval. Eliminates cold-start overhead.
    - Non-persistent mode: Creates/destroys vLLM each eval, but still uses
      adapter-only saves instead of full merged model saves.
    """

    def __init__(
        self,
        config,  # PassAtKConfig
        tokenizer,
        model_name: str,
        base_model_hf: str,
    ):
        # Sort thresholds in descending order (hardest to easiest: 0.7, 0.5, 0.3)
        # Higher pass@k = harder to reach, so we process from largest to smallest
        self.target_pass_at_k_thresholds = sorted(config.target_pass_at_k, reverse=True)
        self.patience = config.patience
        self.min_increase = config.min_increase
        self.tokenizer = tokenizer
        self.k_values = config.k_values
        self.stopping_k = self.k_values[0]  # First k value is used for stopping
        self.n_samples = config.n_samples
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.strict = config.strict
        self.model_name = model_name
        self.metadata_path = None
        self.prevResults = []

        # LoRA adapter / persistent vLLM settings
        self.use_persistent_vllm = config.use_persistent_vllm
        self.base_model_hf = base_model_hf
        self.vllm_gpu_memory_utilization = config.vllm_gpu_memory_utilization
        self.lora_max_rank = getattr(config, 'lora_max_rank', 32)
        self._vllm_engine = None
        self._lora_request_id = 0
        self._chat_template = self.tokenizer.chat_template

        self.test_dataset = get_ifeval_test_dataset()
        if config.num_prompts is not None:
            self.test_dataset = self.test_dataset.select(range(min(config.num_prompts, len(self.test_dataset))))

        # Load IFEval inputs for evaluation
        self.inputs_map = {
            inp.prompt: inp
            for inp in evaluation_lib.read_prompt_list(str(IFEVAL_INPUT_PATH))
        }

        mode_str = "persistent" if self.use_persistent_vllm else "non-persistent"
        if not self.patience:
            print(f"[PassAtKCallback] Initialized with pass@{self.stopping_k} thresholds={self.target_pass_at_k_thresholds}")
            print(f"[PassAtKCallback] Training will stop when hardest threshold is reached: {self.target_pass_at_k_thresholds[0]}")
            print(f"[PassAtKCallback] k_values={self.k_values} (stopping on k={self.stopping_k})")
        else:
            print(f"[PassAtKCallback] Initialized with patience={self.patience}, min_increase={self.min_increase}")
            print(f"[PassAtKCallback] Training will stop if pass@{self.stopping_k} does not improve by {self.min_increase} for {self.patience} evaluations in a row")
            print(f"[PassAtKCallback] k_values={self.k_values} (stopping on k={self.stopping_k})")

        print(f"[PassAtKCallback] n_samples={self.n_samples}, temperature={self.temperature}, strict={self.strict}")
        print(f"[PassAtKCallback] IFEval prompts loaded: {len(self.inputs_map)}, num_prompts={len(self.test_dataset)}")
        print(f"[PassAtKCallback] vLLM mode: {mode_str}, base_model_hf={base_model_hf}, gpu_mem={self.vllm_gpu_memory_utilization}")

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.model_name:
            self.model_name = kwargs.get("model")
        print(f"[PassAtKCallback] on_train_begin: model_name={self.model_name}")
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        self.metadata_path = os.path.join(MODELS_METADATA_DIR, f"{self.model_name}_passatk-{now}.json")

    def on_train_end(self, args, state, control, **kwargs):
        """Cleanup persistent vLLM engine when training ends."""
        self._cleanup_vllm()

    def _init_persistent_vllm(self):
        """Lazily initialize the persistent vLLM engine with LoRA support."""
        if self._vllm_engine is not None:
            return

        print(f"[PassAtKCallback] Initializing persistent vLLM engine with base model: {self.base_model_hf}")
        print(f"[PassAtKCallback] gpu_memory_utilization={self.vllm_gpu_memory_utilization}, max_lora_rank={self.lora_max_rank}")

        # enforce_eager=True is required for LoRA — CUDA graph capture is incompatible with dynamic adapter swapping
        self._vllm_engine = LLM(
            model=self.base_model_hf,
            enable_lora=True,
            max_lora_rank=self.lora_max_rank,
            max_loras=1,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
            # max_model_len=2048,
        )

        print(f"[PassAtKCallback] Persistent vLLM engine initialized successfully")

    def _save_lora_adapter(self, model, adapter_dir: str):
        """Save only the LoRA adapter weights (~50MB instead of ~2GB merged)."""
        print(f"[PassAtKCallback] Saving LoRA adapter to {adapter_dir}...")

        # Use standard PEFT save to ensure adapter_config.json is created for vLLM
        if hasattr(model, 'save_pretrained'):
            print(f"[PassAtKCallback] PEFT saving adaptor only")
            # PEFT model - save adapter only
            model.save_pretrained(adapter_dir)
        else:
            # Fallback: use unsloth's method
            print(f"[PassAtKCallback] Model does not have save_pretrained, using merged method with lora save")
            model.save_pretrained_merged(adapter_dir, self.tokenizer, save_method="lora")
        # Save tokenizer so vLLM doesn't warn about missing tokenizer in adapter dir
        
        self.tokenizer.save_pretrained(adapter_dir)
        print(f"[PassAtKCallback] LoRA adapter saved")

    def _run_vllm_inference(self, llm, adapter_path: str = None) -> List[Dict]:
        """Run inference on a vLLM engine, optionally with a LoRA adapter."""
        inference_config = VLLMSamplingParamsConfig(
            n=self.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        sampling_params = SamplingParams(**inference_config.model_dump())

        lora_request = None
        if adapter_path:
            self._lora_request_id += 1
            lora_request = LoRARequest(
                lora_name=f"adapter_{self._lora_request_id}",
                lora_int_id=self._lora_request_id,
                lora_path=adapter_path,
            )

        mode = "persistent" if self.use_persistent_vllm else "ephemeral"
        lora_info = f", lora_id={self._lora_request_id}" if lora_request else ""
        print(f"[PassAtKCallback] Generating {len(self.test_dataset)} prompts x {self.n_samples} samples ({mode}{lora_info})...")

        outputs = llm.chat(
            self.test_dataset["messages"],
            sampling_params,
            chat_template=self._chat_template,
            lora_request=lora_request,
        )

        return self._format_outputs(outputs)

    def _create_ephemeral_vllm(self):
        """Create an ephemeral vLLM engine with LoRA support."""
        print(f"[PassAtKCallback] Loading ephemeral vLLM with base model: {self.base_model_hf}")
        return LLM(
            model=self.base_model_hf,
            enable_lora=True,
            max_lora_rank=self.lora_max_rank,
            max_loras=1,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            trust_remote_code=True,
        )

    def _cleanup_ephemeral_vllm(self, llm):
        """Destroy an ephemeral vLLM engine and free GPU memory."""
        del llm
        cleanup_gpu(destroy_vllm=True)

    def _cleanup_vllm(self):
        """Destroy the persistent vLLM engine and free GPU memory."""
        if self._vllm_engine is not None:
            print(f"[PassAtKCallback] Cleaning up persistent vLLM engine...")
            del self._vllm_engine
            self._vllm_engine = None
            cleanup_gpu(destroy_vllm=True)
            print(f"[PassAtKCallback] vLLM engine cleaned up")

    def _format_outputs(self, outputs) -> List[Dict]:
        """Format vLLM outputs into grouped results for pass@k evaluation."""
        if self.n_samples == 1:
            responses = [output.outputs[0].text for output in outputs]
        else:
            responses = [[response.text for response in output.outputs] for output in outputs]

        grouped = defaultdict(list)
        for prompt, resp in zip(self.test_dataset["prompt"], responses):
            if isinstance(resp, list):
                grouped[prompt].extend(resp)
            else:
                grouped[prompt].append(resp)

        return [{"prompt": p, "responses": resps} for p, resps in grouped.items()]

    def _save_sweetspot_checkpoint(self, model, threshold, state: TrainerState, args: TrainingArguments):
        """Save a checkpoint when a pass@k sweetspot threshold is reached."""
        return save_sweetspot_checkpoint(
            model=model,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            threshold_label=f"p@{self.stopping_k}-{threshold}",
            state=state,
            args=args,
            metadata_path=self.metadata_path,
            extra_metadata={
                "threshold_type": f"pass_at_{self.stopping_k}",
                "threshold_value": threshold,
                "k_value": self.stopping_k,
                "n_samples": self.n_samples,
                "strict": self.strict,
            },
        )

    def evaluate_pass_at_k(self, model) -> Dict[str, float]:
        """Evaluate pass@k using vLLM with LoRA adapter support."""

        temp_dir = tempfile.mkdtemp()

        try:
            self._save_lora_adapter(model, temp_dir)

            if self.use_persistent_vllm:
                # Persistent mode: keep vLLM engine alive, swap LoRA adapters
                try:
                    self._init_persistent_vllm()
                    model_results = self._run_vllm_inference(self._vllm_engine, adapter_path=temp_dir)
                except Exception as e:
                    print(f"[PassAtKCallback] Persistent vLLM failed: {e}, falling back to ephemeral mode")
                    self._cleanup_vllm()
                    self.use_persistent_vllm = False
                    # Fall through to ephemeral path
                    original_device = next(model.parameters()).device
                    model.cpu()
                    torch.cuda.empty_cache()
                    llm = self._create_ephemeral_vllm()
                    model_results = self._run_vllm_inference(llm, adapter_path=temp_dir)
                    self._cleanup_ephemeral_vllm(llm)
                    model.to(original_device)
                    model.train()
            else:
                # Ephemeral mode: create/destroy vLLM each eval
                original_device = next(model.parameters()).device
                model.cpu()
                torch.cuda.empty_cache()
                llm = self._create_ephemeral_vllm()
                model_results = self._run_vllm_inference(llm, adapter_path=temp_dir)
                self._cleanup_ephemeral_vllm(llm)
                model.to(original_device)
                model.train()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Evaluate responses
        print(f"[PassAtKCallback] Evaluating responses...")
        all_results = []
        for item in model_results:
            prompt = item["prompt"]
            responses = item["responses"]


            eval_input = self.inputs_map[prompt]
            results = [evaluate_single_response(eval_input, r, self.strict) for r in responses]
            all_results.append(results)

        scores = {}
        for k in self.k_values:
            pass_at_k_scores = [pass_at_k(len(r), sum(r), k) for r in all_results]
            scores[f"pass_at_{k}"] = np.mean(pass_at_k_scores)
        scores["num_prompts_evaluated"] = len(all_results)

        return scores

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        """Called after evaluation, check pass@k and stop if target reached."""
        train_batch_size = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        world_size = getattr(args, "world_size", 1)
        data_points_seen = state.global_step * train_batch_size * grad_accum * world_size

        if model is None:
            model = kwargs.get("model")
        if model is None:
            print("[PassAtKCallback] Warning: model is None, skipping pass@k check")
            return control

        scores = self.evaluate_pass_at_k(model)

        # Log all k values to wandb
        log_dict = {"train/global_step": state.global_step}
        for k in self.k_values:
            log_dict[f"eval/pass_at_{k}"] = scores[f"pass_at_{k}"]
        wandb.log(log_dict)

        self.prevResults.append(scores[f"pass_at_{self.stopping_k}"])

        eval_type = "strict" if self.strict else "loose"
        scores_str = ", ".join([f"pass@{k}={scores[f'pass_at_{k}']:.4f}" for k in self.k_values])
        print(f"\n[PassAtKCallback] Step {state.global_step}, Data Points {data_points_seen}: "
              f"{scores_str} ({eval_type}, {scores['num_prompts_evaluated']} prompts)")

        # Check each threshold and save checkpoint if crossed (Fork Strategy)
        # Thresholds are sorted descending (hardest to easiest: 0.7, 0.5, 0.3)
        # We iterate to find the hardest threshold that current pass@k has reached
        if not self.patience:
            current_pass_at_k = scores[f"pass_at_{self.stopping_k}"]
            reached_threshold = None
            reached_index = None

            for i, threshold in enumerate(self.target_pass_at_k_thresholds):
                if current_pass_at_k >= threshold:
                    reached_threshold = threshold
                    reached_index = i
                    break

            if reached_threshold is not None:
                print(f"[PassAtKCallback] Sweetspot threshold {reached_threshold} reached!")
                checkpoint_path = self._save_sweetspot_checkpoint(model, reached_threshold, state, args)

                # Trim thresholds to only include harder ones (before current index)
                self.target_pass_at_k_thresholds = self.target_pass_at_k_thresholds[:reached_index]
                print(f"[PassAtKCallback] Remaining thresholds: {self.target_pass_at_k_thresholds}")

                if len(self.target_pass_at_k_thresholds) == 0:
                    print(f"[PassAtKCallback] All thresholds reached! Stopping training.")
                    control.should_training_stop = True
                else:
                    print(f"[PassAtKCallback] Continuing training to next threshold: {self.target_pass_at_k_thresholds[0]}")

        if self.patience:
            if len(self.prevResults) > self.patience:
                early_stopping = True
                for old,new in zip(self.prevResults[-self.patience-1:], self.prevResults[-self.patience:]):
                    if new - old >= self.min_increase:
                        early_stopping = False
                if early_stopping:
                    checkpoint_path = self._save_sweetspot_checkpoint(model, f"{self.patience}@{self.min_increase}", state, args)
                    print(f"[PassAtKCallback] No significant improvement in the last {self.patience} evaluations. Stopping training.")
                    print(f"[PassAtKCallback] Previous pass@{self.stopping_k} scores: {self.prevResults[-self.patience-1:]}")
                    print(f"[PassAtKCallback] Final checkpoint saved at {checkpoint_path}")
                    control.should_training_stop = True

        return control
