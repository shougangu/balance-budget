import torch
import numpy as np
import wandb
import tempfile
import shutil
import gc
import os
import json
import datetime
from typing import List, Dict
from pathlib import Path
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from collections import defaultdict
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel

from instruction_following_eval import evaluation_lib
from tuning.data.test_dataset import get_ifeval_test_dataset
from tuning.utils.utils import chat_template_func
from tuning.config import MODELS_DIR, MODELS_METADATA_DIR

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
    """
    
    def __init__(
        self, 
        target_pass_at_k: List[float], 
        tokenizer, 
        k_values: List[int] = [1],
        n_samples: int = 16,
        num_prompts: int = 50,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        strict: bool = False,
        model_name: str = None,
        prevWindow: int = None,
    ):
        # Sort thresholds in descending order (hardest to easiest: 0.7, 0.5, 0.3)
        # Higher pass@k = harder to reach, so we process from largest to smallest
        self.target_pass_at_k_thresholds = sorted(target_pass_at_k, reverse=True)
        self.tokenizer = tokenizer
        self.k_values = k_values
        self.stopping_k = self.k_values[0]  # First k value is used for stopping
        self.n_samples = n_samples
        self.num_prompts = num_prompts
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.strict = strict
        self.model_name = model_name
        self.metadata_path = None
        self.prevResults = []
        self.prevWindow = prevWindow
        
        self.test_dataset = get_ifeval_test_dataset()
        if num_prompts is not None:
            self.test_dataset = self.test_dataset.select(range(min(num_prompts, len(self.test_dataset))))
        
        # Load IFEval inputs for evaluation
        self.inputs_map = {
            inp.prompt: inp 
            for inp in evaluation_lib.read_prompt_list(str(IFEVAL_INPUT_PATH))
        }
        
        print(f"[PassAtKCallback] Initialized with pass@{self.stopping_k} thresholds={self.target_pass_at_k_thresholds}")
        print(f"[PassAtKCallback] Training will stop when hardest threshold is reached: {self.target_pass_at_k_thresholds[0]}")
        print(f"[PassAtKCallback] k_values={self.k_values} (stopping on k={self.stopping_k})")
        print(f"[PassAtKCallback] n_samples={n_samples}, temperature={temperature}, strict={strict}")
        print(f"[PassAtKCallback] IFEval prompts loaded: {len(self.inputs_map)}, num_prompts={len(self.test_dataset)}")
        print(f"[PassAtKCallback] Using vLLM with model save/load (replicating run_inference_ifeval)")
    def on_train_begin(self, args, state, control, **kwargs):
        if not self.model_name:
            self.model_name = kwargs.get("model")
        print(f"[PassAtKCallback] on_train_begin: model_name={self.model_name}")
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        self.metadata_path = os.path.join(MODELS_METADATA_DIR, f"{self.model_name}_passatk-{now}.json")
    
    def _save_sweetspot_checkpoint(self, model, threshold: float, state: TrainerState, args: TrainingArguments):
        """Save a checkpoint when a pass@k sweetspot threshold is reached."""
        train_batch_size = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        world_size = getattr(args, "world_size", 1)
        data_points_seen = int(state.global_step * train_batch_size * grad_accum * world_size / 2)

        # Use data_points_seen as the checkpoint name suffix
        checkpoint_name = f"{self.model_name}_pass@{self.stopping_k}-{threshold:.2f}_sft-{data_points_seen}"
        checkpoint_path = os.path.join(MODELS_DIR, checkpoint_name)

        print(f"[PassAtKCallback] Saving sweetspot checkpoint to {checkpoint_path}")
        
        # Save using Unsloth's merged 16bit method (consistent with PerplexityCallback)
        model.save_pretrained_merged(checkpoint_path, self.tokenizer, save_method="merged_16bit")
        
        with open(f"{checkpoint_path}/training_config.json", "w") as f:
            json.dump(args.to_dict(), f, indent=4)

        # Write metadata marker file for orchestrator/downstream DPO
        metadata = {
            "threshold_type": f"pass_at_{self.stopping_k}",
            "threshold_value": threshold,
            "global_step": state.global_step,
            "checkpoint_path": checkpoint_path,
            "data_points_seen": data_points_seen,
            "k_value": self.stopping_k,
            "n_samples": self.n_samples,
            "strict": self.strict,
        }
        
        with open(self.metadata_path, "a") as f:
            f.write(json.dumps(metadata)+"\n")
        
        print(f"[PassAtKCallback] Sweetspot checkpoint saved with metadata at {self.metadata_path}")
        return checkpoint_path
    
    def _run_inference_with_vllm(self, model_path: str) -> List[Dict]:
        """
        Run inference using vLLM, replicating the pattern from run_inference_ifeval.
        Returns list of dicts: [{"prompt": str, "responses": List[str]}, ...]
        """
        # Load model with vLLM (similar to load_vlm_model)
        print(f"[PassAtKCallback] Loading model with vLLM from {model_path}...")
        print("LLM UTILISATION IS 0.8")

        llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
        )
        
        sampling_params = SamplingParams(
            n=self.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        
        # Get tokenizer from vLLM and apply chat template (like in vllm_utils.py)
        tokenizer = llm.get_tokenizer()
        tokenizer = chat_template_func(tokenizer)
        chat_template = tokenizer.chat_template
        
        # Generate responses using llm.chat (like make_vllm_call)
        print(f"[PassAtKCallback] Generating {len(self.test_dataset)} prompts x {self.n_samples} samples...")
        outputs = llm.chat(
            self.test_dataset["messages"], 
            sampling_params, 
            chat_template=chat_template
        )
        
        # Format results (like generate_responses_vllm)
        # Group responses by prompt for pass@k evaluation
        if self.n_samples == 1:
            responses = [output.outputs[0].text for output in outputs]
        else:
            responses = [[response.text for response in output.outputs] for output in outputs]
        
        # Group by prompt (like in calculate_pass@k.py run_inference)
        grouped = defaultdict(list)
        for prompt, resp in zip(self.test_dataset["prompt"], responses):
            if isinstance(resp, list):
                grouped[prompt].extend(resp)
            else:
                grouped[prompt].append(resp)
        
        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


        model_results = [{"prompt": p, "responses": resps} for p, resps in grouped.items()]
        
        return model_results
    
    def evaluate_pass_at_k(self, model) -> Dict[str, float]:
        """Evaluate pass@k using vLLM with model save/load."""
        
        # Step 1: Save model to temp directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            print(f"[PassAtKCallback] Saving model to {temp_dir}...")
            
            model.save_pretrained_merged(temp_dir, self.tokenizer, save_method="merged_16bit")
            
            print(f"[PassAtKCallback] Moving training model to CPU...")
            original_device = next(model.parameters()).device
            model.cpu()
            torch.cuda.empty_cache()
            
            model_results = self._run_inference_with_vllm(temp_dir)
            
            print(f"[PassAtKCallback] Moving training model back to GPU...")
            model.to(original_device)
            model.train()  # Ensure model is back in training mode
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        # Step 5: Evaluate responses (like evaluate_pass_at_k in calculate_pass@k.py)
        print(f"[PassAtKCallback] Evaluating responses...")
        all_results = []
        for item in model_results:
            prompt = item["prompt"]
            responses = item["responses"]
            
            if prompt not in self.inputs_map:
                print(f"[PassAtKCallback] Warning: Prompt not found in inputs_map: {prompt[:50]}...")
                continue
            
            eval_input = self.inputs_map[prompt]
            results = [evaluate_single_response(eval_input, r, self.strict) for r in responses]
            all_results.append(results)
        
        # Compute pass@k scores for all k values
        if not all_results:
            scores = {f"pass_at_{k}": 0.0 for k in self.k_values}
            scores["num_prompts_evaluated"] = 0
            return scores
        
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
            model = self._trainer
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
        if not self.prevWindow:
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

                dpo_data = 8192 - data_points_seen  # Let's say we have 8192 datapoint budget
                print(f"[PassAtKCallback] Launching DPO job with data points {dpo_data} at checkpoint {checkpoint_path}")

                # Trim thresholds to only include harder ones (before current index)
                self.target_pass_at_k_thresholds = self.target_pass_at_k_thresholds[:reached_index]
                print(f"[PassAtKCallback] Remaining thresholds: {self.target_pass_at_k_thresholds}")

                if len(self.target_pass_at_k_thresholds) == 0:
                    print(f"[PassAtKCallback] All thresholds reached! Stopping training.")
                    control.should_training_stop = True
                else:
                    print(f"[PassAtKCallback] Continuing training to next threshold: {self.target_pass_at_k_thresholds[0]}")

        if self.prevWindow:
            if len(self.prevResults) > self.prevWindow:
                if self.prevResults[-self.prevWindow] == max(self.prevResults[-self.prevWindow:]):
                    print(f"[PassAtKCallback] No improvement in last {self.prevWindow} evaluations. Stopping training.")
                    control.should_training_stop = True
                
        return control