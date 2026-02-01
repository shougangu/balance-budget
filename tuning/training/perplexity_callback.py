import os
import json
import datetime
import subprocess

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import wandb

from tuning.config import MODELS_DIR, MODELS_METADATA_DIR

class PerplexityStoppingCallback(TrainerCallback):
    def __init__(
        self, 
        perplexity_thresholds: List[float], 
        test_dataset, 
        tokenizer, 
        num_samples: int = 100,
        model_name: str = None,
        prevWindow: int = None,
    ):
        # Sort thresholds in ascending order (hardest to easiest: 2.0, 2.5, 3.0)
        # Lower perplexity = harder to reach, so we process from smallest to largest
        self.perplexity_thresholds = sorted(perplexity_thresholds, reverse=False)
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.metadata_path = None
        self.model_name = model_name
        self.prevResults = []
        self.prevWindow = prevWindow

        print(f"[PerplexityCallback] Initialized with perplexity_thresholds={self.perplexity_thresholds}")
        print(f"[PerplexityCallback] Training will stop when hardest threshold is reached: {self.perplexity_thresholds[0]}")
        print(f"[PerplexityCallback] num_samples={num_samples}")
        print(f"[PerplexityCallback] Test dataset size: {len(test_dataset)}")
        print(f"Dataset sample 1st one: {test_dataset[0]}")

        
    
    def on_train_begin(self, args, state, control, **kwargs):
        if not self.model_name:
            self.model_name = kwargs.get("model")
        print(f"[PerplexityCallback] on_train_begin: model_name={self.model_name}")
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        self.metadata_path = os.path.join(MODELS_METADATA_DIR, f"{self.model_name}_ppl-{now}.json") 
    
    def _save_sweetspot_checkpoint(self, model, threshold: float, state: TrainerState, args: TrainingArguments):
        train_batch_size = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        world_size = getattr(args, "world_size", 1)
        data_points_seen = state.global_step * train_batch_size * grad_accum * world_size

        # Use data_points_seen as the checkpoint name suffix (e.g., llama3-8B_sft-tuluif-500)
        checkpoint_name = f"{self.model_name}_ppl-{threshold:.2f}_sft-{data_points_seen}"
        if self.prevWindow:
            checkpoint_name = f"{self.model_name}_pplWindow-{self.prevWindow}_sft-{data_points_seen}"
        checkpoint_path = os.path.join(MODELS_DIR, checkpoint_name)

        print(f"[PerplexityCallback] Saving sweetspot checkpoint to {checkpoint_path}")
        
        # Save using Unsloth's merged 16bit method (consistent with PassAtKCallback)
        model.save_pretrained_merged(checkpoint_path, self.tokenizer, save_method="merged_16bit")
        
        with open(f"{checkpoint_path}/training_config.json", "w") as f:
            json.dump(args.to_dict(), f, indent=4)

        # Write metadata marker file for orchestrator/downstream DPO
        metadata = {
            "threshold_type": "perplexity",
            "threshold_value": threshold,
            "global_step": state.global_step,
            "checkpoint_path": checkpoint_path,
            "data_points_seen": data_points_seen,
        }
        with open(self.metadata_path, "a") as f:
            f.write(json.dumps(metadata)+"\n")
        
        print(f"[PerplexityCallback] Sweetspot checkpoint saved with metadata at {self.metadata_path}")
        return checkpoint_path
    
    def compute_perplexity(self, log_probs):
        if len(log_probs) == 0:
            return float('inf')
        avg_neg_logp = -sum(log_probs) / len(log_probs)
        return np.exp(avg_neg_logp)
    
    def evaluate_perplexity(self, model):
        """Evaluate average perplexity on test samples."""
        model.eval()
        perplexities = []
        
        samples = min(self.num_samples, len(self.test_dataset))
        
        for i in range(samples):
            messages = self.test_dataset[i]["messages"]
            prompt_messages = [messages[0], messages[1]]  # system + user
            response_message = messages[2]  # assistant


            prompt_str = self.tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = self.tokenizer.encode(
                prompt_str, return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            
            messages_with_response = prompt_messages + [response_message]
            full_text = self.tokenizer.apply_chat_template(
                messages_with_response, tokenize=False, add_generation_prompt=False
            )
            full_input_ids = self.tokenizer.encode(
                full_text, return_tensors="pt", add_special_tokens=False
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model(full_input_ids)
                logits = outputs.logits
            
            response_start_index = prompt_ids.shape[1]
            relevant_logits = logits[0, response_start_index - 1:-1, :]
            response_ids = full_input_ids[0, response_start_index:]
            
            log_probs = F.log_softmax(relevant_logits, dim=-1)
            response_log_probs = []
            for i in range(len(response_ids)):
                token_id_i = response_ids[i].item()
                log_prob_i = log_probs[i, token_id_i].item()
                response_log_probs.append(log_prob_i)
            perplexities.append(self.compute_perplexity(response_log_probs))
        
        model.train()
        return np.mean(perplexities)
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, 
                    control: TrainerControl, model=None, **kwargs):
        """Called after evaluation, check perplexity and stop if target reached."""
        # Model might be passed directly, via kwargs, or captured from on_train_begin
        train_batch_size = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        world_size = getattr(args, "world_size", 1)
        data_points_seen = state.global_step * train_batch_size * grad_accum * world_size
        
        if model is None:
            model = kwargs.get("model")
        if model is None:
            model = self._trainer
        if model is None:
            print("[PerplexityCallback] Warning: model is None, skipping ppl check")
            return control
        
        current_perplexity = self.evaluate_perplexity(model)
        wandb.log({"eval/perplexity": current_perplexity, "train/global_step": state.global_step})
        self.prevResults.append(current_perplexity)
        
        print(f"\n[PerplexityCallback] Step {state.global_step}, Data Points {data_points_seen}: PPL = {current_perplexity:.4f}")
        
        # Check each threshold and save checkpoint if crossed (Fork Strategy)
        # Thresholds are sorted ascending (hardest to easiest: 2.0, 2.5, 3.0)
        # We iterate to find the hardest threshold that current_perplexity has reached
        if not self.prevWindow:
            reached_threshold = None
            reached_index = None

            for i, threshold in enumerate(self.perplexity_thresholds):
                if current_perplexity <= threshold:
                    reached_threshold = threshold
                    reached_index = i
                    break

            if reached_threshold is not None:
                print(f"[PerplexityCallback] Sweetspot threshold {reached_threshold} reached!")

                checkpoint_path = self._save_sweetspot_checkpoint(model, reached_threshold, state, args)

                dpo_data = 8192 - data_points_seen  # Let's say we have 8192 datapoint budget
                print(f"[PerplexityCallback] Launching DPO job with data points {dpo_data} at checkpoint {checkpoint_path}")
                # subprocess.run(["sbatch", "/home/shougan/projects/aip-fredashi/shougan/balance-budget/tuning/slurm/run_dpo.sh", self.model_name, "tuluif", str(dpo_data), checkpoint_path, str(data_points_seen), "ifeval", "dpo"], env=os.environ.copy())

                self.perplexity_thresholds = self.perplexity_thresholds[:reached_index]
                print(f"[PerplexityCallback] Remaining thresholds: {self.perplexity_thresholds}")

                if len(self.perplexity_thresholds) == 0:
                    print(f"[PerplexityCallback] All thresholds reached! Stopping training.")
                    control.should_training_stop = True
                else:
                    print(f"[PerplexityCallback] Continuing training to next threshold: {self.perplexity_thresholds[0]}")
        
        if self.prevWindow:
            if len(self.prevResults) > self.prevWindow:
                if self.prevResults[-self.prevWindow] == min(self.prevResults[-self.prevWindow:]):
                    print(f"[PerplexityCallback] No improvement in last {self.prevWindow} evaluations. Stopping training.")
                    self._save_sweetspot_checkpoint(model, threshold, state, args)
                    control.should_training_stop = True
        return control
