import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import wandb
from tuning.config import MODELS_DIR
import os

class PerplexityStoppingCallback(TrainerCallback):
    """
    Save checkpoints at perplexity sweetspots for downstream DPO runs.
    
    Implements the "Fork Strategy": training continues through all thresholds,
    saving checkpoints at each sweetspot without stopping. The final threshold
    in the list will stop training.
    """
    
    def __init__(
        self, 
        perplexity_thresholds: List[float], 
        test_dataset, 
        tokenizer, 
        num_samples: int = 100,
        model_name: str = None,
        output_dir: str = None,
    ):
        """
        Args:
            perplexity_thresholds: List of perplexity targets (descending order, e.g., [3.0, 2.5, 2.0]).
                                   Checkpoints saved when each is reached. Training stops at the last one.
            test_dataset: Dataset to evaluate perplexity on.
            tokenizer: Tokenizer for the model.
            num_samples: Number of samples to use for perplexity evaluation.
            output_dir: Directory to save sweetspot checkpoints. If None, uses trainer's output_dir.
        """
        # Sort thresholds in descending order (higher perplexity = easier to reach first)
        self.perplexity_thresholds = sorted(perplexity_thresholds, reverse=False)
        self.completed_thresholds = set()  # Track which thresholds have been crossed
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.model_name = model_name
        self._trainer = None  # Will be set when training starts

        
        print(f"[PerplexityCallback] Initialized with perplexity_thresholds={self.perplexity_thresholds}")
        print(f"[PerplexityCallback] Training will stop at final threshold: {self.perplexity_thresholds[-1]}")
        print(f"[PerplexityCallback] num_samples={num_samples}")
        print(f"[PerplexityCallback] Test dataset size: {len(test_dataset)}")
        print(f"Dataset sample 1st one: {test_dataset[0]}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Capture trainer reference at the start of training."""
        self._trainer = kwargs.get("model")
        # Set output_dir from training args if not provided
        if self.output_dir is None:
            self.output_dir = args.output_dir
    
    def _save_sweetspot_checkpoint(self, model, threshold: float, state: TrainerState, args: TrainingArguments):
        train_batch_size = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        world_size = getattr(args, "world_size", 1)
        data_points_seen = state.global_step * train_batch_size * grad_accum * world_size

        # Use data_points_seen as the checkpoint name suffix (e.g., llama3-8B_sft-tuluif-500)
        checkpoint_name = f"{self.model_name}_ppl-{threshold:.2f}_sft-{data_points_seen}"
        checkpoint_path = os.path.join(MODELS_DIR, checkpoint_name)

        print(f"[PerplexityCallback] Saving sweetspot checkpoint to {checkpoint_path}")
        
        # Save using Unsloth's merged 16bit method (consistent with PassAtKCallback)
        model.save_pretrained_merged(checkpoint_path, self.tokenizer, save_method="merged_16bit")
        
        # Write metadata marker file for orchestrator/downstream DPO
        metadata = {
            "threshold_type": "perplexity",
            "threshold_value": threshold,
            "global_step": state.global_step,
            "checkpoint_path": checkpoint_path,
            "data_points_seen": data_points_seen,
        }
        import json
        metadata_path = os.path.join(checkpoint_path, "sweetspot_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[PerplexityCallback] Sweetspot checkpoint saved with metadata at {metadata_path}")
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
            print("[PerplexityCallback] Warning: model is None, skipping perplexity check")
            return control
        
        current_perplexity = self.evaluate_perplexity(model)
        wandb.log({"eval/perplexity": current_perplexity, "train/global_step": state.global_step})
        
        print(f"\n[PerplexityCallback] Step {state.global_step}, Data Points {data_points_seen}: PPL = {current_perplexity:.4f}")
        
        # Check each threshold and save checkpoint if crossed (Fork Strategy)
        for threshold in self.perplexity_thresholds:
            if threshold in self.completed_thresholds or current_perplexity > threshold:
                continue
            
            print(f"[PerplexityCallback] Sweetspot threshold {threshold} reached!")
            self.completed_thresholds.add(threshold)
            
            # Save the checkpoint and launch DPO job
            checkpoint_path = self._save_sweetspot_checkpoint(model, threshold, state, args)
            dpo_data = 8192 - data_points_seen # Let's say we have 8192 datapoint budget
            print(f"[PerplexityCallback] Launching DPO job with data points {dpo_data} at checkpoint {checkpoint_path}")
            os.system(f"sbatch /home/shougan/projects/aip-fredashi/shougan/balance-budget/tuning/slurm/run_dpo.sh {self.model_name} tuluif {dpo_data} {checkpoint_path} {data_points_seen} ifeval dpo")

            # Stop training if this is the final (lowest) threshold
            if threshold == self.perplexity_thresholds[0]:
                print(f"[PerplexityCallback] Final threshold {threshold} reached! Stopping training.")
                control.should_training_stop = True
            else:
                print(f"[PerplexityCallback] Continuing training to next threshold...")
            break  # Only handle one threshold per evaluation

        return control
