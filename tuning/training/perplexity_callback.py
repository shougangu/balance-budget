import os
import datetime

import torch
import torch.nn.functional as F
import numpy as np
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
import wandb

from tuning.config import MODELS_METADATA_DIR
from tuning.training.callback_utils import save_sweetspot_checkpoint
from tuning.training.config_training import PerplexityConfig

class PerplexityStoppingCallback(TrainerCallback):
    def __init__(
        self, 
        config: PerplexityConfig,
        test_dataset, 
        tokenizer, 
        model_name: str = None,
    ):
        # Sort thresholds in ascending order (hardest to easiest: 2.0, 2.5, 3.0)
        # Lower perplexity = harder to reach, so we process from smallest to largest
        self.perplexity_thresholds = sorted(config.perplexity_thresholds, reverse=False)
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer
        self.num_samples = config.num_samples
        self.metadata_path = None
        self.model_name = model_name
        self.prevResults = []
        self.patience = config.patience
        self.min_decrease = config.min_decrease

        print(f"[PerplexityCallback] Initialized with perplexity_thresholds={self.perplexity_thresholds}")
        print(f"[PerplexityCallback] Training will stop when hardest threshold is reached: {self.perplexity_thresholds[0]}")
        print(f"[PerplexityCallback] num_samples={self.num_samples}")
        print(f"[PerplexityCallback] patience={self.patience}")
        print(f"[PerplexityCallback] min_decrease={self.min_decrease}")
        print(f"[PerplexityCallback] Test dataset size: {len(test_dataset)}")
        print(f"[PerplexityCallback] Dataset sample 1st one: {test_dataset[0]}")

        
    
    def on_train_begin(self, args, state, control, **kwargs):
        if not self.model_name:
            self.model_name = kwargs.get("model")
        print(f"[PerplexityCallback] on_train_begin: model_name={self.model_name}")
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        self.metadata_path = os.path.join(MODELS_METADATA_DIR, f"{self.model_name}_ppl-{now}.json") 
    
    def _save_sweetspot_checkpoint(self, model, threshold, state: TrainerState, args: TrainingArguments):
        return save_sweetspot_checkpoint(
            model=model,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            threshold_label=threshold,
            state=state,
            args=args,
            metadata_path=self.metadata_path,
            extra_metadata={
                "threshold_type": "perplexity",
                "threshold_value": threshold,
            },
        )
    
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
            for token_idx in range(len(response_ids)):
                token_id_i = response_ids[token_idx].item()
                log_prob_i = log_probs[token_idx, token_id_i].item()
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
            print("[PerplexityCallback] Warning: model is None, skipping ppl check")
            return control
        
        current_perplexity = self.evaluate_perplexity(model)
        wandb.log({"eval/perplexity": current_perplexity, "train/global_step": state.global_step})
        self.prevResults.append(current_perplexity)
        
        print(f"\n[PerplexityCallback] Step {state.global_step}, Data Points {data_points_seen}: PPL = {current_perplexity:.4f}")
        
        # Check each threshold and save checkpoint if crossed (Fork Strategy)
        # Thresholds are sorted ascending (hardest to easiest: 2.0, 2.5, 3.0)
        # We iterate to find the hardest threshold that current_perplexity has reached
        if not self.patience:
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
        
        if self.patience:
            if len(self.prevResults) > self.patience:
                early_stopping = True
                for old, new in zip(self.prevResults[-self.patience-1:], self.prevResults[-self.patience:]):
                    if old - new >= self.min_decrease:
                        early_stopping = False
                if early_stopping:
                    checkpoint_path = self._save_sweetspot_checkpoint(model, f"{self.patience}@{self.min_decrease}", state, args)
                    print(f"[PerplexityCallback] No significant improvement in the last {self.patience} evaluations. Stopping training.")
                    print(f"[PerplexityCallback] Previous PPL scores: {self.prevResults[-self.patience-1:]}")
                    print(f"[PerplexityCallback] Final checkpoint saved at {checkpoint_path}")
                    control.should_training_stop = True
        return control
