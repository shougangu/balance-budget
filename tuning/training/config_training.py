from enum import Enum
from pydantic import BaseModel, model_validator
from tuning.config import MODELS_DIR, resolve_chat_template
from typing import Optional
import os

BaseModel.model_config['protected_namespaces'] = ()

EFFECTIVE_BATCH_SIZE = 16  # Increased for H100/L40 GPUs

def sft_batch_size(dataset_size: int):
    return 2  # H100/L40 can handle larger per-device batch sizes

def dpo_batch_size(dataset_size: int):
    return 2  # H100/L40 can handle larger per-device batch sizes

def effective_batch_size(dataset_size: int):
    return 16  # Increased for H100/L40 GPUs

class ModelLoadConfig(BaseModel):
    max_seq_length: str = 1024 
    dtype: str = None 
    load_in_4bit: bool = False 

class LoraConfig(BaseModel):
    r: int = 32
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",]
    lora_alpha: int = 32
    lora_dropout: int = 0
    bias: str = "none"
    use_gradient_checkpointing: bool = False
    random_state: int = 42
    use_rslora: bool = False
    loftq_config: str = None

class TrainingArgumentsConfig(BaseModel):
    # sft training parameters
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = EFFECTIVE_BATCH_SIZE // per_device_train_batch_size # one opt step uses effective_batch_size data
    per_device_eval_batch_size: int = 2
    eval_strategy: str = "steps"
    eval_steps: float = 4
    logging_steps: int = 1
    do_eval: bool = True
    warmup_ratio: int = 0.1
    num_train_epochs: int = 2
    learning_rate: float = 5e-5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    report_to: list[str] = ["wandb"]
    save_strategy: str = "no"
    save_steps: int = 4 # each checkpoint step is one gradient weight update (optimizer.step()), data = grad_acc * batch_size * save_steps = effective_batch_size * save_steps
    save_total_limit: int = 1
    load_best_model_at_end: bool = False
    dataloader_drop_last: bool = False
    eval_accumulation_steps: int = 1
    # prediction_loss_only: bool = True
    # eval_do_concat_batches: bool = False

    def to_hf_args(self, output_dir: str) -> dict:
        """Return kwargs for TrainingArguments/DPOConfig constructor."""
        from unsloth import is_bfloat16_supported
        d = self.model_dump()
        d.pop("beta", None)  # beta is DPO-specific, not a TrainingArguments field
        d["output_dir"] = output_dir
        d["fp16"] = not is_bfloat16_supported()
        d["bf16"] = is_bfloat16_supported()
        d["seed"] = 42
        return d


class DPOTrainingConfig(TrainingArgumentsConfig):
    beta: float = 1
    learning_rate: float = 5e-6
    num_train_epochs: int = 2
    per_device_eval_batch_size: int = 2
    dataset_num_proc: int = 2

    def to_hf_args(self, output_dir: str) -> dict:
        """Return kwargs for DPOConfig constructor, including beta."""
        d = super().to_hf_args(output_dir)
        d["beta"] = self.beta
        d["save_strategy"] = "no"
        d["dataset_num_proc"] = self.dataset_num_proc
        return d


class PassAtKConfig(BaseModel):
    """Configuration for pass@k evaluation callback."""
    target_pass_at_k: list[float] = [0.8]  # Target pass@k score to stop training (0.0 to 1.0)
    early_tuples: list[tuple[int, float]] | None = None  # Each tuple: (patience, min_increase)
    k_values: list[int] = [1]  # The k values for pass@k evaluation. First value is used for stopping.
    n_samples: int = 1  # Number of samples to generate per prompt
    num_prompts: int = 541  # Number of prompts to evaluate (subset for speed)
    temperature: float = 0.5  # Sampling temperature for generation
    max_tokens: int = 1024  # Maximum tokens to generate per response
    strict: bool = True  # Use strict (True) or loose (False) IFEval evaluation
    enabled: bool = True  # Whether to enable the callback
    use_persistent_vllm: bool = True  # Keep vLLM engine alive between evals (saves cold-start time)
    vllm_gpu_memory_utilization: float = 0.4  # GPU memory fraction for vLLM (conservative for coexistence with training)
    num_inference_gpus: int = 1  # Number of GPUs for data-parallel vLLM inference (>1 forces ephemeral mode)


class PerplexityConfig(BaseModel):
    """Configuration for perplexity evaluation callback."""
    perplexity_thresholds: list[float] = [3.0, 2.5, 2.0]
    num_samples: int = 100
    early_tuples: list[tuple[int, float]] | None = None  # Each tuple: (patience, min_decrease)
    enabled: bool = True


class DatasetConfig(BaseModel):
    dataset: str = "gsm8k"
    dataset_type: str = "sft"
    train_size: int = 100
    dynamic_path: str = None

    @property
    def dataset_full_name(self):
        if self.dynamic_path:
            return os.path.basename(self.dynamic_path)
        if not self.train_size:
            return f"{self.dataset_type}-{self.dataset}"
        return f"{self.dataset_type}-{self.dataset}-{self.train_size}"
    
    def __str__(self):
        return self.dataset_full_name

    
class SFTRunConfig(BaseModel):
    model_name_hf: str = "unsloth/Meta-Llama-3.1-8B"
    dataset_config: Optional[DatasetConfig] = None
    model_name: str = "llama3-8B"
    chat_template: Optional[str] = None
    task_name: str = "math"
    run_type: str = "sft"
    do_training: bool = False
    do_inference: bool = False
    do_evaluation: bool = False

    @model_validator(mode="after")
    def _set_default_chat_template(self):
        if self.chat_template is None:
            self.chat_template = resolve_chat_template(self.model_name)
        # Also set the global so inference configs auto-resolve
        import tuning.config
        tuning.config.DEFAULT_CHAT_TEMPLATE = self.chat_template
        return self

    @property
    def run_name(self):
        if not self.dataset_config or not self.dataset_config.train_size:
            return self.model_name
        return f"{self.model_name}_{self.dataset_config.dataset_full_name}"
    
    @property
    def output_dir(self):
        return f"{MODELS_DIR}/{self.run_name}"
    
    def __str__(self):
        return self.run_name
    

class PTRunConfig(BaseModel):
    model_name_hf: str = "unsloth/Meta-Llama-3.1-8B"
    model_name: str = "llama3-8B"
    chat_template: Optional[str] = None
    dataset_config: DatasetConfig = None
    sft_run_config: Optional[SFTRunConfig] = None
    run_type: str = "pt"
    task_name: str = "math"
    do_training: bool = False
    do_inference: bool = False
    do_evaluation: bool = False
    pft_method: str = "dpo"
    add_beta_run_name: bool = False
    beta: float = 0.1

    @model_validator(mode="after")
    def _set_default_chat_template(self):
        if self.chat_template is None:
            self.chat_template = resolve_chat_template(self.model_name)
        # Also set the global so inference configs auto-resolve
        import tuning.config
        tuning.config.DEFAULT_CHAT_TEMPLATE = self.chat_template
        return self

    @property
    def run_name(self):
        run_name = self.model_name
        if self.sft_run_config:
            run_name = self.sft_run_config.run_name
        if self.dataset_config:
            run_name = f"{run_name}_{self.dataset_config.dataset_full_name}"
        if self.pft_method == "kto":
            run_name = f"{run_name}_{self.pft_method}"
        if self.add_beta_run_name:
            run_name = f"{run_name}_beta-{self.beta}"

            run_name = run_name.replace(".", "-")   
        return run_name
    
    @property
    def output_dir(self):
        return f"{MODELS_DIR}/{self.run_name}"
    
    def __str__(self):
        return self.run_name
