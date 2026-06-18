from pydantic import BaseModel, model_validator
import tuning.config
from tuning.config import MODELS_DIR
from typing import Optional, Union
import os

BaseModel.model_config['protected_namespaces'] = ()

EFFECTIVE_BATCH_SIZE = 16  # Increased for H100/L40 GPUs
DEFAULT_VLLM_MAX_MODEL_LEN = 6144

def sft_batch_size(dataset_size: int):
    return 2  # H100/L40 can handle larger per-device batch sizes

def dpo_batch_size(dataset_size: int):
    return 2  # H100/L40 can handle larger per-device batch sizes

def effective_batch_size(dataset_size: int):
    return 16  # Increased for H100/L40 GPUs

class ModelLoadConfig(BaseModel):
    max_seq_length: int = 1024 
    dtype: Optional[str] = None 
    load_in_4bit: bool = False 

class LoraConfig(BaseModel):
    r: int = 32
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",]
    lora_alpha: int = 32
    lora_dropout: int = 0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: Optional[int] = None
    use_rslora: bool = False
    loftq_config: dict = {}

    @model_validator(mode="after")
    def _resolve_defaults(self):
        if self.random_state is None:
            self.random_state = tuning.config.DEFAULT_SEED
        return self

class TrainingArgumentsConfig(BaseModel):
    # sft training parameters
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = EFFECTIVE_BATCH_SIZE // per_device_train_batch_size # one opt step uses effective_batch_size data
    per_device_eval_batch_size: int = 2
    eval_strategy: str = "steps"
    eval_steps: float = 64
    logging_steps: int = 1
    do_eval: bool = True
    warmup_ratio: int = 0.0
    num_train_epochs: int = 1
    learning_rate: float = 5e-5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    report_to: list[str] = ["wandb"]
    save_strategy: str = "steps"
    save_steps: int = 625
    save_total_limit: int = 1
    load_best_model_at_end: bool = False
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 2
    eval_accumulation_steps: int = 1
    restore_callback_states_from_checkpoint: bool = True
    # prediction_loss_only: bool = True
    # eval_do_concat_batches: bool = False
    resume_from_checkpoint: bool = False  # forwarded to trainer.train(); not an HF args field

    def to_hf_args(self, output_dir: str) -> dict:
        """Return kwargs for TrainingArguments/DPOConfig constructor."""
        import torch
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        d = self.model_dump()
        d.pop("beta", None)  # beta is DPO-specific, not a TrainingArguments field
        d.pop("resume_from_checkpoint", None)  # consumed by trainer.train(), not its constructor
        d["output_dir"] = output_dir
        d["fp16"] = not bf16_supported
        d["bf16"] = bf16_supported
        d["seed"] = tuning.config.DEFAULT_SEED
        return d


class DPOTrainingConfig(TrainingArgumentsConfig):
    beta: float = 0.1 # set to 1, previously
    learning_rate: float = 5e-6
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    eval_steps: float = 256
    save_steps: int = 256
    per_device_eval_batch_size: int = 2
    dataset_num_proc: int = 4

    def to_hf_args(self, output_dir: str) -> dict:
        """Return kwargs for DPOConfig constructor, including beta."""
        d = super().to_hf_args(output_dir)
        d["beta"] = self.beta
        d["dataset_num_proc"] = self.dataset_num_proc
        return d


class GRPOTrainingConfig(TrainingArgumentsConfig):
    num_generations: int = 8
    num_generations_eval: int = 8
    num_iterations: int = 1
    # max_prompt_length: int = 512
    beta: float = 0.0
    temperature: float = 1.0
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    loss_type: str = "dapo"
    scale_rewards: Union[str, bool] = "group"
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.65 # 0.7 is perfect for Q2 and L1
    vllm_enable_sleep_mode: bool = True
    vllm_server_base_url: Optional[str] = None
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 300.0
    vllm_group_port: int = 51216
    vllm_max_model_length: int = DEFAULT_VLLM_MAX_MODEL_LEN
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    max_completion_length: int = 2048
    eval_steps: float = 64
    gradient_accumulation_steps: int = 32
    log_completions: bool = True
    num_completions_to_print: int = 4 # 4 printed on logs only, all on W&B
    save_strategy: str = "steps"
    save_steps: int = 64
    vllm_importance_sampling_correction: bool = True
    upcast_lm_head_fp32: bool = False  # MiniMax/ScaleRL stability: fp32 lm_head on trainer + vLLM
    use_liger_kernel: bool = False  # Fused Triton lm_head+GRPO loss; avoids materializing [B,T,V] logits
    zero_variance_filter: bool = True  # Drop prompt groups with zero reward variance from policy loss
    zero_variance_filter_epsilon: float = 0.0

    def to_hf_args(self, output_dir: str) -> dict:
        """Return kwargs for GRPOConfig constructor."""
        d = super().to_hf_args(output_dir)
        d.pop("upcast_lm_head_fp32", None)  # consumed by grpo_training.py, not GRPOConfig
        d.pop("zero_variance_filter", None)  # consumed by grpo_training.py, not GRPOConfig
        d.pop("zero_variance_filter_epsilon", None)  # consumed by grpo_training.py, not GRPOConfig
        d["num_generations"] = self.num_generations
        d["num_generations_eval"] = self.num_generations_eval
        d["num_iterations"] = self.num_iterations
        d["max_completion_length"] = self.max_completion_length
        # d["max_prompt_length"] = self.max_prompt_length
        d["beta"] = self.beta
        d["temperature"] = self.temperature
        d["epsilon"] = self.epsilon
        d["epsilon_high"] = self.epsilon_high
        d["loss_type"] = self.loss_type
        d["scale_rewards"] = self.scale_rewards
        d["use_vllm"] = self.use_vllm
        d["vllm_mode"] = self.vllm_mode
        d["vllm_gpu_memory_utilization"] = self.vllm_gpu_memory_utilization
        d["vllm_enable_sleep_mode"] = self.vllm_enable_sleep_mode
        d["vllm_server_base_url"] = self.vllm_server_base_url
        d["vllm_server_host"] = self.vllm_server_host
        d["vllm_server_port"] = self.vllm_server_port
        d["vllm_server_timeout"] = self.vllm_server_timeout
        d["vllm_group_port"] = self.vllm_group_port
        d["vllm_max_model_length"] = self.vllm_max_model_length
        d["log_completions"] = self.log_completions
        d["num_completions_to_print"] = self.num_completions_to_print
        d["vllm_importance_sampling_correction"] = self.vllm_importance_sampling_correction
        d.pop("eval_accumulation_steps", None)
        return d


class JudgeConfig(BaseModel):
    """Configuration for asynchronous LLM-as-a-judge quality eval."""
    enabled: bool = False
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    samples_per_prompt: int = 1  # 1 = first response; <=0 = all responses
    concurrency: int = 16
    timeout: float = 60.0
    max_retries: int = 3
    temperature: float = 0.0
    max_tokens: int = 64
    conditioned_metrics: bool = True


class PassAtKConfig(BaseModel):
    """Configuration for generation-based evaluation callback."""
    target_pass_at_k: list[float] = [0.8]  # Target pass@k score to stop training (0.0 to 1.0)
    early_tuples: list[tuple[int, float]] | None = None  # Each tuple: (patience, min_increase)
    temperature: float = 0.5  # Sampling temperature for generation
    max_tokens: int = 4096  # Maximum tokens to generate per response
    vllm_max_model_len: int = DEFAULT_VLLM_MAX_MODEL_LEN
    enabled: bool = True  # Whether to enable the callback
    use_persistent_vllm: bool = False  # Keep vLLM engine alive between evals (saves cold-start time)
    vllm_gpu_memory_utilization: float = 0.4  # GPU memory fraction for vLLM (conservative for coexistence with training)
    num_inference_gpus: int = 1  # Number of GPUs for data-parallel vLLM inference (>1 forces ephemeral mode)
    max_checkpoint_gap: int | None = None  # Save a fallback checkpoint if no checkpoint for this many data points
    target_data_points: list[int] | None = None  # Save a checkpoint when cumulative training data points cross any of these absolute marks
    initial_global_step: int = 0  # Step offset for W&B logging continuity across chained runs
    judge: Optional[JudgeConfig] = None  # Optional asynchronous LLM-as-a-judge quality eval

    def __str__(self):
        lines = [f"[{self.__class__.__name__}]"]
        for name, value in self:
            lines.append(f"  {name}={value}")
        return "\n".join(lines)


class PerplexityConfig(BaseModel):
    """Configuration for perplexity evaluation callback."""
    perplexity_thresholds: list[float] = [1.0]
    num_samples: int = 541
    early_tuples: list[tuple[int, float]] | None = None  # Each tuple: (patience, min_decrease)
    enabled: bool = True
    initial_global_step: int = 0  # Step offset for W&B logging continuity across chained runs


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
    task_name: str = "math"
    run_type: str = "sft"
    do_training: bool = False
    do_inference: bool = False
    do_evaluation: bool = False
    wandb_run_id: str = ""

    @property
    def run_name(self):
        if self.dataset_config and self.dataset_config.dynamic_path:
            return self.dataset_config.dataset_full_name
        if not self.dataset_config or not self.dataset_config.train_size:
            return self.model_name
        return f"{self.model_name}_{self.dataset_config.dataset_full_name}"

    @property
    def output_dir(self):
        base = f"{MODELS_DIR}/{self.run_name}"
        if self.wandb_run_id:
            return f"{base}_{self.wandb_run_id}"
        return base

    def __str__(self):
        return self.run_name


class PTRunConfig(BaseModel):
    model_name_hf: str = "unsloth/Meta-Llama-3.1-8B"
    model_name: str = "llama3-8B"
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
    simple_template: bool = False
    wandb_run_id: str = ""

    @property
    def run_name(self):
        run_name = self.model_name
        if self.sft_run_config:
            run_name = self.sft_run_config.run_name
        if self.dataset_config:
            run_name = f"{run_name}_{self.dataset_config.dataset_full_name}"
        if self.pft_method in ("kto", "grpo"):
            run_name = f"{run_name}_{self.pft_method}"
        if self.add_beta_run_name:
            run_name = f"{run_name}_beta-{self.beta}"

            run_name = run_name.replace(".", "-")
        return run_name

    @property
    def output_dir(self):
        base = f"{MODELS_DIR}/{self.run_name}"
        if self.wandb_run_id:
            return f"{base}_{self.wandb_run_id}"
        return base
    
    def __str__(self):
        return self.run_name
