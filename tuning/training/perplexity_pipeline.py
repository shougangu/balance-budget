from tuning.training.config_training import (
    ModelLoadConfig, LoraConfig, SFTRunConfig, PTRunConfig,
    DPOTrainingConfig, TrainingArgumentsConfig, PassAtKConfig,
    PerplexityConfig, DatasetConfig,
)
from tuning.config import HF_MODEL_MAP
from tuning.training.sft_training import train_model_sft
from tuning.training.dpo_training import train_model_dpo

MODEL = "llama3-1B"
total_train_size = 4096

perplexity_config = PerplexityConfig(
    perplexity_thresholds=[4.0, 3.5, 2.0],
)
