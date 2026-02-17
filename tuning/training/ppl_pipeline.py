from unsloth import FastLanguageModel, is_bfloat16_supported


from tuning.training.config_training import (
    ModelLoadConfig, LoraConfig, SFTRunConfig, PTRunConfig,
    DPOTrainingConfig, TrainingArgumentsConfig, PassAtKConfig,
    PerplexityConfig, DatasetConfig,
)
from tuning.training.perplexity_callback import PerplexityStoppingCallback
from tuning.config import HF_MODEL_MAP
from tuning.training.sft_training import train_model_sft
from tuning.training.dpo_training import train_model_dpo
from tuning.utils.gpu import cleanup_gpu
import json
import subprocess
import wandb
import gc
import torch
from pathlib import Path
from tuning.training.wandb_utils import get_early_pairs, early_pair_tag

def get_early_pairs(config):
    early_tuples = getattr(config, "early_tuples", None)
    if not early_tuples:
        return []
    return [[int(p), float(t)] for p, t in early_tuples]


def early_pair_tag(early_pairs):
    if not early_pairs:
        return "early_pair:none"
    return "early_pair:" + ",".join(f"{p}@{t:g}" for p, t in early_pairs)

MODEL_TO_GPU_1 = {
    "llama3-1B": 0.75,
    "llama3-3B": 0.75,
    "llama3-8B": 0.68,
    "qwen2-3B": 0.75,
}
MODEL_TO_GPU_2 = {
    "llama3-1B": 0.7,
    "llama3-3B": 0.68,
    "llama3-8B": 0.55,
    "qwen2-3B": 0.6,
}

if __name__ == '__main__':
    MODEL = "qwen2-3B"
    gpu_utilisation_1 = MODEL_TO_GPU_1[MODEL]
    gpu_utilisation_2 = MODEL_TO_GPU_2[MODEL]
    total_train_size = 10240

    dataset_config = DatasetConfig(
        dataset = "tuluif",
        dataset_type = "sft",
        train_size = total_train_size,
    )

    run_config = SFTRunConfig(
        chat_template="chatml",
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[MODEL],
        model_name = MODEL,
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )
    
    perplexity_config = PerplexityConfig(
        perplexity_thresholds=[6.0, 5.0, 4.0, 3.75, 3.5, 3.25, 3.0],
        num_samples=541,
        early_tuples=[(1, 0.02), (2, 0.02), (3, 0.02), (4, 0.02), (5, 0.02)],
        enabled=True,
    )

    passk_config = PassAtKConfig(
        target_pass_at_k=[1.2],
        k_values=[1],
        n_samples=1,
        num_prompts=541,
        temperature=0.5,
        vllm_gpu_memory_utilization=gpu_utilisation_1,
        strict=True,
        enabled=True,
    )

    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = 4096
    training_args = TrainingArgumentsConfig()
    training_args.eval_steps = 32
    training_args.per_device_train_batch_size = 16
    training_args.gradient_accumulation_steps = 1
    sft_early_pairs = get_early_pairs(perplexity_config)
    run = wandb.init(
        name=run_config.model_name,
        project="tuning",
        job_type="sft",
        tags=["ppl", "sft", early_pair_tag(sft_early_pairs)],
        config={
            "pipeline_type": "ppl",
            "stage": "sft",
            "early_pairs": sft_early_pairs,
        },
    )

    with run:
        model, tokenizer, trainer, callbacks = train_model_sft(
            run_config = run_config,
            lora_config = lora_config,
            model_load_config = model_load_config,
            training_args = training_args,
            perplexity_config = perplexity_config,
            # passk_config = passk_config,
        )

    ppl_callback = next(c for c in callbacks if isinstance(c, PerplexityStoppingCallback))
    metadata_file = ppl_callback.metadata_path
    checkpoints = []
    with open(metadata_file, "r") as f:
        for line in f:
            checkpoints.append(json.loads(line))
    print(checkpoints)

    del model, tokenizer, trainer, callbacks
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_gpu(destroy_vllm = True)
    print(subprocess.check_output("nvidia-smi").decode())

    for checkpoint in checkpoints:
        model_name = Path(checkpoint["checkpoint_path"]).name
        data = total_train_size - checkpoint["data_points_seen"]
        model_load_config = ModelLoadConfig()
        training_args = DPOTrainingConfig()

        training_args.per_device_train_batch_size = 4
        training_args.gradient_accumulation_steps = 4
        training_args.eval_steps = 64

        dataset_config = DatasetConfig(
            dataset = "tuluif",
            dataset_type = "pt",
            train_size = data,
        )
        sft_run_config = SFTRunConfig(
            dataset_config = DatasetConfig(
                dataset = "tuluif",
                dataset_type = "sft",
                train_size = checkpoint["data_points_seen"],
                dynamic_path = model_name,
            ),
            model_name = MODEL,
            model_name_hf = HF_MODEL_MAP[MODEL],
            task_name = "ifeval",
        )
        run_config = PTRunConfig(
            chat_template = "chatml",
            dataset_config = dataset_config,
            model_name_hf = HF_MODEL_MAP[MODEL],
            model_name = MODEL,
            sft_run_config = sft_run_config,
            task_name = "ifeval",
            pft_method = "dpo",
            do_training = True,
        )
        passk_config = PassAtKConfig(
            target_pass_at_k=[1.2],
            k_values=[1],
            n_samples=1,
            num_prompts=541,
            temperature=0.5,
            vllm_gpu_memory_utilization=gpu_utilisation_2, #0.58
            strict=True,
            enabled=True,
        )

        dpo_early_pairs = get_early_pairs(perplexity_config)
        run = wandb.init(
            name=run_config.model_name,
            project="tuning",
            job_type="dpo",
            tags=["ppl", "dpo", early_pair_tag(dpo_early_pairs)],
            config={
                "pipeline_type": "ppl",
                "stage": "dpo",
                "early_pairs": dpo_early_pairs,
            },
            reinit=True,
        )
        with run:
            model, tokenizer, trainer, _ = train_model_dpo(
                run_config = run_config,
                lora_config = lora_config,
                model_load_config = model_load_config,
                training_args = training_args,
                perplexity_config = perplexity_config,
                # passk_config = passk_config,
            )

        try:
            wandb.finish()
        except Exception:
            pass

        del model, tokenizer, trainer
        cleanup_gpu()
        print(subprocess.check_output("nvidia-smi").decode())
