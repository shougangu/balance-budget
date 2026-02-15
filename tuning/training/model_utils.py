import json
from unsloth import FastLanguageModel


def load_model_with_lora(model_path, model_name, model_load_config, lora_config):
    """Load a pretrained model and apply LoRA configuration.

    Handles model-specific target_modules (e.g., qwen2-7B needs embed_tokens/lm_head).
    Does NOT mutate lora_config.target_modules.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=model_load_config.max_seq_length,
        dtype=model_load_config.dtype,
        load_in_4bit=model_load_config.load_in_4bit,
    )

    target_modules = list(lora_config.target_modules)
    if model_name == "qwen2-7B":
        target_modules = target_modules + ["embed_tokens", "lm_head"]

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        target_modules=target_modules,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
        use_rslora=lora_config.use_rslora,
        loftq_config=lora_config.loftq_config,
    )

    return model, tokenizer


def save_trained_model(model, tokenizer, trainer, output_dir):
    """Save merged model and training config to output_dir."""
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(trainer.args.to_dict(), f, indent=4)
