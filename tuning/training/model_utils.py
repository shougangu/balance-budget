# ABOUTME: Model loading and saving utilities for training pipelines.
# ABOUTME: Supports both Unsloth-optimized and standard HuggingFace/PEFT loading.

import json
import torch


def top_layer_indices(model_name_hf, fraction):
    """Return the layer indices for the top fraction of a model's transformer layers.

    Args:
        model_name_hf: HuggingFace model path (used to read num_hidden_layers from config).
        fraction: Float 0.0-1.0, fraction of top layers to keep.
                  E.g. 0.2 on a 28-layer model returns [22, 23, 24, 25, 26, 27].
    """
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name_hf)
    num_layers = config.num_hidden_layers
    num_trainable = num_layers - int(num_layers * (1.0 - fraction))
    start = num_layers - num_trainable
    indices = list(range(start, num_layers))
    print(f"[layers_to_transform] {num_trainable}/{num_layers} layers: {indices}")
    return indices


def load_model_with_lora(model_path, model_name, model_load_config, lora_config, use_unsloth=True, layers_to_transform=None):
    """Load a pretrained model and apply LoRA configuration.

    Handles model-specific target_modules (e.g., qwen2-7B needs embed_tokens/lm_head).
    Does NOT mutate lora_config.target_modules.
    """
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=model_load_config.max_seq_length,
            dtype=model_load_config.dtype,
            load_in_4bit=model_load_config.load_in_4bit,
        )

        target_modules = list(lora_config.target_modules)
        # if model_name == "qwen2-7B":
        #     target_modules = target_modules + ["embed_tokens", "lm_head"]

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
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig as PeftLoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        quantization_config = None
        if model_load_config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        dtype = model_load_config.dtype
        if dtype is None:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map="auto",
        )

        if lora_config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable() 
            # "unsloth" mode, default for Unsloth, is still truthy, so we keep standard checkpointing here

        target_modules = list(lora_config.target_modules)
        peft_config = PeftLoraConfig(
            r=lora_config.r,
            target_modules=target_modules,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            bias=lora_config.bias,
            use_rslora=lora_config.use_rslora,
            loftq_config=lora_config.loftq_config,
            task_type="CAUSAL_LM",
            layers_to_transform=layers_to_transform,
        )
        model = get_peft_model(model, peft_config)

    return model, tokenizer


def save_trained_model(model, tokenizer, trainer, output_dir):
    """Save merged model and training config to output_dir."""
    if hasattr(model, 'save_pretrained_merged'):
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    else:
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(trainer.args.to_dict(), f, indent=4)
