# ABOUTME: Model loading and saving utilities for training pipelines.
# ABOUTME: Supports both Unsloth-optimized and standard HuggingFace/PEFT loading.

import json
import os
import torch

from tuning.training.callback_utils import save_trainer_state


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

        # In distributed mode (LOCAL_RANK set by torchrun), device_map="auto"
        # spreads the model across all GPUs on the node, conflicting with DDP.
        # Let Accelerate handle placement instead.
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        device_map = None if local_rank >= 0 else "auto"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map=device_map,
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


def upcast_lm_head_to_fp32(model):
    """Cast the model's lm_head matmul to fp32 for GRPO numerical stability.

    Mirrors MiniMax M1 / ScaleRL "FP32 logits" — keeps log_softmax in a precision
    where trainer and vLLM logprobs agree well enough for the importance ratio.

    If lm_head shares storage with the input embeddings (tie_word_embeddings=True),
    the weight is cloned first so embed_tokens stays in its original dtype.
    A forward pre-hook upcasts hidden_states so the matmul itself runs in fp32.
    """
    import torch.nn as nn

    lm_head = model.get_output_embeddings()
    embed = model.get_input_embeddings()
    tied = lm_head.weight.data_ptr() == embed.weight.data_ptr()

    src = lm_head.weight.data
    new_weight = src.detach().clone().to(torch.float32) if tied else src.to(torch.float32)
    lm_head.weight = nn.Parameter(new_weight, requires_grad=lm_head.weight.requires_grad)

    def _cast_inputs_to_fp32(_module, inputs):
        x = inputs[0]
        if x.dtype != torch.float32:
            return (x.to(torch.float32),) + inputs[1:]
        return inputs

    lm_head.register_forward_pre_hook(_cast_inputs_to_fp32)
    return model


def upcast_vllm_lm_head_to_fp32(llm):
    """Force vLLM's lm_head matmul + log_softmax to run in fp32.

    Patches the model's LogitsProcessor._get_logits to (a) upcast hidden_states
    and (b) use an fp32 lm_head weight. Tied embeddings (e.g., Qwen2.5-3B,
    Qwen3-4B) need a separate fp32 weight buffer so embed_tokens forward stays
    in its original dtype; untied models upcast lm_head.weight in place so TRL's
    sync_weights writes fp32→fp32 each step.
    """
    import types
    import torch.nn.functional as F

    vmodel = llm.llm_engine.model_executor.driver_worker.model_runner.model
    lm_head = vmodel.lm_head
    embed = vmodel.model.embed_tokens
    tied = lm_head is embed  # vLLM ties by module identity (Qwen2/3 ForCausalLM)

    if tied:
        weight_holder = {"w": lm_head.weight.data.detach().to(torch.float32).clone()}
    else:
        lm_head.weight.data = lm_head.weight.data.to(torch.float32)
        weight_holder = {"w": lm_head.weight}

    lp = vmodel.logits_processor

    def _fp32_get_logits(self, hidden_states, _lm_head, embedding_bias):
        w = weight_holder["w"]
        logits = F.linear(hidden_states.to(torch.float32), w, embedding_bias)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    lp._get_logits = types.MethodType(_fp32_get_logits, lp)
    return llm


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
    save_trainer_state(trainer.state, output_dir)
