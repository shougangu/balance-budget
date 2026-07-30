# ABOUTME: Model loading and saving utilities for training pipelines.
# ABOUTME: Supports both Unsloth-optimized and standard HuggingFace/PEFT loading.

import json
import os
import torch
import torch.nn.functional as F

from tuning.training.callback_utils import save_trainer_state


def is_adapter_checkpoint(model_path):
    """True when model_path is a saved LoRA adapter directory (no base weights)."""
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def _adapter_base_path(model_path):
    """Return the base model path recorded in an adapter checkpoint's config."""
    with open(os.path.join(model_path, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    base_path = adapter_config.get("base_model_name_or_path")
    if not base_path:
        raise ValueError(
            f"adapter_config.json in {model_path} has no base_model_name_or_path; "
            "cannot resolve the base model to merge the adapter into."
        )
    return base_path


def load_trainable_adapter(model, checkpoint_path, adapter_name="default"):
    """Load and activate the trainable PEFT adapter at a checkpoint root."""
    model.load_adapter(
        os.fspath(checkpoint_path), adapter_name, is_trainable=True
    )
    model.set_adapter(adapter_name)


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


def load_model_with_lora(model_path, model_name, model_load_config, lora_config, use_unsloth=True, layers_to_transform=None, full_finetune=False):
    """Load a pretrained model and apply LoRA configuration.

    Handles model-specific target_modules (e.g., qwen2-7B needs embed_tokens/lm_head).
    Does NOT mutate lora_config.target_modules.

    full_finetune=True skips the LoRA adapter entirely so every weight trains;
    it requires the plain HF branch (use_unsloth=False) and full-precision weights.
    """
    if full_finetune and use_unsloth:
        raise ValueError("full_finetune requires use_unsloth=False (plain HF/TRL path)")
    if full_finetune and model_load_config.load_in_4bit:
        raise ValueError("full_finetune cannot train 4bit-quantized weights; set load_in_4bit=False")
    # An adapter checkpoint stores only the LoRA delta: load its base, then merge the
    # adapter into the weights so the fresh adapter below starts from the same
    # full-model point a merged checkpoint would have provided.
    is_adapter = is_adapter_checkpoint(model_path)
    load_path = _adapter_base_path(model_path) if is_adapter else model_path
    if use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_path,
            max_seq_length=model_load_config.max_seq_length,
            dtype=model_load_config.dtype,
            load_in_4bit=model_load_config.load_in_4bit,
        )
        if is_adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, model_path).merge_and_unload()

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
        from peft import LoraConfig as PeftLoraConfig, PeftModel, get_peft_model

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

        # An adapter checkpoint stores only the LoRA delta: load its base, then merge
        # the adapter into the weights so the fresh adapter below starts from the same
        # full-model point a merged checkpoint would have provided.
        load_path = _adapter_base_path(model_path) if is_adapter else model_path
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            quantization_config=quantization_config,
            torch_dtype=dtype,
            device_map=device_map,
        )
        if is_adapter:
            model = PeftModel.from_pretrained(model, model_path).merge_and_unload()

        if lora_config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable() 
            # "unsloth" mode, default for Unsloth, is still truthy, so we keep standard checkpointing here

        if full_finetune:
            return model, tokenizer

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
    The forward override upcasts hidden_states and runs the matmul with autocast
    disabled: mixed-precision training (bf16=True) wraps the model forward in
    torch.autocast(bf16), which would otherwise cast the fp32 inputs of F.linear
    back down to bf16 and silently undo the upcast.
    """
    import torch.nn as nn

    lm_head = model.get_output_embeddings()
    embed = model.get_input_embeddings()
    tied = lm_head.weight.data_ptr() == embed.weight.data_ptr()

    src = lm_head.weight.data
    new_weight = src.detach().clone().to(torch.float32) if tied else src.to(torch.float32)
    lm_head.weight = nn.Parameter(new_weight, requires_grad=lm_head.weight.requires_grad)
    if getattr(lm_head, "bias", None) is not None:
        lm_head.bias = nn.Parameter(
            lm_head.bias.data.to(torch.float32), requires_grad=lm_head.bias.requires_grad
        )

    def _fp32_forward(x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.to(torch.float32), lm_head.weight, lm_head.bias)

    lm_head.forward = _fp32_forward
    return model


def disable_loss_kwargs_if_unsupported(trainer) -> bool:
    """Keep the trainer's loss invariant to gradient_accumulation_steps.

    A model class sets accepts_loss_kwargs = False to declare that its loss cannot
    be normalized by num_items_in_batch — the multimodal Gemma-3 classes do this
    because they filter logits and labels before computing the loss. Trainer only
    divides the loss by gradient_accumulation_steps when it believes the model did
    not normalize it, so a model whose declaration is ignored has both its logged
    loss and its gradients multiplied by gradient_accumulation_steps.

    Unsloth rewrites Trainer.__init__ to skip that declaration and infer support
    from **kwargs in the forward signature, which is always true for Gemma-3.
    Reading the declaration off the base model restores it for this trainer.

    Returns True when loss kwargs were disabled.
    """
    model = trainer.accelerator.unwrap_model(trainer.model)
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        model = model.base_model.model

    if getattr(model, "accepts_loss_kwargs", True):
        return False

    trainer.model_accepts_loss_kwargs = False
    return True


def install_vllm_fp32_logits_patch() -> bool:
    """Patch vLLM LogitsProcessor to compute output projection logits in fp32.

    This is installed before vLLM engine construction so the actual lm_head module
    passed by vLLM is used, including tied embedding/output projection modules.
    vLLM's LogitsProcessor.forward still owns scaling and soft-cap handling.
    """
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    if getattr(LogitsProcessor, "_balance_budget_fp32_logits_patch", False):
        return False

    original_get_logits = LogitsProcessor._get_logits

    def _fp32_get_logits(self, hidden_states, lm_head, embedding_bias):
        bias = embedding_bias.float() if embedding_bias is not None else None
        logits = F.linear(hidden_states.float(), lm_head.weight.float(), bias)
        logits = self._gather_logits(logits)
        if logits is not None:
            logits = logits[..., : self.org_vocab_size]
        return logits

    LogitsProcessor._balance_budget_original_get_logits = original_get_logits
    LogitsProcessor._get_logits = _fp32_get_logits
    LogitsProcessor._balance_budget_fp32_logits_patch = True
    return True


def install_trl_vllm_dtype_patch(dtype_name: str) -> bool:
    """Force TRL's colocated vLLM LLM constructor wrapper to receive dtype."""
    import importlib

    vllm_generation = None
    for module_name in (
        "trl.generation.vllm_generation",
        "trl.extras.vllm_generation",
        "trl.trainer.grpo_trainer",
    ):
        try:
            candidate = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(candidate, "LLM"):
            vllm_generation = candidate
            break

    if vllm_generation is None:
        raise RuntimeError("Could not find TRL's vLLM LLM constructor wrapper")

    original_llm = getattr(
        vllm_generation,
        "_balance_budget_original_LLM",
        vllm_generation.LLM,
    )
    if getattr(vllm_generation, "_balance_budget_vllm_dtype", None) == dtype_name:
        return False

    def _llm_with_dtype(*args, **kwargs):
        kwargs["dtype"] = dtype_name
        return original_llm(*args, **kwargs)

    vllm_generation._balance_budget_original_LLM = original_llm
    vllm_generation.LLM = _llm_with_dtype
    vllm_generation._balance_budget_vllm_dtype = dtype_name
    return True


def sync_colocated_vllm_chat_template(trainer, tokenizer) -> bool:
    """Give TRL's colocated vLLM backend the trainer tokenizer's chat template.

    TRL leaves VLLMGeneration.chat_template as None for normal non-tools GRPO,
    which makes vLLM read the template from model.name_or_path. Adapter-only
    checkpoints load through their base model, so pass the active tokenizer
    template explicitly.
    """
    if getattr(trainer, "vllm_mode", None) != "colocate":
        return False

    vllm_generation = getattr(trainer, "vllm_generation", None)
    if vllm_generation is None:
        return False

    if getattr(vllm_generation, "chat_template", None) is not None:
        return False

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return False

    vllm_generation.chat_template = chat_template
    return True


def save_trained_model(model, tokenizer, trainer, output_dir):
    """Save the LoRA adapter, tokenizer, and training config to output_dir."""
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_dir)
    else:
        model.save_pretrained_merged(output_dir, tokenizer, save_method="lora")
    tokenizer.save_pretrained(output_dir)
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(trainer.args.to_dict(), f, indent=4)
    save_trainer_state(trainer.state, output_dir)
