import json
import warnings
from pathlib import Path

import tuning.config


LLAMA_31_SIMPLE_TEMPLATE = """\
{% if 'role' in messages[0] %}{{- bos_token }}\
{%- if messages[0]['role'] == 'system' %}\
    {%- set system_message = messages[0]['content'] %}\
    {%- set messages = messages[1:] %}\
{%- else %}\
    {%- set system_message = "" %}\
{%- endif %}\
{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\
{{- system_message }}\
{{- "<|eot_id|>" }}\
{%- for message in messages %}\
    {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] + '<|eot_id|>' }}\
{%- endfor %}\
{%- if add_generation_prompt %}\
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\
{%- else %}\
    {{- "<|end_of_text|>" }}\
{%- endif %}\
{% else %}{{- bos_token }}\
{%- if messages[0]['from'] == 'system' %}\
    {%- set system_message = messages[0]['value'] %}\
    {%- set messages = messages[1:] %}\
{%- else %}\
    {%- set system_message = "" %}\
{%- endif %}\
{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\
{{- system_message }}\
{{- "<|eot_id|>" }}\
{%- for message in messages %}\
    {{- '<|start_header_id|>' + message['from'] + '<|end_header_id|>\\n\\n'+ message['value'] + '<|eot_id|>' }}\
{%- endfor %}\
{%- if add_generation_prompt %}\
    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\
{%- else %}\
    {{- "<|end_of_text|>" }}\
{%- endif %}\
{% endif %}\
"""


SIMPLE_TEMPLATE = """\
{% if 'role' in messages[0] %}{{- bos_token }}\
{%- for message in messages %}\
{%- if message['role'] != 'system' %}\
{{ message['content'] }}\
{%- endif %}\
{%- endfor %}\
{%- if not add_generation_prompt %}\
{{- eos_token }}\
{%- endif %}\
{% else %}{{- bos_token }}\
{%- for message in messages %}\
{%- if message['from'] != 'system' %}\
{{ message['value'] }}\
{%- endif %}\
{%- endfor %}\
{%- if not add_generation_prompt %}\
{{- eos_token }}\
{%- endif %}\
{% endif %}\
"""


def chat_template_func(tokenizer):
    from unsloth.chat_templates import get_chat_template

    chat_template = tuning.config.DEFAULT_CHAT_TEMPLATE

    # For simple mode, use the base template for unsloth setup (special tokens,
    # ShareGPT mapping), then override with the simple Jinja2 string.
    setup_template = chat_template
    if chat_template == "simple":
        setup_template = tuning.config._BASE_CHAT_TEMPLATE

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = setup_template,
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
        map_eos_token = False,
    )

    if chat_template == "llama-3.1":
        tokenizer.chat_template = LLAMA_31_SIMPLE_TEMPLATE
    elif chat_template == "simple":
        tokenizer.chat_template = SIMPLE_TEMPLATE

    return tokenizer


def apply_chat_template(tokenizer, dataset):
    def _format(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(_format, batched=True)
    # Remove "messages" column so TRL SFTTrainer doesn't redundantly
    # re-process the dataset (spawning num_proc=os.cpu_count() workers
    # which causes OOM on SLURM nodes with many cores but limited --mem).
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    return dataset


def apply_chat_template_pt(tokenizer, dataset):
    def _format(examples):
        prompts = []
        chosens = []
        rejecteds = []

        for system_message, prompt, chosen, rejected in zip(
            examples["system_message"],
            examples["prompt"],
            examples["chosen"],
            examples["rejected"],
        ):
            conv_prompt = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ]
            conv_chosen = conv_prompt + [{"role": "assistant", "content": chosen}]
            conv_rejected = conv_prompt + [{"role": "assistant", "content": rejected}]

            prompt_text = tokenizer.apply_chat_template(
                conv_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            chosen_full = tokenizer.apply_chat_template(
                conv_chosen,
                tokenize=False,
                add_generation_prompt=False,
            )
            rejected_full = tokenizer.apply_chat_template(
                conv_rejected,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not chosen_full.startswith(prompt_text):
                raise ValueError(
                    "Chat template prefix mismatch: chosen text does not start with prompt text. "
                    f"Prompt: {prompt_text!r}, Chosen full: {chosen_full!r}"
                )

            prompts.append(prompt_text)
            chosens.append(chosen_full[len(prompt_text):])
            rejecteds.append(rejected_full[len(prompt_text):])

        return {
            "prompt": prompts,
            "chosen": chosens,
            "rejected": rejecteds,
        }

    return dataset.map(_format, batched=True)


STOP_TOKENS = {
    "chatml": ["<|im_end|>", "<|end_of_text|>"],
    "llama-3.1": ["<|eot_id|>", "<|end_of_text|>"],
    "gemma-3": ["<end_of_turn>", "<eos>"],
    "simple": ["<|end_of_text|>", "</s>", "<|im_end|>", "<|eot_id|>"],
}


def get_stop_tokens() -> list[str]:
    chat_template = tuning.config.DEFAULT_CHAT_TEMPLATE
    if chat_template not in STOP_TOKENS:
        raise ValueError(
            f"No stop tokens defined for chat template '{chat_template}'. "
            f"Supported: {list(STOP_TOKENS.keys())}"
        )
    return STOP_TOKENS[chat_template]


RESPONSE_DELIMITERS = {
    "chatml": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "llama-3.1": {
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "gemma-3": {
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
    },
}


def get_response_delimiters() -> dict:
    chat_template = tuning.config.DEFAULT_CHAT_TEMPLATE
    if chat_template not in RESPONSE_DELIMITERS:
        raise ValueError(
            f"No response delimiters defined for chat template '{chat_template}'. "
            f"Supported: {list(RESPONSE_DELIMITERS.keys())}"
        )
    return RESPONSE_DELIMITERS[chat_template]


def _read_on_disk_chat_template(checkpoint_path: str) -> str | None:
    """Read the saved chat_template at a checkpoint dir, or None if absent."""
    cp = Path(checkpoint_path)
    jinja = cp / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text()
    cfg_path = cp / "tokenizer_config.json"
    if cfg_path.is_file():
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            return None
        return cfg.get("chat_template")
    return None


def warn_if_template_mismatch(checkpoint_path: str, simple_requested: bool) -> None:
    """Warn when --simple-template disagrees with the SFT checkpoint's saved template.

    The CLI flag only mutates the in-memory tokenizer; vLLM (used for GRPO rollouts)
    loads its own tokenizer from the checkpoint dir, so a mismatch silently trains
    the model on prompts formatted with the on-disk template instead of the requested
    one. Re-run SFT with the matching flag, or align the flag with the checkpoint.
    """
    on_disk = _read_on_disk_chat_template(checkpoint_path)
    if on_disk is None:
        warnings.warn(
            f"Could not verify chat-template consistency: no chat_template found at "
            f"{checkpoint_path}. Expected --simple-template={simple_requested}.",
            stacklevel=2,
        )
        return

    on_disk_is_simple = on_disk.strip() == SIMPLE_TEMPLATE.strip()
    if simple_requested and not on_disk_is_simple:
        warnings.warn(
            f"Chat-template mismatch: --simple-template was passed, but the SFT "
            f"checkpoint at {checkpoint_path} has a different (non-simple) chat_template "
            f"saved on disk. vLLM rollouts will use the on-disk template, not the simple "
            f"one. Re-run SFT with --simple-template, or drop the flag from this stage.",
            stacklevel=2,
        )
    elif not simple_requested and on_disk_is_simple:
        warnings.warn(
            f"Chat-template mismatch: --simple-template was NOT passed, but the SFT "
            f"checkpoint at {checkpoint_path} has the simple chat_template saved on disk. "
            f"vLLM rollouts will use the simple template, not the model's default. "
            f"Pass --simple-template, or re-run SFT without it.",
            stacklevel=2,
        )
