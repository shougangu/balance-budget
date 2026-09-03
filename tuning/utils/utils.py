import json
import sys
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
{% if 'role' in messages[0] %}{% if bos_token %}{{- bos_token }}{% endif %}\
{%- for message in messages %}\
{%- if message['role'] != 'system' %}\
{{ message['content'] }}\
{%- endif %}\
{%- endfor %}\
{%- if not add_generation_prompt %}\
{{- eos_token }}\
{%- endif %}\
{% else %}{% if bos_token %}{{- bos_token }}{% endif %}\
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


GEMMA_3_CHAT_TEMPLATE = """\
{{ bos_token }}\
{%- if 'role' in messages[0] %}\
    {%- if messages[0]['role'] == 'system' %}\
        {%- set first_user_prefix = messages[0]['content'] + '\\n\\n' %}\
        {%- set loop_messages = messages[1:] %}\
    {%- else %}\
        {%- set first_user_prefix = "" %}\
        {%- set loop_messages = messages %}\
    {%- endif %}\
    {%- for message in loop_messages %}\
        {%- if message['role'] == 'assistant' %}\
            {%- set role = "model" %}\
        {%- else %}\
            {%- set role = message['role'] %}\
        {%- endif %}\
        {{- '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else "") + (message['content'] | trim) + '<end_of_turn>\\n' }}\
    {%- endfor %}\
{%- else %}\
    {%- if messages[0]['from'] == 'system' %}\
        {%- set first_user_prefix = messages[0]['value'] + '\\n\\n' %}\
        {%- set loop_messages = messages[1:] %}\
    {%- else %}\
        {%- set first_user_prefix = "" %}\
        {%- set loop_messages = messages %}\
    {%- endif %}\
    {%- for message in loop_messages %}\
        {%- if message['from'] == 'human' %}\
            {%- set role = "user" %}\
        {%- else %}\
            {%- set role = "model" %}\
        {%- endif %}\
        {{- '<start_of_turn>' + role + '\\n' + (first_user_prefix if loop.first else "") + (message['value'] | trim) + '<end_of_turn>\\n' }}\
    {%- endfor %}\
{%- endif %}\
{%- if add_generation_prompt %}\
    {{- '<start_of_turn>model\\n' }}\
{%- endif %}\
"""


def chat_template_func(tokenizer):
    chat_template = tuning.config.DEFAULT_CHAT_TEMPLATE

    # For simple mode, use the base template for unsloth setup (special tokens,
    # ShareGPT mapping), then override with the simple Jinja2 string.
    setup_template = chat_template
    if chat_template == "simple":
        setup_template = tuning.config._BASE_CHAT_TEMPLATE

    # unsloth's setup belongs to the unsloth training path only: on plain-HF runs
    # its import is deliberately skipped, and its chatml setup stamps literal
    # sentinel tokens (<EOS_TOKEN>) onto tokenizers it doesn't recognize.
    if "unsloth" in sys.modules:
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(
            tokenizer,
            chat_template = setup_template,
            mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
            map_eos_token = False,
        )

    if chat_template == "llama-3.1":
        tokenizer.chat_template = LLAMA_31_SIMPLE_TEMPLATE
    elif chat_template == "gemma-3":
        tokenizer.chat_template = GEMMA_3_CHAT_TEMPLATE
    elif chat_template == "simple":
        tokenizer.chat_template = SIMPLE_TEMPLATE

    return tokenizer


def apply_chat_template(tokenizer, dataset, mask_prompt=False, num_proc=None):
    """Render conversations to text, optionally recording where each prompt ends.

    ``mask_prompt`` adds a ``prompt_length`` column holding the character offset at
    which the assistant response begins, which ``tokenize_sft_dataset`` turns into a
    per-token ``completion_mask``. The offset is taken from the generation-prompt
    render, so it is the exact prefix the model is shown at inference time.
    """

    def _format(examples):
        if not mask_prompt:
            texts = [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in examples["messages"]
            ]
            return {"text": texts}

        texts, prompt_lengths = [], []
        for convo in examples["messages"]:
            responses = sum(message.get("role") == "assistant" for message in convo)
            if responses != 1:
                raise ValueError(
                    "Prompt masking requires a single assistant turn per conversation; "
                    f"found {responses}."
                )
            text = tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            prompt = tokenizer.apply_chat_template(
                convo[:-1], tokenize=False, add_generation_prompt=True
            )
            if not text.startswith(prompt):
                raise ValueError(
                    f"Rendered prompt is not a prefix of the rendered conversation: {prompt!r}"
                )
            texts.append(text)
            prompt_lengths.append(len(prompt))
        return {"text": texts, "prompt_length": prompt_lengths}

    map_kwargs = {"batched": True}
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    dataset = dataset.map(_format, **map_kwargs)
    # Remove "messages" column so TRL SFTTrainer doesn't redundantly
    # re-process the dataset (spawning num_proc=os.cpu_count() workers
    # which causes OOM on SLURM nodes with many cores but limited --mem).
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    return dataset


def tokenize_sft_dataset(tokenizer, dataset, max_length, num_proc=4, mask_prompt=False):
    """Tokenize rendered SFT text without adding special tokens a second time.

    Chat templates already render BOS/EOS into the text. The resulting ``input_ids``
    also tell TRL that the dataset is processed, so it skips its default text-tokenization
    path, which would prepend another BOS for Llama tokenizers.

    ``mask_prompt`` adds a ``completion_mask`` column marking which tokens carry loss.
    """

    def _tokenize(examples):
        if not mask_prompt:
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=False,
            )

        # The prompt is tokenized on its own, exactly as it is at inference, so no
        # token can straddle the prompt/response seam (the Llama BPE fuses e.g.
        # "Answer:" + "<think>" into one ":<" token the model would never be prompted with).
        prompts = [text[:n] for text, n in zip(examples["text"], examples["prompt_length"])]
        completions = [text[n:] for text, n in zip(examples["text"], examples["prompt_length"])]
        prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completions, add_special_tokens=False)["input_ids"]
        input_ids, attention_mask, completion_mask = [], [], []
        for p_ids, c_ids in zip(prompt_ids, completion_ids):
            ids = (p_ids + c_ids)[:max_length]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            completion_mask.append(([0] * len(p_ids) + [1] * len(c_ids))[:max_length])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }

    map_kwargs = {
        "batched": True,
        "desc": 'Tokenizing SFT dataset["text"]',
    }
    if num_proc is not None:
        map_kwargs["num_proc"] = num_proc
    return dataset.map(_tokenize, **map_kwargs)


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
    "simple": ["<|end_of_text|>", "</s>", "<|im_end|>", "<|eot_id|>", "<|endoftext|>"],
}


def get_stop_tokens() -> list[str]:
    chat_template = tuning.config.DEFAULT_CHAT_TEMPLATE
    if chat_template not in STOP_TOKENS:
        raise ValueError(
            f"No stop tokens defined for chat template '{chat_template}'. "
            f"Supported: {list(STOP_TOKENS.keys())}"
        )
    return STOP_TOKENS[chat_template]


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


def on_disk_template_is_simple(checkpoint_path: str) -> bool | None:
    """Whether the checkpoint's saved chat_template is SIMPLE_TEMPLATE; None if absent."""
    on_disk = _read_on_disk_chat_template(checkpoint_path)
    if on_disk is None:
        return None
    return on_disk.strip() == SIMPLE_TEMPLATE.strip()


def warn_if_template_mismatch(checkpoint_path: str, simple_requested: bool) -> None:
    """Warn when --simple-template disagrees with the SFT checkpoint's saved template.

    The CLI flag only mutates the in-memory tokenizer; vLLM (used for GRPO rollouts)
    loads its own tokenizer from the checkpoint dir, so a mismatch silently trains
    the model on prompts formatted with the on-disk template instead of the requested
    one. Re-run SFT with the matching flag, or align the flag with the checkpoint.
    """
    on_disk_is_simple = on_disk_template_is_simple(checkpoint_path)
    if on_disk_is_simple is None:
        warnings.warn(
            f"Could not verify chat-template consistency: no chat_template found at "
            f"{checkpoint_path}. Expected --simple-template={simple_requested}.",
            stacklevel=2,
        )
        return

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
