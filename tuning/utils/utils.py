
def chat_template_func(tokenizer, chat_template="chatml"):
    from unsloth.chat_templates import get_chat_template

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = chat_template, # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        map_eos_token = False, # Maps <|im_end|> to </s> instead
    )

    return tokenizer


def apply_chat_template(tokenizer, dataset):
    def _format(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(_format, batched=True)
    return dataset


RESPONSE_DELIMITERS = {
    "chatml": {
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
    },
    "llama-3.1": {
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    },
}


def get_response_delimiters(chat_template: str) -> dict:
    if chat_template not in RESPONSE_DELIMITERS:
        raise ValueError(
            f"No response delimiters defined for chat template '{chat_template}'. "
            f"Supported: {list(RESPONSE_DELIMITERS.keys())}"
        )
    return RESPONSE_DELIMITERS[chat_template]
