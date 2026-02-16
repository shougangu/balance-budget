from datasets import DatasetDict, Dataset


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

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts }

    dataset = dataset.map(formatting_prompts_func, batched = True,)

    return dataset

def apply_chat_template_pt(tokenizer, dataset):

    def _extract_completion(prompt_messages, assistant_content):
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            prompt_messages + [{"role": "assistant", "content": assistant_content}],
            tokenize=False,
            add_generation_prompt=False,
        )

        if not full_text.startswith(prompt_text):
            template_name = getattr(
                tokenizer,
                "chat_template",
                getattr(tokenizer, "template_name", "<unknown>"),
            )
            raise ValueError(
                f"Chat template prefix mismatch for template '{template_name}': "
                f"full_text does not start with prompt_text "
                f"(prompt_len={len(prompt_text)}, full_len={len(full_text)})."
            )

        return full_text[len(prompt_text):]

    def formatting_prompts_func(example):
        
        prompt = example["prompt"]

        if type(prompt) == str:
            message = [
                {"role": "system", "content": example["system_message"]},
                {"role": "user", "content": example["prompt"]},
            ]
        elif type(prompt) == list:
            message = prompt

        example["prompt"] = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        example["chosen"] = _extract_completion(message, example["chosen"])
        example["rejected"] = _extract_completion(message, example["rejected"])

        return example


    dataset = dataset.map(formatting_prompts_func, batched = False)
    return dataset


def get_kto_rows(dataset):
    def get_rows(dataset_split):
        rows = []
        for prompt, chosen, rejected in zip(dataset_split["prompt"], dataset_split["chosen"], dataset_split["rejected"]): 
            rows.extend([
                {
                    "prompt": prompt[100:],
                    "completion": chosen,
                    "label": True,
                },
                {
                    "prompt": prompt[100:],
                    "completion": rejected,
                    "label": False,
                }
            ])

        return rows
    
    dataset_kto = DatasetDict()
    dataset_kto["train"] = Dataset.from_list(get_rows(dataset["train"]))
    dataset_kto["test"] = Dataset.from_list(get_rows(dataset["test"]))

    print(dataset_kto["train"][0])
    return dataset_kto
        
