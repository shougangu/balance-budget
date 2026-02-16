from datasets import Dataset, DatasetDict

from tuning.utils import utils


MODEL_TO_TEMPLATE = {
    "llama3-8B": "llama",
    "qwen2-7B": "chatml",
    "mistral-7b": "mistral",
}

TEMPLATE_ASSISTANT_TOKENS = {
    "chatml": ("<|assistant|>\n", "\n<|end|>"),
    "llama": ("<|start_assistant|>", "<|eot|>"),
    "mistral": ("[/INST]", "</s>"),
}


class FakeTokenizer:
    def __init__(self):
        self.template_name = "chatml"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assistant_prefix, assistant_suffix = TEMPLATE_ASSISTANT_TOKENS[self.template_name]
        parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "assistant":
                parts.append(f"{assistant_prefix}{content}{assistant_suffix}")
            else:
                parts.append(f"<{role}>{content}</{role}>")
        if add_generation_prompt:
            parts.append(assistant_prefix)
        rendered = "".join(parts)
        if tokenize:
            return [ord(ch) for ch in rendered]
        return rendered


def _build_dataset():
    split = Dataset.from_dict(
        {
            "system_message": ["You are concise.", "You are strict."],
            "prompt": ["What is 2+2?", "Say yes."],
            "chosen": ["4", "yes"],
            "rejected": ["5", "no"],
        }
    )
    return DatasetDict({"train": split, "test": split})


def test_apply_chat_template_pt_across_model_templates(monkeypatch):
    def _fake_chat_template_func(tokenizer, chat_template="chatml"):
        tokenizer.template_name = chat_template
        return tokenizer

    monkeypatch.setattr(utils, "chat_template_func", _fake_chat_template_func)

    for model_name, chat_template in MODEL_TO_TEMPLATE.items():
        tokenizer = FakeTokenizer()
        tokenizer = utils.chat_template_func(tokenizer, chat_template=chat_template)
        output = utils.apply_chat_template_pt(tokenizer, _build_dataset())
        assistant_prefix, assistant_suffix = TEMPLATE_ASSISTANT_TOKENS[chat_template]
        row0 = output["train"][0]

        assert row0["prompt"].endswith(assistant_prefix), f"{model_name}: prompt missing generation prefix"
        assert row0["chosen"] == f"4{assistant_suffix}", f"{model_name}: chosen completion mismatch"
        assert row0["rejected"] == f"5{assistant_suffix}", f"{model_name}: rejected completion mismatch"
        assert not row0["chosen"].startswith(assistant_prefix), f"{model_name}: assistant prefix leaked into completion"


def test_apply_chat_template_pt_raises_clear_error_on_prefix_mismatch(monkeypatch):
    class MismatchTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            text = super().apply_chat_template(messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt)
            if not tokenize and not add_generation_prompt and messages and messages[-1]["role"] == "assistant":
                return f"WRAP:{text}"
            return text

    def _fake_chat_template_func(tokenizer, chat_template="chatml"):
        tokenizer.template_name = chat_template
        return tokenizer

    monkeypatch.setattr(utils, "chat_template_func", _fake_chat_template_func)

    dataset = _build_dataset()
    tokenizer = MismatchTokenizer()
    tokenizer = utils.chat_template_func(tokenizer, chat_template="chatml")

    try:
        utils.apply_chat_template_pt(tokenizer, dataset)
        assert False, "Expected ValueError for prefix mismatch"
    except ValueError as exc:
        message = str(exc)
        assert "Chat template prefix mismatch" in message
