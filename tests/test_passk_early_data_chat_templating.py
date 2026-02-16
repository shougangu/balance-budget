from datasets import Dataset, DatasetDict

from tuning.data import train_dataset as td
from tuning.training.config_training import DatasetConfig, PTRunConfig, SFTRunConfig
from tuning.utils import utils


class FakeTokenizer:
    def __init__(self, assistant_prefix, assistant_suffix):
        self.assistant_prefix = assistant_prefix
        self.assistant_suffix = assistant_suffix
        self.chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for message in messages:
            if message["role"] == "assistant":
                parts.append(f"{self.assistant_prefix}{message['content']}{self.assistant_suffix}")
            else:
                parts.append(f"<{message['role']}>{message['content']}</{message['role']}>")
        if add_generation_prompt:
            parts.append(self.assistant_prefix)
        text = "".join(parts)
        if tokenize:
            return [ord(ch) for ch in text]
        return text


LLAMA33_ASSISTANT_PREFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA33_ASSISTANT_SUFFIX = "<|eot_id|>"


def test_passk_early_sft_loaded_data_is_chat_templated(monkeypatch, tmp_path):
    loaded_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "messages": [[
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "Say hello."},
                        {"role": "assistant", "content": "hello"},
                    ]]
                }
            ),
            "test": Dataset.from_dict(
                {
                    "messages": [[
                        {"role": "system", "content": "Be concise."},
                        {"role": "user", "content": "Say bye."},
                        {"role": "assistant", "content": "bye"},
                    ]]
                }
            ),
        }
    )

    tmp_data_dir = tmp_path / "passk_early_sft_data"
    save_dir = tmp_data_dir / "sft-tuluif-1"
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    loaded_dataset.save_to_disk(str(save_dir))
    monkeypatch.setattr(td, "DATASETS_DIR", str(tmp_data_dir))

    run_config = SFTRunConfig(
        model_name="llama3-8B",
        dataset_config=DatasetConfig(dataset="tuluif", dataset_type="sft", train_size=1),
    )
    dataset = td.get_train_dataset(run_config)

    tokenizer = FakeTokenizer(LLAMA33_ASSISTANT_PREFIX, LLAMA33_ASSISTANT_SUFFIX)
    templated = utils.apply_chat_template(tokenizer, dataset)

    row = templated["train"][0]
    print("\n[SFT formatted train row text]\n", row["text"])
    assert "text" in row
    assert "<system>Be concise.</system>" in row["text"]
    assert f"{LLAMA33_ASSISTANT_PREFIX}hello{LLAMA33_ASSISTANT_SUFFIX}" in row["text"]


def test_passk_early_pt_loaded_data_is_chat_templated(monkeypatch, tmp_path):
    loaded_dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "system_message": ["Be concise."],
                    "prompt": ["What is 2+2?"],
                    "chosen": ["4"],
                    "rejected": ["5"],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "system_message": ["Be concise."],
                    "prompt": ["What is 3+3?"],
                    "chosen": ["6"],
                    "rejected": ["7"],
                }
            ),
        }
    )

    tmp_data_dir = tmp_path / "passk_early_pt_data"
    save_dir = tmp_data_dir / "pt-tuluif-1"
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    loaded_dataset.save_to_disk(str(save_dir))
    monkeypatch.setattr(td, "DATASETS_DIR", str(tmp_data_dir))

    run_config = PTRunConfig(
        model_name="llama3-8B",
        dataset_config=DatasetConfig(dataset="tuluif", dataset_type="pt", train_size=1),
    )
    dataset = td.get_train_dataset(run_config)

    tokenizer = FakeTokenizer(LLAMA33_ASSISTANT_PREFIX, LLAMA33_ASSISTANT_SUFFIX)
    templated = utils.apply_chat_template_pt(tokenizer, dataset)

    row = templated["train"][0]
    print("\n[PT formatted train row prompt]\n", row["prompt"])
    print("\n[PT formatted train row chosen]\n", row["chosen"])
    print("\n[PT formatted train row rejected]\n", row["rejected"])
    assert row["prompt"].endswith(LLAMA33_ASSISTANT_PREFIX)
    assert row["chosen"] == f"4{LLAMA33_ASSISTANT_SUFFIX}"
    assert row["rejected"] == f"5{LLAMA33_ASSISTANT_SUFFIX}"
    assert not row["chosen"].startswith(LLAMA33_ASSISTANT_PREFIX)
