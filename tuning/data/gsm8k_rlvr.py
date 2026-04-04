# ABOUTME: Prompt-only GSM8K dataset for GRPO/RLVR training.
# ABOUTME: One row per unique question with reference_answer for reward function.

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_GSM8K, GSM8K_STRING


class GSM8KRLVR(HFDataset):
    def __init__(self):
        super().__init__("gsm8k")

    def _get_rows(self, dataset):
        prompts = dataset["prompt"]
        reference_answers = dataset["reference_answer"]

        seen = set()
        rows = []
        for prompt_text, ref in zip(prompts, reference_answers):
            formatted = GSM8K_STRING.format(question=prompt_text)
            if formatted in seen:
                continue
            seen.add(formatted)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE_GSM8K},
                    {"role": "user", "content": formatted},
                ],
                "reference_answer": ref,
            })
        return rows

    def format_dataset(self):
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(test_size=100, shuffle=False)
        print(f"RLVR GSM8K Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    gsm8k = GSM8KRLVR()
    gsm8k.load_from_huggingface("mohit-raghavendra/gsm8k-gpt4ogenerations", split="train")
    gsm8k.format_dataset()
    gsm8k.clear_old_datasets(prefix="rlvr-gsm8k")
    gsm8k.save_dataset_to_disk(save_name="rlvr-gsm8k")
