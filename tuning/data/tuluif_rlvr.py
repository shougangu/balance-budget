# ABOUTME: Prompt-only IFEval dataset for GRPO/RLVR training.
# ABOUTME: One row per instruction-following prompt (no reference answer needed).

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING


class TuluIFRLVR(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="tulu")

    def _get_rows(self, dataset):
        prompts = dataset["prompt"]

        seen = set()
        rows = []
        for prompt_text in prompts:
            if prompt_text in seen:
                continue
            seen.add(prompt_text)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
                    {"role": "user", "content": prompt_text},
                ],
            })
        return rows

    def format_dataset(self):
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(test_size=100, shuffle=False)
        print(f"RLVR IFEval Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    tuluif = TuluIFRLVR()
    tuluif.load_from_huggingface("allenai/tulu-3-sft-personas-instruction-following", split="train")
    tuluif.format_dataset()
    tuluif.clear_old_datasets(prefix="rlvr-tuluif")
    tuluif.save_dataset_to_disk(save_name="rlvr-tuluif")
