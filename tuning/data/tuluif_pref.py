from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING

class TuluIFPT(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="conifer")

    def _get_messages(self, examples):
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        prompt = examples["prompt"]

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
                {"role": "user", "content": prompt},
            ],
            "chosen": [
                {"role": "assistant", "content": chosen[-1]["content"]},
            ],
            "rejected": [
                {"role": "assistant", "content": rejected[-1]["content"]},
            ],
        }

    def _filter_long(self, examples):
        prompt_text = examples["prompt"][1]["content"]  # user message
        system_text = examples["prompt"][0]["content"]  # system message
        chosen_text = examples["chosen"][0]["content"]  # assistant message
        rejected_text = examples["rejected"][0]["content"]  # assistant message

        keep_chosen = len(prompt_text.split(" ")) + len(chosen_text.split(" ")) + len(system_text.split(" ")) < 1024
        keep_rejected = len(prompt_text.split(" ")) + len(rejected_text.split(" ")) + len(system_text.split(" ")) < 1024

        return keep_chosen and keep_rejected

    def format_dataset(self):

        formatted_dataset = self._dataset["train"].map(self._get_messages)
        formatted_dataset = formatted_dataset.filter(self._filter_long)
        print(f"Tuluif sft dataset - {formatted_dataset}")
        print(f'Example Tuluif sft row - {formatted_dataset[0]}')
        print("***")
        self._dataset = formatted_dataset.train_test_split(test_size=200, shuffle=False)

        dataset = self._dataset["test"]
        longest = 0
        longest_prompt = ""
        longest_response = ""
        for row in dataset:
            prompt_text = row["prompt"][1]["content"]
            system_text = row["prompt"][0]["content"]
            chosen_text = row["chosen"][0]["content"]
            rejected_text = row["rejected"][0]["content"]

            total_len_c = len(prompt_text.split(" ")) + len(chosen_text.split(" ")) + len(system_text.split(" "))
            total_len_r = len(prompt_text.split(" ")) + len(rejected_text.split(" ")) + len(system_text.split(" "))

            if total_len_c > longest:
                longest = total_len_c
                longest_prompt = prompt_text
                longest_response = chosen_text
            if total_len_r > longest:
                longest = total_len_r
                longest_prompt = prompt_text
                longest_response = rejected_text

        print(f"Longest row: {longest}")
        print(f"Prompt: {longest_prompt}")
        print(f"Response: {longest_response}")

if __name__ == "__main__":

    tuluif = TuluIFPT()

    tuluif.load_from_huggingface("allenai/tulu-3-pref-personas-instruction-following")
    tuluif.format_dataset()
    tuluif.clear_old_datasets(prefix="pt-tuluif")
    tuluif.save_dataset_to_disk(save_name="pt-tuluif")