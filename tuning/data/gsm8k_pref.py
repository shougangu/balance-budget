from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_GSM8K
from tuning.evaluation.gsm8k_scoring import is_correct
from datasets import DatasetDict

GSM8K_STRING = """Question: {question}\nAnswer:"""


def filter_correct_chosen(dataset: DatasetDict) -> DatasetDict:
    """Filter DPO dataset to rows where chosen answer is verified correct."""
    train = dataset["train"]
    keep_indices = [
        i for i in range(len(train))
        if is_correct(train[i]["chosen"], train[i]["reference_solution"])
    ]
    filtered_train = train.select(keep_indices) if keep_indices else train.select([])
    return DatasetDict({"train": filtered_train, "test": dataset["test"]})



class GSM8KPT(HFDataset):
    def __init__(self):
        super().__init__("gsm8k")

    def _get_rows(self, dataset):
        prompts = dataset["prompt"]
        corrects = dataset["correct"]
        incorrects = dataset["incorrect"]
        reference_answers = dataset["reference_answer"]
        system_message = dataset["system_message"]

        pairwise_rows = []
        for prompt, correct_set, incorrect_set, ref in zip(prompts, corrects, incorrects, reference_answers):
            for correct in correct_set:
                for incorrect in incorrect_set:
                    pairwise_rows.append({
                        "prompt": [
                            {"role": "system", "content": SYSTEM_MESSAGE_GSM8K},
                            {"role": "user", "content": prompt},
                        ],
                        "chosen": [
                            {"role": "assistant", "content": correct},
                        ],
                        "rejected": [
                            {"role": "assistant", "content": incorrect},
                        ],
                        "reference": ref,
                    })
        
        return pairwise_rows
    
    def format_dataset(self):

        formatted_dataset = self._dataset.train_test_split(test_size=200, shuffle=False)
        print(f"Dataset - {formatted_dataset}")
        print(f'Example row - {formatted_dataset["train"][57]}')
        print(f'Example row - {formatted_dataset["train"][5]}')
        print(f'Example row - {formatted_dataset["train"][27]}')
        self._dataset = formatted_dataset        

if __name__ == "__main__":
    gsm8k = GSM8KPT()
    gsm8k.load_from_huggingface("mohit-raghavendra/gsm8k-gpt4opreferences", split="train")

    gsm8k.format_dataset()
    filtered = filter_correct_chosen(gsm8k.get_dataset())
    print(f"After filtering (correct chosen only): {filtered}")
    gsm8k.clear_old_datasets(prefix="pt-gsm8k")
    gsm8k.save_dataset_to_disk(dataset=filtered, save_name="pt-gsm8k")

    