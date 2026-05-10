# ABOUTME: SFT dataset from nvidia/OpenMathReasoning (cot split) filtered to QwQ solutions.
# ABOUTME: Keeps solutions within 1024-8000 token range.

from datasets import Dataset
from transformers import AutoTokenizer

from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING

TOKENIZER_NAME = "unsloth/Meta-Llama-3.1-8B"
MIN_RESPONSE_TOKENS = 1024
MAX_RESPONSE_TOKENS = 8000


class OpenMathReasoningSFT(HFDataset):
    def __init__(self, tokenizer=None, min_tokens=MIN_RESPONSE_TOKENS, max_tokens=MAX_RESPONSE_TOKENS):
        super().__init__("openmath-reasoning")
        self._tokenizer = tokenizer
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        return self._tokenizer

    def _format_rows(self, dataset, tokenizer):
        min_tok = self._min_tokens
        max_tok = self._max_tokens

        qwq = dataset.filter(
            lambda rows: [m == "QwQ-32B" for m in rows["generation_model"]],
            batched=True,
        )
        has_answer = qwq.filter(
            lambda rows: [bool(a.strip()) for a in rows["expected_answer"]],
            batched=True,
        )

        def in_token_range(rows):
            enc = tokenizer(rows["generated_solution"], add_special_tokens=False)["input_ids"]
            return [min_tok <= len(ids) < max_tok for ids in enc]

        length_filtered = has_answer.filter(in_token_range, batched=True)

        # correct = length_filtered.filter(
        #     lambda row: math500_is_correct(row["generated_solution"], row["expected_answer"]),
        #     batched=False,
        # )

        rows = []
        for row in length_filtered:
            prompt = COMPMATH_STRING.format(problem=row["problem"])
            rows.append({
                "prompt": prompt,
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": row["generated_solution"]},
                ],
            })
        return rows

    def format_dataset(self):
        tokenizer = self._load_tokenizer()
        rows = self._format_rows(self._dataset, tokenizer)
        print(f"OpenMathReasoning SFT: {len(rows)} rows after filtering")
        formatted_dataset = Dataset.from_list(rows).train_test_split(
            test_size=min(200, len(rows) - 1), shuffle=False
        )
        print(f"OpenMathReasoning SFT Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    openmath_reasoning = OpenMathReasoningSFT()
    openmath_reasoning.load_from_huggingface("nvidia/OpenMathReasoning", split="cot")
    openmath_reasoning.format_dataset()
    openmath_reasoning.clear_old_datasets(prefix="sft-openmath-reasoning")
    openmath_reasoning.save_dataset_to_disk(save_name="sft-openmath-reasoning")
