# ABOUTME: SFT dataset from nvidia/OpenMathReasoning (cot split) across all generation models.
# ABOUTME: Keeps solutions within 1024-8000 token range.

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING
from tuning.data.heldout_math_eval import build_heldout_math_eval

TOKENIZER_NAME = "unsloth/Meta-Llama-3.1-8B"
MIN_RESPONSE_TOKENS = 1024
MAX_RESPONSE_TOKENS = 8000


class OpenMathReasoningGeneralSFT(HFDataset):
    def __init__(self, tokenizer=None, min_tokens=MIN_RESPONSE_TOKENS, max_tokens=MAX_RESPONSE_TOKENS):
        super().__init__("openmath-reasoning-general")
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

        has_answer = dataset.filter(
            lambda rows: [bool(a.strip()) for a in rows["expected_answer"]],
            batched=True,
        )

        def in_token_range(rows):
            enc = tokenizer(rows["generated_solution"], add_special_tokens=False)["input_ids"]
            return [min_tok <= len(ids) < max_tok for ids in enc]

        length_filtered = has_answer.filter(in_token_range, batched=True)

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

    def format_dataset(self, eval_dataset=None):
        # The cot split holds several solutions per problem, so a row-level tail cut
        # would leave every held-out problem in training. Evaluate on problems the
        # corpus was never seeded from instead.
        tokenizer = self._load_tokenizer()
        rows = self._format_rows(self._dataset, tokenizer)
        print(f"OpenMathReasoningGeneral SFT: {len(rows)} rows after filtering")
        if eval_dataset is None:
            eval_dataset = build_heldout_math_eval()
        formatted_dataset = DatasetDict(
            {"train": Dataset.from_list(rows), "test": eval_dataset}
        )
        print(f"OpenMathReasoningGeneral SFT Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    openmath_reasoning = OpenMathReasoningGeneralSFT()
    openmath_reasoning.load_from_huggingface("nvidia/OpenMathReasoning", split="cot")
    openmath_reasoning.format_dataset()
    openmath_reasoning.clear_old_datasets(prefix="sft-openmath-reasoning-general")
    openmath_reasoning.save_dataset_to_disk(save_name="sft-openmath-reasoning-general")
