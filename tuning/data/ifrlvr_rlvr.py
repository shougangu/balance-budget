# ABOUTME: IF-RLVR dataset for GRPO training with verifiable instruction-following rewards.
# ABOUTME: Loads allenai/IF_multi_constraints_upto5 (95k prompts, up to 5 constraints each).

from dataclasses import dataclass

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
from tuning.utils.utils import GEMMA_3_CHAT_TEMPLATE, LLAMA_31_SIMPLE_TEMPLATE

MAX_PROMPT_TOKENS = 1024
TOKENIZE_BATCH_SIZE = 256


@dataclass(frozen=True)
class PromptTokenizerSpec:
    label: str
    model_name: str
    chat_template: str | None = None


# Prompts above this exact token budget are dropped from the dataset under any
# supported model chat template. Truncation would break constraint text and
# corrupt the reward signal.
PROMPT_TOKENIZER_SPECS = (
    PromptTokenizerSpec("llama3.1", "unsloth/Meta-Llama-3.1-8B", LLAMA_31_SIMPLE_TEMPLATE),
    PromptTokenizerSpec("qwen2.5", "unsloth/Qwen2.5-7B-Instruct"),
    PromptTokenizerSpec("gemma3", "unsloth/gemma-3-4b-pt", GEMMA_3_CHAT_TEMPLATE),
)


class IfrlvrRLVR(HFDataset):
    def __init__(
        self,
        max_prompt_tokens: int = MAX_PROMPT_TOKENS,
        tokenizer_specs: tuple[PromptTokenizerSpec, ...] = PROMPT_TOKENIZER_SPECS,
        tokenizers: tuple[tuple[str, object], ...] | None = None,
    ):
        super().__init__(dataset_name="ifrlvr")
        self.max_prompt_tokens = max_prompt_tokens
        self._tokenizer_specs = tokenizer_specs
        self._prompt_tokenizers = tokenizers

    def load_from_huggingface(self, hf_path: str, *args, **kwargs):
        # datasets 3.6.0 doesn't support the 'List' feature type in this dataset's schema,
        # so we keep the parquet path and stream rows during formatting.
        from huggingface_hub import hf_hub_download

        parquet_path = hf_hub_download(
            hf_path, "data/train-00000-of-00001.parquet", repo_type="dataset"
        )
        self._dataset = parquet_path
        self._raw_dataset = parquet_path
        self.hf_path = hf_path

    def _iter_parquet_rows(self, parquet_path):
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches(
            batch_size=TOKENIZE_BATCH_SIZE,
            columns=["messages", "ground_truth"],
        ):
            messages = batch.column("messages").to_pylist()
            ground_truths = batch.column("ground_truth").to_pylist()
            for row_messages, ground_truth in zip(messages, ground_truths):
                yield {"messages": row_messages, "ground_truth": ground_truth}

    def _iter_dataset_rows(self, dataset):
        if isinstance(dataset, str):
            yield from self._iter_parquet_rows(dataset)
        else:
            yield from dataset

    def _load_prompt_tokenizers(self):
        if self._prompt_tokenizers is not None:
            return self._prompt_tokenizers

        from transformers import AutoTokenizer

        tokenizers = []
        for spec in self._tokenizer_specs:
            tokenizer = AutoTokenizer.from_pretrained(spec.model_name)
            if spec.chat_template is not None:
                tokenizer.chat_template = spec.chat_template
            tokenizers.append((spec.label, tokenizer))

        self._prompt_tokenizers = tuple(tokenizers)
        return self._prompt_tokenizers

    def _filter_row_batch_by_prompt_tokens(self, rows, prompt_tokenizers, drop_counts):
        keep = [True] * len(rows)

        for label, tokenizer in prompt_tokenizers:
            texts = [
                tokenizer.apply_chat_template(
                    row["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in rows
            ]
            enc = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_tokens + 1,
            )["input_ids"]
            for idx, input_ids in enumerate(enc):
                if len(input_ids) > self.max_prompt_tokens:
                    keep[idx] = False
                    drop_counts[label] += 1

        return [row for row, should_keep in zip(rows, keep) if should_keep]

    def _print_filter_summary(self, kept_count, total_count, drop_counts):
        dropped = total_count - kept_count
        print(
            f"IF-RLVR prompt-token filter: kept {kept_count:,} / {total_count:,} "
            f"unique prompts with max_prompt_tokens={self.max_prompt_tokens}",
            flush=True,
        )
        for label, count in drop_counts.items():
            print(f"  {label}: {count:,} prompts over budget", flush=True)
        print(f"  union dropped: {dropped:,} prompts", flush=True)

    def _filter_rows_by_prompt_tokens(self, rows):
        prompt_tokenizers = self._load_prompt_tokenizers()
        if not prompt_tokenizers:
            return rows

        drop_counts = {label: 0 for label, _ in prompt_tokenizers}
        filtered_rows = self._filter_row_batch_by_prompt_tokens(
            rows, prompt_tokenizers, drop_counts
        )
        self._print_filter_summary(len(filtered_rows), len(rows), drop_counts)
        return filtered_rows

    def _build_row(self, prompt_text, ground_truth):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
                {"role": "user", "content": prompt_text},
            ],
            "ground_truth": ground_truth,
        }

    def _get_rows(self, dataset):
        prompt_tokenizers = self._load_prompt_tokenizers()
        if prompt_tokenizers:
            labels = ", ".join(label for label, _ in prompt_tokenizers)
            print(
                f"Filtering IF-RLVR prompts with max_prompt_tokens={self.max_prompt_tokens} "
                f"for: {labels}",
                flush=True,
            )

        seen = set()
        rows = []
        batch = []
        total_unique = 0
        drop_counts = {label: 0 for label, _ in prompt_tokenizers}
        next_progress = 10_000

        for row in self._iter_dataset_rows(dataset):
            prompt_text = row["messages"][0]["content"]
            if prompt_text in seen:
                continue
            seen.add(prompt_text)
            total_unique += 1
            batch.append(self._build_row(prompt_text, row["ground_truth"]))

            if len(batch) >= TOKENIZE_BATCH_SIZE:
                if prompt_tokenizers:
                    rows.extend(
                        self._filter_row_batch_by_prompt_tokens(
                            batch, prompt_tokenizers, drop_counts
                        )
                    )
                else:
                    rows.extend(batch)
                batch = []
                if total_unique >= next_progress:
                    print(
                        f"Processed {total_unique:,} unique IF-RLVR prompts; kept {len(rows):,}",
                        flush=True,
                    )
                    next_progress += 10_000

        if batch:
            if prompt_tokenizers:
                rows.extend(
                    self._filter_row_batch_by_prompt_tokens(batch, prompt_tokenizers, drop_counts)
                )
            else:
                rows.extend(batch)

        if prompt_tokenizers:
            self._print_filter_summary(len(rows), total_unique, drop_counts)
        return rows

    def format_dataset(self):
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(
            test_size=min(100, len(rows) - 1), shuffle=False
        )
        print(f"IF-RLVR Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    ifrlvr = IfrlvrRLVR()
    ifrlvr.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ifrlvr.format_dataset()
    ifrlvr.clear_old_datasets(prefix="rlvr-ifrlvr")
    ifrlvr.save_dataset_to_disk(save_name="rlvr-ifrlvr")
