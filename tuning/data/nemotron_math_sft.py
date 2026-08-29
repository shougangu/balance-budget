# ABOUTME: SFT dataset from nvidia/Nemotron-Math-v2 (gpt-oss-120b traces) at one reasoning effort.
# ABOUTME: Keeps tool-free single-turn solutions, one per problem, rendered as <think> reasoning + answer.

import argparse
from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING
from tuning.data.heldout_math_eval import build_heldout_math_eval

CORPUS = "nvidia/Nemotron-Math-v2"
EFFORTS = ("low", "medium", "high")
TOKENIZER_NAME = "unsloth/Meta-Llama-3.1-8B"
# Training runs at max_seq_length 8192; the cap leaves room for the chat template.
MAX_SEQUENCE_TOKENS = 8000
FILTER_NUM_PROC = 8


def render_solution(reasoning: str, answer: str) -> str:
    """Join the teacher's reasoning channel and final answer into one assistant turn."""
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer.strip()}"


def is_tool_free_single_turn(tools, messages) -> bool:
    """True for a plain user -> assistant exchange with no Python tool involvement."""
    if tools:
        return False
    if [m["role"] for m in messages] != ["user", "assistant"]:
        return False
    assistant = messages[1]
    if assistant.get("tool_calls"):
        return False
    return bool((assistant.get("reasoning_content") or "").strip()) and \
        bool((assistant.get("content") or "").strip())


class NemotronMathSFT(HFDataset):
    def __init__(self, tokenizer=None, max_tokens=MAX_SEQUENCE_TOKENS,
                 max_solutions_per_problem=1):
        super().__init__("nemotron-math")
        self._tokenizer = tokenizer
        self._max_tokens = max_tokens
        self._max_solutions_per_problem = max_solutions_per_problem

    def _load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        return self._tokenizer

    def _format_rows(self, dataset, tokenizer):
        tool_free = dataset.filter(
            lambda rows: [is_tool_free_single_turn(t, m)
                          for t, m in zip(rows["tools"], rows["messages"])],
            batched=True,
            num_proc=min(FILTER_NUM_PROC, max(1, dataset.num_rows // 1000)),
        )

        # The corpus holds several solutions per problem; keep the first few tool-free
        # ones so a problem contributes more than one route to its answer.
        seen = Counter()
        keep = []
        for i, problem in enumerate(tool_free["problem"]):
            if seen[problem] < self._max_solutions_per_problem:
                seen[problem] += 1
                keep.append(i)
        one_per_problem = tool_free.select(keep)

        def transform(row):
            prompt = COMPMATH_STRING.format(problem=row["problem"])
            assistant = row["messages"][1]
            solution = render_solution(assistant["reasoning_content"], assistant["content"])
            return {
                "prompt": prompt,
                "messages": [
                    {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": solution},
                ],
            }

        rendered = one_per_problem.map(
            transform, remove_columns=one_per_problem.column_names,
        )

        max_tok = self._max_tokens

        def within_cap(rows):
            texts = [SYSTEM_MESSAGE_OPENMATH + p + m[2]["content"]
                     for p, m in zip(rows["prompt"], rows["messages"])]
            enc = tokenizer(texts, add_special_tokens=False)["input_ids"]
            return [len(ids) <= max_tok for ids in enc]

        return rendered.filter(within_cap, batched=True)

    def format_dataset(self, eval_dataset=None):
        # Problems repeat across rows, so a row-level tail cut would leak every
        # held-out problem into training. Evaluate on problems the corpus never seeded.
        tokenizer = self._load_tokenizer()
        train = self._format_rows(self._dataset, tokenizer)
        print(f"NemotronMath SFT: {train.num_rows} rows after filtering")
        if eval_dataset is None:
            eval_dataset = build_heldout_math_eval()
        formatted_dataset = DatasetDict({"train": train, "test": eval_dataset})
        print(f"NemotronMath SFT Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


def build_nemotron_math_sft(effort, save_name, max_tokens=MAX_SEQUENCE_TOKENS,
                            clear_existing=False, max_solutions_per_problem=1):
    """Build and save the SFT dataset for one reasoning-effort split of the corpus."""
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
    loader = NemotronMathSFT(max_tokens=max_tokens,
                             max_solutions_per_problem=max_solutions_per_problem)
    loader._dataset = load_dataset(CORPUS, data_files=f"data/{effort}.parquet", split="train")
    loader._raw_dataset = loader._dataset
    loader.hf_path = CORPUS
    if clear_existing:
        loader.clear_old_datasets(prefix=save_name)
    loader.format_dataset()
    loader.save_dataset_to_disk(save_name=save_name)
    return loader.get_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Nemotron-Math SFT dataset.")
    parser.add_argument("--effort", default="medium", choices=EFFORTS)
    parser.add_argument("--save-name", default=None,
                        help="Defaults to sft-nemotron-math-<effort>")
    parser.add_argument("--max-tokens", type=int, default=MAX_SEQUENCE_TOKENS)
    parser.add_argument("--max-solutions-per-problem", type=int, default=1)
    parser.add_argument("--clear-existing", action="store_true")
    args = parser.parse_args()
    build_nemotron_math_sft(
        effort=args.effort,
        save_name=args.save_name or f"sft-nemotron-math-{args.effort}",
        max_tokens=args.max_tokens,
        clear_existing=args.clear_existing,
        max_solutions_per_problem=args.max_solutions_per_problem,
    )
