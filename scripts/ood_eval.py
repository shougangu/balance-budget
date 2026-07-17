# ABOUTME: Standalone OOD benchmark evaluation (Minerva Math, OlympiadBench) for the
# ABOUTME: r128 budget-grid target snapshots; one vLLM engine per model, adapters swapped per cell.

import argparse
import json
from pathlib import Path

from tuning.config import HF_MODEL_MAP, MODELS_DIR, set_chat_template
from tuning.data.test_dataset import (
    get_minervamath_test_dataset,
    get_olympiadbench_test_dataset,
)
from tuning.training.eval_strategy import MATH500EvalStrategy

BUDGETS_HOURS = [4, 16, 64]
FRACTIONS = [0, 25, 50, 75, 100]

# (model) -> {budget_hours: [run ids for 0/25/50/75/100% SFT]}; the 100% column
# is the SFT lineage, the rest are GRPO lineages. Mirrors the W&B grid tables.
GRID_RUNS = {
    "gemma3-4B": {
        4: ["k0oi912e", "kqlan2ae", "zhn17te2", "2f22vn9u", "dcoa9vsp"],
        16: ["k0oi912e", "fgdhqk89", "bohcwj92", "wkzlqrzk", "dcoa9vsp"],
        64: ["k0oi912e", "wpsjue9e", "8tdbsj8s", "8um6gxvd", "dcoa9vsp"],
    },
    "llama3-3B": {
        4: ["pm6fdbzm", "xo8ppeei", "87dprku7", "refrrdkn", "1swanogm"],
        16: ["pm6fdbzm", "zrpfbnwa", "hbl3j927", "5p36cm1d", "1swanogm"],
        64: ["pm6fdbzm", "gx6ux0m4", "klmbedtb", "pwp7xcd8", "1swanogm"],
    },
    "gemma3-12B": {
        4: ["vmoh3nah", "clr8aodj", "44vrscyh", "xjxm20hk", "hc8s6t6m"],
        16: ["vmoh3nah", "rr58zq3g", "s88emh6a", "fy6x3gou", "hc8s6t6m"],
        64: ["vmoh3nah", "mv7y3xap", "6oshr446", "inj34k54", "hc8s6t6m"],
    },
    "llama3-8B": {
        4: ["9ou3bxvy", "8m6m2qkv", "cskocfi6", "kqf9wovo", "0i93c6my"],
        16: ["9ou3bxvy", "gwfoyk7s", "fqx0e3ym", "ar368tkg", "0i93c6my"],
        64: ["9ou3bxvy", "c80lwmn1", "ar9r252k", "haoeksck", "0i93c6my"],
    },
}

BENCHMARK_LOADERS = {
    "minervamath": get_minervamath_test_dataset,
    "olympiadbench": get_olympiadbench_test_dataset,
}


class MathVerifyEvalStrategy(MATH500EvalStrategy):
    """MATH500's math-verify pass@k scoring over an arbitrary boxed-answer dataset."""

    def __init__(self, name, dataset, k_values=None, n_samples=4):
        # Deliberately does not call super().__init__, which would load MATH-500;
        # only the scoring path (reference_answers, k_values) is inherited.
        k_values = k_values or [1, 2, 4]
        # pass_at_k returns 1.0 whenever n - c < k, so a k above the sample count
        # scores every prompt as solved rather than failing.
        if max(k_values) > n_samples:
            raise ValueError(
                f"k_values {k_values} exceed n_samples={n_samples}; "
                f"pass@k needs at least k samples per prompt"
            )
        self.k_values = k_values
        self._n_samples = n_samples
        self.stopping_k = k_values[0]
        self._name = name
        self.test_dataset = dataset
        self.reference_answers = {
            prompt: ref
            for prompt, ref in zip(dataset["prompt"], dataset["reference_answer"])
        }
        print(f"[MathVerifyEvalStrategy:{name}] k_values={k_values}, "
              f"n_samples={n_samples}, num_prompts={len(dataset)}")

    @property
    def id(self) -> str:
        return self._name


def resolve_cell_checkpoints(model: str, models_dir: str = MODELS_DIR):
    """Map each (budget, fraction) grid cell to its local target-snapshot directory.

    Target snapshots are named <model>_math500-p@1-<minutes>m_sft-<n>_<runid>;
    GRPO *training output* dirs carry an _rlvr- segment and are excluded. When a
    cell has several snapshot dirs (restart races), the newest one wins.
    """
    cells = []
    root = Path(models_dir)
    for budget, run_ids in GRID_RUNS[model].items():
        for fraction, run_id in zip(FRACTIONS, run_ids):
            minutes = budget * 60
            matches = [
                p for p in root.glob(f"{model}_math500-p@1-{minutes}m_sft-*_{run_id}")
                if "_rlvr-" not in p.name
            ]
            path = max(matches, key=lambda p: p.stat().st_mtime) if matches else None
            cells.append({
                "model": model,
                "budget_hours": budget,
                "sft_fraction": fraction,
                "run_id": run_id,
                "checkpoint": str(path) if path else None,
            })
    return cells


def load_done(output_path: Path):
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                row = json.loads(line)
                done.add((row["budget_hours"], row["sft_fraction"], row["benchmark"]))
    return done


def evaluate_model(model, benchmarks, output_path, limit_prompts=None, max_cells=None,
                   gpu_memory_utilization=0.85, n_samples=4, k_values=None):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    # The grid trained and evaluated with the simple template; this also sets
    # the stop-token list picked up by VLLMSamplingParamsConfig.
    set_chat_template(model, simple=True)

    strategies = {}
    for benchmark in benchmarks:
        dataset = BENCHMARK_LOADERS[benchmark](num_prompts=limit_prompts)
        strategies[benchmark] = MathVerifyEvalStrategy(
            benchmark, dataset, k_values=k_values, n_samples=n_samples,
        )

    cells = [c for c in resolve_cell_checkpoints(model) if c["checkpoint"]]
    if max_cells:
        cells = cells[:max_cells]
    done = load_done(output_path)
    print(f"[ood_eval] {model}: {len(cells)} local cells, {len(done)} already done")

    llm = LLM(
        model=HF_MODEL_MAP[model],
        enable_lora=True,
        max_lora_rank=128,
        max_model_len=6144,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
    )

    lora_id = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        chat_template = Path(cell["checkpoint"], "chat_template.jinja").read_text()
        tokenizer = AutoTokenizer.from_pretrained(cell["checkpoint"])
        lora_id += 1
        lora_request = LoRARequest(
            lora_name=f"adapter_{lora_id}", lora_int_id=lora_id,
            lora_path=cell["checkpoint"],
        )
        for benchmark, strategy in strategies.items():
            key = (cell["budget_hours"], cell["sft_fraction"], benchmark)
            if key in done:
                continue
            params = SamplingParams(**VLLMSamplingParamsConfig(
                n=strategy.n_samples, max_tokens=4096, temperature=0.5,
            ).model_dump())
            outputs = llm.chat(
                strategy.get_test_messages(), params,
                chat_template=chat_template, lora_request=lora_request,
            )
            results = [
                {"prompt": p, "responses": [r.text for r in out.outputs]}
                for p, out in zip(strategy.get_test_prompts(), outputs)
            ]
            scores = strategy.score_responses(results, tokenizer)
            row = {
                **{k: cell[k] for k in
                   ("model", "budget_hours", "sft_fraction", "run_id", "checkpoint")},
                "benchmark": benchmark,
                **{k: scores[k] for k in scores if k.startswith("pass_at_")},
                "num_prompts": scores["num_prompts_evaluated"],
                "avg_response_length_tokens": scores["avg_response_length_tokens"],
                "per_prompt_correct_counts": [
                    [len(r["per_response_correct"]), sum(r["per_response_correct"])]
                    for r in results
                ],
            }
            with open(output_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[ood_eval] {model} {cell['budget_hours']}h/{cell['sft_fraction']}% "
                  f"{benchmark}: pass@1={scores['pass_at_1']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(GRID_RUNS))
    parser.add_argument("--benchmarks", nargs="+", default=list(BENCHMARK_LOADERS),
                        choices=list(BENCHMARK_LOADERS))
    parser.add_argument("--output", default=None,
                        help="Output JSONL; defaults to tuning/outputs/ood/<model>.jsonl")
    parser.add_argument("--limit-prompts", type=int, default=None,
                        help="Evaluate only the first N prompts per benchmark (smoke test)")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Evaluate only the first N resolved cells (smoke test)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--n-samples", type=int, default=4,
                        help="Generations per prompt; must be >= max(--k-values)")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 2, 4],
                        help="pass@k values to record")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print resolved cell checkpoints and exit")
    args = parser.parse_args()

    if args.dry_run:
        for cell in resolve_cell_checkpoints(args.model):
            mark = "OK " if cell["checkpoint"] else "-- "
            print(f"{mark}{cell['budget_hours']:>2}h {cell['sft_fraction']:>3}% "
                  f"{cell['run_id']} {cell['checkpoint'] or '(missing locally)'}")
        return

    output_path = Path(args.output) if args.output else (
        Path(MODELS_DIR).parent / "outputs" / "ood" / f"{args.model}.jsonl"
    )
    evaluate_model(
        args.model, args.benchmarks, output_path,
        limit_prompts=args.limit_prompts, max_cells=args.max_cells,
        gpu_memory_utilization=args.gpu_memory_utilization,
        n_samples=args.n_samples, k_values=args.k_values,
    )


if __name__ == "__main__":
    main()
