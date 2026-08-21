# ABOUTME: Runs our benchmark suite on external post-trained models under several
# ABOUTME: prompt/template/decoding protocols, to separate harness effects from training quality.

"""Calibrate our eval harness against externally reported numbers.

Evaluates any HF model on the same benchmarks and scorers the training callback
uses, but lets the protocol vary along three axes so the gap to a published
number can be attributed:

  --prompt-style   ours   : our system message + "Problem: ...\\nAnswer:" wrapper
                   boxed  : the OpenMath / nemo-skills instruction, bare problem
                   plain  : the bare problem with no system message
  --template       simple : the SIMPLE_TEMPLATE our runs train and evaluate with
                   native : the chat template shipped with the model
                   repo   : the family template this repo maps the model to
  --temperature / --top-p / --top-k / --n-samples : decoding

Inference reuses the callback's vLLM helpers so the engine, template rendering
and scoring path match training-time evaluation exactly.

Usage:
    python scripts/external_eval_calibration.py \
        --model nvidia/OpenMath2-Llama3.1-8B --model-family llama3-8B \
        --benchmarks math500,gsm8k,amc \
        --prompt-style boxed --template native --temperature 0.0 --n-samples 1 \
        --out outputs/eval_calibration/openmath2_theirs.json
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone

from tuning.data.config import (
    COMPMATH_STRING,
    GSM8K_STRING,
    SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING,
    SYSTEM_MESSAGE_OPENMATH,
)

PROMPT_STYLES = ("ours", "boxed", "plain")
TEMPLATES = ("simple", "native", "repo")

# The instruction nvidia/OpenMath2-Llama3.1-8B ships in its model card example,
# and that nemo-skills uses for MATH/GSM8K evaluation.
BOXED_INSTRUCTION = (
    "Solve the following math problem. Make sure to put the answer "
    "(and only answer) inside \\boxed{}."
)

_MATH_PREFIX = COMPMATH_STRING.split("{problem}")[0]
_MATH_SUFFIX = COMPMATH_STRING.split("{problem}")[1]
_GSM8K_PREFIX = GSM8K_STRING.split("{question}")[0]
_GSM8K_SUFFIX = GSM8K_STRING.split("{question}")[1]

MATH_BENCHMARKS = ("math500", "math", "amc", "aime24", "gsm8k")

# Merged SFT-parent weights (~16 GB per parent) are cached under here, one dir
# per parent, and shared by every arm that serves a child of that parent.
DEFAULT_MERGE_ROOT = os.path.expanduser("~/scratch/eval_calibration_merged")


def strip_prompt_wrapper(prompt: str) -> str:
    """Recover the raw problem text from a repo-formatted eval prompt.

    Math and GSM8K prompts are wrapped as "Problem: {problem}\\nAnswer:" /
    "Question: {question}\\nAnswer:". Instruction-following prompts are stored
    raw and pass through unchanged.
    """
    for prefix, suffix in ((_MATH_PREFIX, _MATH_SUFFIX), (_GSM8K_PREFIX, _GSM8K_SUFFIX)):
        if prompt.startswith(prefix) and prompt.endswith(suffix):
            return prompt[len(prefix): len(prompt) - len(suffix)]
    return prompt


def build_messages(style: str, benchmark: str, prompt: str) -> list[dict]:
    """Chat messages for one eval prompt under the requested prompt style."""
    if style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style {style!r}; expected one of {PROMPT_STYLES}")

    is_math = benchmark in MATH_BENCHMARKS

    if style == "ours":
        system = SYSTEM_MESSAGE_OPENMATH if is_math else SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    if style == "boxed":
        if not is_math:
            return [{"role": "user", "content": prompt}]
        problem = strip_prompt_wrapper(prompt)
        return [{"role": "user", "content": f"{BOXED_INSTRUCTION}\n\n{problem}"}]

    # plain
    content = strip_prompt_wrapper(prompt) if is_math else prompt
    return [{"role": "user", "content": content}]


def remap_strategy_prompts(strategy, style: str):
    """Rewrite a strategy's chat messages in place, leaving prompts/refs untouched.

    Scoring and reference-answer lookup are keyed on the `prompt` column, so only
    `messages` changes and every downstream scorer keeps working.
    """
    dataset = strategy.test_dataset
    messages = [build_messages(style, strategy.id, p) for p in dataset["prompt"]]
    strategy.test_dataset = dataset.remove_columns("messages").add_column("messages", messages)
    return strategy


def build_strategy(benchmark: str, n_samples: int, k_values: list[int], num_prompts,
                   strict: dict):
    from tuning.training.eval_strategy import (
        AIME24EvalStrategy,
        AMCEvalStrategy,
        GSM8KEvalStrategy,
        IFBenchStrategy,
        IFEvalStrategy,
        MATH500EvalStrategy,
        MATHEvalStrategy,
    )

    if benchmark == "math500":
        return MATH500EvalStrategy(k_values=k_values, n_samples=n_samples, num_prompts=num_prompts)
    if benchmark == "math":
        return MATHEvalStrategy(k_values=k_values, n_samples=n_samples, num_prompts=num_prompts)
    if benchmark == "aime24":
        return AIME24EvalStrategy(k_values=k_values, n_samples=n_samples, num_prompts=num_prompts)
    if benchmark == "gsm8k":
        return GSM8KEvalStrategy(k_values=k_values, n_samples=n_samples, num_prompts=num_prompts)
    if benchmark == "amc":
        return AMCEvalStrategy(k_values=k_values, n_samples=n_samples, num_prompts=num_prompts)
    if benchmark == "ifeval":
        kwargs = {"k_values": k_values, "n_samples": n_samples, "strict": strict["ifeval"]}
        if num_prompts is not None:
            kwargs["num_prompts"] = num_prompts
        return IFEvalStrategy(**kwargs)
    if benchmark == "ifbench":
        return IFBenchStrategy(k_values=k_values, n_samples=n_samples, num_prompts=num_prompts,
                               strict=strict["ifbench"])
    raise ValueError(f"Unknown benchmark {benchmark!r}")


def cross_score(benchmark: str, results: list[dict], strategy, tokenizer) -> dict:
    """Re-score the same generations under the alternative scorer for this benchmark.

    Math: GSM8K is graded by the lm-eval numeric path and MATH-500 / AMC by
    math-verify, so each is also scored with the other grader. Instruction
    following: our IFEval runs score loose and our IFBench runs score strict, so
    each is also scored the other way. Both isolate scoring choices from how many
    problems the model actually solved.
    """
    if benchmark not in MATH_BENCHMARKS:
        original = strategy.strict
        try:
            strategy.strict = not original
            flipped = strategy.score_responses(results, tokenizer)
        finally:
            strategy.strict = original
        label = "loose" if original else "strict"
        return {f"{k}_{label}": float(v) for k, v in flipped.items()
                if k.startswith("pass_at_")}

    import numpy as np
    from tuning.evaluation.gsm8k_scoring import is_correct as gsm8k_is_correct
    from tuning.evaluation.math500_scoring import is_correct as mathverify_is_correct

    alt = gsm8k_is_correct if benchmark != "gsm8k" else mathverify_is_correct
    per_prompt = []
    for item in results:
        ref = strategy.reference_answers[item["prompt"]]
        per_prompt.append(np.mean([bool(alt(r, ref)) for r in item["responses"]]))
    label = "gsm8k_numeric" if benchmark != "gsm8k" else "math_verify"
    return {f"pass_at_1_{label}_grader": float(np.mean(per_prompt))}


def canonical_answer(response: str):
    """Hashable key for the final answer a response commits to, or None.

    The last \\boxed{} (falling back to a GSM8K-style "#### answer") is parsed
    with math-verify so that \\frac{1}{2}, 1/2 and 0.5 vote as one answer.
    """
    from tuning.data.test_dataset import last_boxed_content
    from tuning.evaluation.math500_scoring import EXTRACTION_CONFIG, HASH_PATTERN, parse

    try:
        content = last_boxed_content(response)
    except ValueError:
        match = HASH_PATTERN.search(response)
        if not match:
            return None
        content = match.group(1).strip()
    parsed = parse(rf"\boxed{{{content}}}", extraction_config=EXTRACTION_CONFIG)
    if not parsed:
        return None
    expr = parsed[0]
    if getattr(expr, "is_Float", False) and expr == int(expr):
        return str(int(expr))
    return str(expr)


def majority_vote_scores(results: list[dict], reference_answers: dict,
                         k_values: list[int]) -> dict:
    """maj@k over the first k responses of each prompt, for every k every prompt can fill.

    Responses vote by canonical_answer; unparsable responses abstain and ties go
    to the answer seen first. Each distinct answer is graded once per prompt with
    the math-verify grader, on the first response that produced it.
    """
    from tuning.evaluation.math500_scoring import is_correct

    usable = [k for k in k_values if all(k <= len(item["responses"]) for item in results)]
    per_k = {k: [] for k in usable}
    for item in results:
        ref = reference_answers[item["prompt"]]
        keys = [canonical_answer(r) for r in item["responses"]]
        correct_by_key = {}
        for k in usable:
            votes = Counter(key for key in keys[:k] if key is not None)
            if not votes:
                per_k[k].append(0.0)
                continue
            winner = votes.most_common(1)[0][0]
            if winner not in correct_by_key:
                first = item["responses"][keys.index(winner)]
                correct_by_key[winner] = bool(is_correct(first, ref))
            per_k[k].append(float(correct_by_key[winner]))
    return {f"maj_at_{k}": sum(v) / len(v) for k, v in per_k.items()}


def resolve_template(tokenizer, template: str, model_family: str) -> str:
    """Return the Jinja template string to render eval prompts with."""
    import tuning.config as tuning_config
    from tuning.utils.utils import (
        GEMMA_3_CHAT_TEMPLATE,
        LLAMA_31_SIMPLE_TEMPLATE,
        SIMPLE_TEMPLATE,
    )

    if template == "native":
        native = tokenizer.chat_template
        if native is None:
            raise ValueError(
                f"--template native requested but {model_family} ships no chat_template"
            )
        # Stop strings still come from the family mapping, which matches the
        # special tokens the native template emits.
        tuning_config.set_chat_template(model_family, simple=False)
        return native

    if template == "simple":
        tuning_config.set_chat_template(model_family, simple=True)
        return SIMPLE_TEMPLATE

    resolved = tuning_config.set_chat_template(model_family, simple=False)
    return {
        "llama-3.1": LLAMA_31_SIMPLE_TEMPLATE,
        "gemma-3": GEMMA_3_CHAT_TEMPLATE,
    }.get(resolved, tokenizer.chat_template)


def sft_parent_adapter(model: str):
    """Path of the SFT adapter a GRPO adapter was trained on top of, or None.

    A GRPO run's training_config.json names its output dir
    <sft parent dir>_rlvr-<set>-<n>_grpo_<id>; the parent is that prefix as a
    sibling of the adapter dir. adapter_config.base_model_name_or_path only
    records the plain base, because the parent adapter was merged before RL.
    """
    from tuning.training.model_utils import is_adapter_checkpoint

    config_path = os.path.join(model, "training_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as fh:
        output_dir = os.path.basename(json.load(fh)["output_dir"].rstrip("/"))
    if "_rlvr-" not in output_dir:
        return None
    parent = os.path.join(os.path.dirname(os.path.abspath(model)), output_dir.split("_rlvr-")[0])
    if not is_adapter_checkpoint(parent):
        raise FileNotFoundError(
            f"{model} is a GRPO adapter trained on {parent}, which is not an adapter dir"
        )
    return parent


MERGE_DONE_MARKER = "merge_complete"


def merge_adapter_into_base(base: str, adapter: str, out: str) -> str:
    """Write base weights with the adapter merged in (plus the adapter's tokenizer) to out.

    A finished merge is marked and reused; concurrent jobs merging into the same
    out each write a private dir and the first rename wins.
    """
    import shutil

    if os.path.isfile(os.path.join(out, MERGE_DONE_MARKER)):
        print(f"[calibration] reusing merged weights at {out}", flush=True)
        return out

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[calibration] merging {adapter} into {base} -> {out}", flush=True)
    staging = f"{out}.staging-{os.getpid()}"
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.save_pretrained(staging, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(staging)
    open(os.path.join(staging, MERGE_DONE_MARKER), "w").close()
    try:
        os.rename(staging, out)
    except OSError:
        if not os.path.isfile(os.path.join(out, MERGE_DONE_MARKER)):
            raise
        shutil.rmtree(staging)
    return out


def resolve_model(model: str, merge_root: str = DEFAULT_MERGE_ROOT):
    """Map --model to (engine model path, LoRA adapter path, adapter rank).

    A LoRA adapter checkpoint is served as its base model with the adapter
    attached. For a GRPO adapter that base is the SFT parent adapter merged into
    the plain base, cached under merge_root. Anything else is served directly.
    """
    from tuning.training.model_utils import _adapter_base_path, is_adapter_checkpoint

    if not is_adapter_checkpoint(model):
        return model, None, None
    with open(os.path.join(model, "adapter_config.json")) as fh:
        rank = json.load(fh)["r"]
    parent = sft_parent_adapter(model)
    if parent is None:
        return _adapter_base_path(model), model, rank
    merge_dir = os.path.join(merge_root, os.path.basename(parent))
    return merge_adapter_into_base(_adapter_base_path(parent), parent, merge_dir), model, rank


def run(args):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from tuning.inference.config_inference import VLLMSamplingParamsConfig
    from tuning.training.passk.runners import (
        RunnerConfig,
        VLLMRunner,
        _cleanup_llm,
        _make_llm,
    )

    model_path, adapter_path, adapter_rank = resolve_model(args.model, args.merge_root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    chat_template = resolve_template(tokenizer, args.template, args.model_family)

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    strategies = []
    for benchmark in benchmarks:
        n_samples = {"amc": args.amc_n_samples, "gsm8k": args.gsm8k_n_samples}.get(
            benchmark, args.n_samples)
        k_values = [k for k in args.k_values if k <= n_samples] or [1]
        strategy = build_strategy(
            benchmark, n_samples, k_values, args.num_prompts,
            strict={"ifeval": args.ifeval_strict, "ifbench": args.ifbench_strict},
        )
        strategies.append(remap_strategy_prompts(strategy, args.prompt_style))

    runner_config = RunnerConfig(
        base_model_hf=model_path,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
        lora_max_rank=adapter_rank or 32,
        chat_template=chat_template,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        available_gpus=[],
        num_inference_gpus=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    sample_messages = strategies[0].get_test_messages()[0]
    rendered = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=True,
        chat_template=chat_template,
    )
    print("=" * 70)
    print(f"[calibration] model={args.model} family={args.model_family}")
    if adapter_path:
        print(f"[calibration] serving adapter (r={adapter_rank}) on base {model_path}")
    print(f"[calibration] template={args.template} prompt_style={args.prompt_style}")
    print(f"[calibration] temperature={args.temperature} top_p={args.top_p} "
          f"top_k={args.top_k} max_tokens={args.max_tokens}")
    print(f"[calibration] sample rendered prompt:\n{rendered}")
    print("=" * 70, flush=True)

    llm = _make_llm(runner_config, model_path=model_path, enable_lora=adapter_path is not None)
    lora_request = None
    if adapter_path:
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(lora_name="adapter", lora_int_id=1, lora_path=adapter_path)
    report = {
        "model": args.model,
        "base_model": model_path,
        "adapter": adapter_path,
        "model_family": args.model_family,
        "template": args.template,
        "prompt_style": args.prompt_style,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "n_samples": args.n_samples,
        "amc_n_samples": args.amc_n_samples,
        "gsm8k_n_samples": args.gsm8k_n_samples,
        "sample_prompt": rendered,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": {},
    }

    try:
        for strategy in strategies:
            sampling_config = VLLMSamplingParamsConfig(
                n=strategy.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_tokens=args.max_tokens,
            )
            sampling_params = SamplingParams(**sampling_config.model_dump())
            print(f"\n[calibration] {strategy.id}: {sampling_params}", flush=True)

            outputs = llm.chat(
                strategy.get_test_messages(),
                sampling_params,
                chat_template=chat_template,
                lora_request=lora_request,
            )
            results = VLLMRunner._format_outputs(outputs, strategy)
            scores = {k: float(v) for k, v in
                      strategy.score_responses(results, tokenizer).items()}
            scores.update(cross_score(strategy.id, results, strategy, tokenizer))
            if strategy.id in MATH_BENCHMARKS:
                scores.update(majority_vote_scores(results, strategy.reference_answers,
                                                   args.k_values))
            report["benchmarks"][strategy.id] = scores
            print(f"[calibration] {strategy.id}: "
                  + ", ".join(f"{k}={v:.4f}" for k, v in scores.items()), flush=True)

            if args.save_generations:
                gen_path = f"{os.path.splitext(args.out)[0]}.{strategy.id}.generations.jsonl"
                with open(gen_path, "w") as fh:
                    for item in results:
                        fh.write(json.dumps(item) + "\n")
                print(f"[calibration] generations -> {os.path.abspath(gen_path)}")
    finally:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[calibration] report -> {os.path.abspath(args.out)}")
        _cleanup_llm(llm)

    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="HF model id, local full checkpoint, or local LoRA adapter dir")
    parser.add_argument("--model-family", required=True,
                        help="Key into MODEL_CHAT_TEMPLATE_MAP, e.g. llama3-8B / gemma3-12B")
    parser.add_argument("--benchmarks", default="math500,gsm8k,amc",
                        help="Comma list from math500,math,amc,aime24,gsm8k,ifeval,ifbench")
    parser.add_argument("--prompt-style", default="ours", choices=PROMPT_STYLES)
    parser.add_argument("--template", default="simple", choices=TEMPLATES)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=150)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--amc-n-samples", type=int, default=8)
    parser.add_argument("--gsm8k-n-samples", type=int, default=None,
                        help="Defaults to --n-samples.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--ifeval-strict", action="store_true",
                        help="Score IFEval strict; our production runs score loose.")
    parser.add_argument("--ifbench-strict", dest="ifbench_strict",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Score IFBench strict (matches our production runs).")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--save-generations", action="store_true")
    parser.add_argument("--merge-root", default=DEFAULT_MERGE_ROOT,
                        help="Base+SFT-parent merged weights for a GRPO adapter are written "
                             "to <merge-root>/<parent dir name> and reused if already there.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.gsm8k_n_samples is None:
        args.gsm8k_n_samples = args.n_samples
    return args


if __name__ == "__main__":
    run(parse_args())
