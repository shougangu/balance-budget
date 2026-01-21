"""
Pass@k evaluation script for IFEval.
Evaluates the probability that at least one of k sampled responses is correct.
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, '/home/shougan/projects/aip-fredashi/shougan/balance-budget')
BASE_DIR = Path('/home/shougan/projects/aip-fredashi/shougan/balance-budget')

from tuning.config import OUTPUTS_DIR
from tuning.inference.ifeval_inference import run_inference_ifeval
from tuning.data.test_dataset import get_ifeval_test_dataset
from instruction_following_eval import evaluation_lib


# === PATHS ===
OUTPUTS = Path(OUTPUTS_DIR) / "pass@k_responses"
IFEVAL_INPUT_PATH = BASE_DIR / "instruction_following_eval/data/input_data.jsonl"


# === UTILITY FUNCTIONS ===

def pass_at_k(n: int, c: int, k: int) -> float:
    """Calculate pass@k: probability that at least one of k samples is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k_scores(results_per_prompt: List[List[bool]], k_values: List[int]) -> Dict[int, float]:
    """Compute average pass@k across all prompts."""
    scores = {k: [] for k in k_values}
    for results in results_per_prompt:
        n, c = len(results), sum(results)
        for k in k_values:
            if k <= n:
                scores[k].append(pass_at_k(n, c, k))
    return {k: np.mean(v) for k, v in scores.items() if v}


def save_responses(results: List[Dict], model_name: str):
    path = OUTPUTS / model_name
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "responses_multi_sample.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved responses to {path / 'responses_multi_sample.jsonl'}")


def load_responses(model_name: str) -> List[Dict]:
    with open(OUTPUTS / model_name / "responses_multi_sample.jsonl") as f:
        return [json.loads(line) for line in f]


def evaluate_single_response(inp: evaluation_lib.InputExample, response: str, strict: bool = True) -> bool:
    """Evaluate a single response using the pre-built IFEval functions."""
    prompt_to_response = {inp.prompt: response}
    
    if strict:
        result = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
    else:
        result = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)
    
    return result.follow_all_instructions


# === MAIN FUNCTIONS ===

def run_inference(
    model_name: str,
    n_samples: int,
    temperature: float = 0.7,
    num_examples: int = None
) -> List[Dict]:
    print(f"\n{'='*50}")
    print(f"Running Inference: {model_name}")
    
    test_dataset = get_ifeval_test_dataset()
    if num_examples is not None:
        test_dataset = test_dataset.select(range(num_examples))
    
    raw_results = run_inference_ifeval(
        model_name=model_name,
        n_samples=n_samples,
        temperature=temperature,
        save_results=False,
        num_examples=num_examples
    )
    print(raw_results)
    # Group responses by prompt for pass@k evaluation
    # raw_results: [{prompt: "", response: ""}, ...] where each response is separate
    grouped = defaultdict(list)
    for r in raw_results:
        grouped[r["prompt"]].append(r["response"])  # Key is "response" (singular)
    
    model_results = [{"prompt": p, "responses": resps} for p, resps in grouped.items()]
    print(f"Generated {len(model_results)} prompts with {n_samples} samples each")
    save_responses(model_results, model_name)
    print("Responses saved.")
    return model_results


def evaluate_pass_at_k(
    model_name: str,
    k_values: List[int],
    model_results: List[Dict] = None # could be loaded
) -> Dict:
    print(f"\n{'='*50}")
    print(f"Evaluating Pass@k: {model_name}")
    print(f"{'='*50}")
    print(f"k_values: {k_values}")
    
    # Load IFEval inputs
    inputs_map = {inp.prompt: inp for inp in evaluation_lib.read_prompt_list(str(IFEVAL_INPUT_PATH))}
    print(f"Loaded {len(inputs_map)} IFEval prompts")
    
    # Load responses if not provided
    if model_results is None:
        print("Loading cached responses...")
        model_results = load_responses(model_name)
        print(f"Loaded {len(model_results)} results")
    
    # Evaluate responses
    print("\nEvaluating responses...")
    results_per_prompt = {}
    strict_all = []
    loose_all = []
    
    for item in tqdm(model_results, desc="Evaluating"):
        prompt = item["prompt"]
        responses = item["responses"]
        
        if prompt not in inputs_map:
            print(f"Warning: Prompt not found in inputs_map: {prompt[:50]}...")
            continue
            
        eval_input = inputs_map[prompt]
        
        strict_results = [evaluate_single_response(eval_input, r, strict=True) for r in responses]
        loose_results = [evaluate_single_response(eval_input, r, strict=False) for r in responses]
        
        strict_all.append(strict_results)
        loose_all.append(loose_results)
        
        results_per_prompt[prompt] = {
            "strict": strict_results,
            "loose": loose_results,
            "strict_pass_at_k": {k: pass_at_k(len(strict_results), sum(strict_results), k) for k in k_values},
            "loose_pass_at_k": {k: pass_at_k(len(loose_results), sum(loose_results), k) for k in k_values},
        }
    
    # Compute aggregate scores
    strict_scores = compute_pass_at_k_scores(strict_all, k_values)
    loose_scores = compute_pass_at_k_scores(loose_all, k_values)
    
    # Save results
    output = {
        "model": model_name,
        "k_values": k_values,
        "num_prompts": len(model_results),
        "strict_pass_at_k": strict_scores,
        "loose_pass_at_k": loose_scores,
        "per_prompt_results": results_per_prompt
    }
    
    output_path = OUTPUTS / model_name / "pass_at_k_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved results to {output_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*50}")
    print(f"Strict pass@k: {strict_scores}")
    print(f"Loose pass@k:  {loose_scores}")
    
    return output


def main():
    parser = argparse.ArgumentParser(description="Run pass@k evaluation for IFEval")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64], help="k values for pass@k")
    parser.add_argument("--n-samples", type=int, default=128, help="Number of samples per prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--num-examples", type=int, default=None, help="Number of examples (None for all)")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference and use cached responses")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation (only run inference)")
    
    args = parser.parse_args()
    
    model_results = None
    
    # Step 1: Run inference (unless skipped)
    print(f"Arguments: {args}")
    if not args.skip_inference:
        model_results = run_inference(
            model_name=args.model,
            n_samples=args.n_samples,
            temperature=args.temperature,
            num_examples=args.num_examples
        )
    
    # Step 2: Run evaluation (unless skipped)
    if not args.skip_eval:
        evaluate_pass_at_k(
            model_name=args.model,
            k_values=args.k_values,
            model_results=model_results
        )


if __name__ == "__main__":
    main()
