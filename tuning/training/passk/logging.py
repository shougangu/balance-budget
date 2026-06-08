# ABOUTME: W&B and console logging helpers for the Pass@K callback.
# ABOUTME: One place that knows the [PassAtKCallback] prefix and the train/global_step keys.

import datetime
import json
from typing import Dict, List

import wandb


_LOG_PREFIX = "[PassAtKCallback]"


def log_eval_metrics(
    *,
    eval_strategy,
    scores: Dict[str, float],
    raw_results: List[Dict],
    global_step: int,
    step_offset: int,
    total_minutes: float,
    thresholds_remaining: List[float],
    is_primary: bool,
) -> None:
    """Single entry point for wandb metrics + raw-generations table + console summary."""
    log_dict = {
        "train/global_step": global_step,
        "train/total_global_step": global_step + step_offset,
        "train/total_minutes": total_minutes,
    }
    log_dict.update(eval_strategy.wandb_metrics(scores))
    wandb.log(log_dict)

    stopping_key = eval_strategy.stopping_metric()
    stopping_value = scores.get(stopping_key)
    _log_raw_generation_table(
        eval_strategy=eval_strategy,
        model_results=raw_results,
        global_step=global_step,
        step_offset=step_offset,
        total_minutes=total_minutes,
        stopping_metric_name=stopping_key,
        stopping_metric_value=stopping_value,
        thresholds_remaining=thresholds_remaining,
    )

    label = "Primary" if is_primary else f"Monitor ({eval_strategy.__class__.__name__})"
    score_summary = ", ".join(
        f"{k}={v:.4f}" for k, v in scores.items() if isinstance(v, float)
    )
    if is_primary:
        print(f"\n{_LOG_PREFIX} Step {global_step}: {score_summary} "
              f"({scores.get('num_prompts_evaluated', '?')} prompts)")
    else:
        print(f"{_LOG_PREFIX} {label}: {score_summary}")


def _log_raw_generation_table(
    *,
    eval_strategy,
    model_results: List[Dict],
    global_step: int,
    step_offset: int,
    total_minutes: float,
    stopping_metric_name: str,
    stopping_metric_value,
    thresholds_remaining,
) -> None:
    """Best-effort: log raw generations as a per-step W&B Table."""
    eval_slug = eval_strategy.id
    table_key = f"raw_generations/{eval_slug}/step_{global_step}"
    try:
        table = wandb.Table(columns=[
            "global_step", "eval_name", "prompt_index", "prompt", "responses",
            "num_responses", "per_response_correct", "per_response_instructions",
            "prompt_accuracy", "stopping_metric_name", "stopping_metric_value",
            "thresholds_remaining", "timestamp_utc",
        ])
        timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        thresholds_remaining_json = json.dumps(thresholds_remaining)

        for prompt_index, item in enumerate(model_results):
            prompt = item.get("prompt", "")
            responses = item.get("responses", [])
            if not isinstance(responses, list):
                responses = [responses]
            try:
                responses_json = json.dumps(responses)
            except TypeError:
                responses_json = json.dumps([str(r) for r in responses])

            correctness = item.get("per_response_correct", [])
            prompt_accuracy = (sum(correctness) / len(correctness)
                               if correctness else None)
            instructions = item.get("per_response_instructions", [])

            table.add_data(
                global_step, eval_slug, prompt_index, prompt, responses_json,
                len(responses),
                json.dumps(correctness) if correctness else None,
                json.dumps(instructions) if instructions else None,
                prompt_accuracy, stopping_metric_name, stopping_metric_value,
                thresholds_remaining_json, timestamp_utc,
            )
        wandb.log({
            "train/global_step": global_step,
            "train/total_global_step": global_step + step_offset,
            "train/total_minutes": total_minutes,
            table_key: table,
        })
    except Exception as exc:
        print(f"{_LOG_PREFIX} Warning: failed to log raw generation table "
              f"({table_key}): {exc}")
