# Eval Strategy Refactor Design

Refactor `passk_callback.py` to separate eval-specific logic (prompts, scoring) from
infrastructure (vLLM lifecycle, stopping/forking, checkpointing) using composition.

## Interface

```python
class EvalStrategy(ABC):
    def get_test_messages(self) -> list[list[dict]]:
        """Chat messages to send to vLLM."""

    def score_responses(self, results: list[dict]) -> dict[str, float]:
        """Score vLLM outputs. Returns metric dict (e.g., {"pass_at_1": 0.72})."""

    def stopping_metric(self) -> str:
        """Which key from score_responses() to use for thresholds."""

    def wandb_metrics(self, scores: dict) -> dict[str, float]:
        """Format scores for wandb logging."""
```

## Callback Design

`PassAtKStoppingCallback.__init__` takes:
- `primary_eval: EvalStrategy` — drives stopping/forking
- `monitor_evals: list[EvalStrategy] = []` — logged to wandb only
- Stopping config (thresholds, early_tuples) stays on the callback
- vLLM config stays on the callback

`on_evaluate` flow:
1. Save LoRA adapter to temp dir
2. Primary eval: get prompts → vLLM inference → score → check thresholds → checkpoint if needed
3. Each monitor eval: get prompts → vLLM inference → score → log to wandb
4. Cleanup

## Checkpoint Labels

Labels include the eval metric prefix for consistency:
- Absolute threshold: `p@1-0.5` (unchanged)
- Patience early tuple: `p@1-3@0.5` (adds metric prefix, was `3@0.5`)

The eval strategy provides a `label_prefix` property (e.g., `"p@1"`) so the callback
can construct labels generically: `f"{prefix}-{threshold}"` or `f"{prefix}-{patience}@{min_increase}"`.

## Files

- `tuning/training/eval_strategy.py` — ABC + `IFEvalStrategy`
- `tuning/training/passk_callback.py` — refactored to use eval strategies

## What moves to IFEvalStrategy

- `evaluate_single_response()`, `pass_at_k()` functions
- IFEval dataset loading (`get_ifeval_test_dataset()`, `inputs_map`)
- Response scoring loop (the per-prompt evaluation in `evaluate_pass_at_k`)
- k_values, n_samples, strict config
- Token length computation

## What stays in passk_callback.py

- All vLLM machinery (persistent/ephemeral/data-parallel)
- LoRA adapter saving
- Threshold + early_tuples + Fork Strategy logic
- `on_train_begin`, `on_train_end`, `on_evaluate` orchestration
- Checkpoint saving via `callback_utils`
