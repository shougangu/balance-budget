# Per-Eval-Step Raw Generation Logging to W&B Tables Design

## Goal

Log full evaluation raw generations across the training run as W&B Tables, with one table per eval step and per eval strategy.

## Scope

- Keep existing scalar eval logging unchanged.
- Add per-eval-step W&B table logging for:
  - Primary eval strategy
  - All monitor eval strategies
- Do not add artifact-based logging in this change.

## Architecture

1. Extend `PassAtKStoppingCallback` to emit a raw-generation table after each eval run.
2. Refactor evaluation execution to provide both:
   - Aggregate `scores` (existing behavior)
   - Raw `model_results` (`prompt` + generated `responses`)
3. Log each table under a unique per-step key:
   - `eval/raw_generations/<eval_name>/step_<global_step>`
4. Preserve existing eval metric logs (e.g., `eval/pass_at_1`) for dashboards and stopping.

## Table Schema

Each row is one prompt at one eval step.

- `global_step` (int)
- `eval_name` (str)
- `prompt_index` (int)
- `prompt` (str)
- `responses` (str, JSON-serialized list)
- `num_responses` (int)
- `stopping_metric_name` (str)
- `stopping_metric_value` (float)
- `thresholds_remaining` (str, JSON-serialized list)
- `timestamp_utc` (str, ISO8601)

## Data Flow

1. `on_evaluate` runs primary eval.
2. Callback logs scalar metrics as before.
3. Callback builds and logs a step-scoped W&B table from primary raw results.
4. Callback runs each monitor eval, logging:
   - Scalar metrics (existing behavior)
   - Its own step-scoped raw table
5. Threshold checks and checkpoint behavior remain unchanged.

## Error Handling

- Table logging is best-effort; failures should not interrupt training.
- If raw results are empty, skip table logging and print a warning.
- If response JSON serialization fails for a row, coerce responses to strings and continue.

## Testing Plan

1. Add callback tests to verify per-step table logging key and row content for primary eval.
2. Add callback tests to verify monitor eval tables are also logged.
3. Add callback tests to verify training continues if table logging throws.
4. Keep existing eval strategy tests unchanged.
