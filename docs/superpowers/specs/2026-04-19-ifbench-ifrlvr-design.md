# IFBench Eval + IF-RLVR Training Integration

Based on "Generalizing Verifiable Instruction Following" (arXiv 2507.02833v1) by AllenAI.

## Problem

IFEval (25 constraint templates, 541 prompts) is saturated — models score 80%+ by overfitting to its small constraint set, not by genuinely following instructions. We need:

1. **IFBench** — an out-of-distribution eval benchmark with 58 novel constraint templates (300 prompts) to measure true instruction-following generalization.
2. **IF-RLVR** — a training data source (95k prompts with up to 5 verifiable constraints each) for GRPO-based instruction-following training.

## Architecture

### Vendored Constraint Packages

Two separate constraint codebases with zero overlap in constraint IDs:

```
balance-budget/
  instruction_following_eval/   # existing IFEval (25 constraints) — unchanged except evaluation_lib.py
  ifbench_eval/                 # 58 OOD eval constraints (vendored from allenai/IFBench)
    __init__.py
    instructions.py
    instructions_registry.py    # INSTRUCTION_DICT
    instructions_util.py
  ifrlvr/                       # 54 training constraints (vendored from open-instruct/IFEvalG)
    __init__.py
    instructions.py
    instructions_registry.py    # INSTRUCTION_DICT
```

All three packages share the same `Instruction` base class interface:
- `build_description(**kwargs)` — configure constraint parameters
- `check_following(value: str) -> bool` — verify response compliance
- `get_instruction_args()` / `get_instruction_args_keys()`

### Shared Evaluation Harness

IFBench reuses the existing `instruction_following_eval/evaluation_lib.py` with one refactor:

**Make `instruction_dict` injectable** in `test_instruction_following_strict` and `test_instruction_following_loose`:

```python
def test_instruction_following_strict(inp, prompt_to_response, instruction_dict=None):
    if instruction_dict is None:
        instruction_dict = instructions_registry.INSTRUCTION_DICT  # default: IFEval
    # ... rest uses instruction_dict
```

**Add null-filtering** for kwargs before calling `build_description()`:
```python
inp.kwargs[index] = {k: v for k, v in inp.kwargs[index].items() if v is not None}
```

Both changes are backward-compatible — existing IFEval code works unchanged.

## Components

### 1. IFBench Eval Strategy

**File:** `tuning/training/eval_strategy.py` — new class `IFBenchStrategy(EvalStrategy)`

**Test data loader:** `get_ifbench_test_dataset()` in `tuning/data/test_dataset.py`
- Source: `allenai/IFBench_test` on HuggingFace (300 prompts)
- Format: `{messages, prompt, instruction_id_list, kwargs}`
- System message: `SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING` (reused)
- Single-turn only (multi-turn deferred)

**Scoring:** Calls the shared `evaluation_lib.test_instruction_following_strict/loose` with `instruction_dict=IFBENCH_INSTRUCTION_DICT`.
- Instruction-level pass@k: fraction of individual constraints passed
- Prompt-level pass@k: all-or-nothing per prompt

**IFBench eval kwargs format** (dense, all keys present, irrelevant ones null):
```json
{"keyword1": "kaleidoscope", "keyword2": "nebula", "num_paragraphs": null, "N": null, ...}
```
Null-filtering in the shared evaluation_lib handles this.

**W&B metrics:** `eval/ifbench_pass_at_{k}`, `eval/ifbench_pass_at_{k}_prompt`, `eval/ifbench_avg_response_length_tokens`

**Registration:** Both `--task-name ifbench` and `--monitor-evals ifbench`.

### 2. IF-RLVR Dataset

**File:** `tuning/data/ifrlvr_rlvr.py` — new class `IfrlvrRLVR(HFDataset)`

**Source:** `allenai/IF_multi_constraints_upto5` on HuggingFace (95,373 rows)

**Format per row:**
```python
{
    "prompt": [
        {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
        {"role": "user", "content": messages[0]["content"]},  # constraints already baked in
    ],
    "ground_truth": ground_truth_string,  # preserved for reward function
}
```

The `ground_truth` field is a string-encoded Python list of dicts:
```python
[{"instruction_id": ["copy:copying_multiple", "length_constraints:number_paragraphs"],
  "kwargs": [{"prompt_to_repeat": "...", "N": 3}, {"num_paragraphs": 6}]}]
```
Some kwargs entries are large because `prompt_to_repeat` includes the full prompt text.

Deduplication by prompt text, same pattern as `TuluIFRLVR`.

### 3. IF-RLVR Reward Function

**File:** `tuning/training/reward_functions.py` — new function `ifrlvr_reward_func`

Follows the canonical `IFEvalVerifier.__call__` pattern from `open_instruct/ground_truth_utils.py`:

```python
def ifrlvr_reward_func(prompts, completions, ground_truth, **kwargs):
    # For each (prompt, completion, ground_truth):
    #   1. ast.literal_eval(ground_truth) -> list of dicts
    #   2. constraint_dict = result[0]
    #   3. answer = remove_thinking_section(completion)
    #   4. For each (instruction_id, kwargs) pair:
    #        - args = {} if kw is None else {k:v for k,v in kw.items() if v is not None}
    #        - checker = ifrlvr.instructions_registry.INSTRUCTION_DICT[inst_id](inst_id)
    #        - checker.build_description(**args)
    #        - passed = checker.check_following(answer)
    #   5. reward = passed / total (fractional)
```

Key details:
- Uses `ast.literal_eval()` (not `json.loads()`) — the field uses Python syntax
- `remove_thinking_section()` strips `<|assistant|>`, everything before `</think>`, and `<answer>`/`</answer>` tags before checking constraints. Defined as a helper in `reward_functions.py`.
- Lazy-loads the IFEvalG registry on first call
- `ground_truth` column flows through TRL's GRPOTrainer as a dataset column

### 4. Pipeline Registration

**File:** `tuning/training/unified_early_pipeline.py`

| Registration point | Change |
|---|---|
| `--dataset` choices | Add `"ifrlvr"` |
| `--task-name` choices | Add `"ifbench"` |
| `--monitor-evals` choices | Add `"ifbench"` |
| `_build_reward_funcs()` | `"ifrlvr"` → `[ifrlvr_reward_func]` |
| `_build_eval_components()` | `"ifbench"` → `IFBenchStrategy(...)` |
| `_build_monitor_evals()` | `"ifbench"` → `IFBenchStrategy(...)` |

## Dependencies

**New pip dependencies:** `langdetect`, `syllapy`, `emoji`

**NLTK data downloads:** `punkt`, `punkt_tab`, `stopwords`, `averaged_perceptron_tagger_eng`

## Data Sources

| Dataset | HuggingFace path | Size | Purpose |
|---------|-----------------|------|---------|
| IFBench test | `allenai/IFBench_test` | 300 prompts | Eval |
| IF-RLVR training | `allenai/IF_multi_constraints_upto5` | 95,373 prompts | GRPO training |

## Usage Examples

```bash
# Train IF-RLVR with GRPO, primary eval on IFEval, monitor OOD on IFBench
python tuning/training/unified_early_pipeline.py \
  --dataset ifrlvr --task-name ifeval --monitor-evals ifbench \
  --model llama3-8B --run-grpo

# Monitor IFBench alongside existing math training
python tuning/training/unified_early_pipeline.py \
  --dataset gsm8k --task-name gsm8k --monitor-evals ifbench \
  --model llama3-8B

# Use IFBench as primary eval
python tuning/training/unified_early_pipeline.py \
  --dataset ifrlvr --task-name ifbench \
  --model llama3-8B --run-grpo
```

## Testing

- Unit tests for null-filtering in evaluation_lib (kwargs with None values)
- Unit tests for `ifrlvr_reward_func` with sample ground_truth strings
- Unit tests for `IFBenchStrategy` scoring with sample responses
- Integration test: load IF-RLVR dataset, verify format
- Integration test: load IFBench test dataset, verify scoring pipeline

## Scope exclusions

- Multi-turn IFBench eval (deferred)
- Preference RM mixing to prevent reward hacking (optional enhancement)
- DPO/SFT data variants of IF-RLVR (RLVR only for now)
