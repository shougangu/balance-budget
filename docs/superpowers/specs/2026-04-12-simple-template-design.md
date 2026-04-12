# Simple Template Mode for SFT, DPO, and Evaluation Callbacks

## Summary

Add a global `--simple-template` CLI flag that replaces the chat template with a minimal Jinja2 template that strips system prompts and chat scaffolding. Training and evaluation use raw prompt strings (e.g., `"Question: {q}\nAnswer: {a}"`) instead of chat-formatted text.

## Motivation

GRPO training already supports a simple template mode via manual prompt stripping (`grpo_training.py:43-49`). This design extends the same concept to SFT, DPO, and evaluation callbacks through a global chat template — a single Jinja2 string that the entire pipeline picks up automatically, eliminating per-trainer branching.

## Semantics

When `simple_template=True`:
- **No system message**: the template skips any message with `role == "system"`.
- **No chat scaffolding**: no `<|start_header_id|>`, `<|im_start|>`, role headers, or turn-ending tokens. Just BOS + raw content + EOS.
- **Applies everywhere**: training data rendering, DPOTrainer's internal tokenization, vLLM eval callbacks — all use the same tokenizer chat template.

Rendered examples (GSM8K):
- **Training** (`add_generation_prompt=False`): `<bos>Question: What is 2+2?\nAnswer:The answer is 4.\n\n#### 4<eos>`
- **Inference** (`add_generation_prompt=True`): `<bos>Question: What is 2+2?\nAnswer:`

For IFEval: raw instruction string with no system prompt, no Q/A framing.

## Design

### The Simple Chat Template (Jinja2)

Follows the same dual-format pattern as `LLAMA_31_SIMPLE_TEMPLATE` in `utils/utils.py` — handles both `role`/`content` (standard) and `from`/`value` (ShareGPT-mapped) message formats:

```jinja
{% if 'role' in messages[0] %}{{- bos_token }}\
{%- for message in messages %}\
{%- if message['role'] != 'system' %}\
{{ message['content'] }}\
{%- endif %}\
{%- endfor %}\
{%- if not add_generation_prompt %}\
{{- eos_token }}\
{%- endif %}\
{% else %}{{- bos_token }}\
{%- for message in messages %}\
{%- if message['from'] != 'system' %}\
{{ message['value'] }}\
{%- endif %}\
{%- endfor %}\
{%- if not add_generation_prompt %}\
{{- eos_token }}\
{%- endif %}\
{% endif %}
```

### Global Template Registration

`tuning/config.py`: `set_chat_template` gains a `simple: bool` parameter. When `True`, it stores the model's real template in `_BASE_CHAT_TEMPLATE` (needed for tokenizer setup in `get_chat_template`) and sets `DEFAULT_CHAT_TEMPLATE = "simple"`.

`tuning/utils/utils.py`: `chat_template_func` uses `_BASE_CHAT_TEMPLATE` for `get_chat_template(tokenizer, ...)` base setup (special tokens, ShareGPT mapping), then overrides `tokenizer.chat_template` with the simple Jinja2 string. Same pattern as the existing `llama-3.1` override at `utils.py:56-57`.

### Stop Tokens

`STOP_TOKENS["simple"]` covers both model families: `["<|end_of_text|>", "</s>"]`. No turn-ending tokens since the simple template has none.

### SFT Training

`sft_training.py`: `apply_chat_template(tokenizer, dataset)` still works — it calls `tokenizer.apply_chat_template` which uses the simple template, producing flat text in the `text` column. The only behavioral change: **skip `train_on_responses_only`** when the template is `"simple"`, since there are no structural delimiters to split on. Loss is computed over the entire sequence (the few extra tokens on the question are negligible).

### DPO Training

`dpo_training.py`: **No code changes.** DPOTrainer calls `tokenizer.apply_chat_template` internally. With the simple template set on the tokenizer, it renders raw strings (no system prompt, no chat scaffolding) automatically.

### GRPO Training

`grpo_training.py`: **Remove the manual `_strip_chat_template` logic** (lines 43-51). The simple template handles it now. The `simple_template` field on `PTRunConfig` is still set from the CLI flag but no longer needs per-trainer handling.

### Evaluation Callbacks

`passk_callback.py` and `eval_strategy.py`: **No code changes.** `llm.chat(messages, chat_template=self._chat_template)` continues to work — `self._chat_template` is read from `tokenizer.chat_template` at init time, which is now the simple Jinja2 string. vLLM renders messages through it and produces raw prompt text for generation.

Data-parallel workers (`_data_parallel_worker`): also unchanged — they receive `chat_template` as a parameter and pass it to `llm.chat()`.

### CLI Surface

`unified_early_pipeline.py`:
- **Remove** `--grpo-simple-template` (BooleanOptionalAction, default True).
- **Add** `--simple-template` (BooleanOptionalAction, default False).
- In `run_sft`, `run_dpo`, `run_grpo`: call `set_chat_template(args.model, simple=args.simple_template)` instead of `set_chat_template(args.model)`.
- `run_config.simple_template = args.simple_template` for PTRunConfig (GRPO/DPO).
- `SFTRunConfig` does not need `simple_template` — the global template handles everything.

### Config Changes

`config_training.py`: `simple_template` on `PTRunConfig` stays (used by GRPO/DPO run naming or downstream logic). No changes to `SFTRunConfig`.

## Files Changed

| File | Change |
|------|--------|
| `tuning/config.py` | `set_chat_template` gains `simple` param; stores `_BASE_CHAT_TEMPLATE` |
| `tuning/utils/utils.py` | Define `SIMPLE_TEMPLATE` Jinja2 string; handle in `chat_template_func`; add `STOP_TOKENS["simple"]` |
| `tuning/training/unified_early_pipeline.py` | Replace `--grpo-simple-template` with `--simple-template`; pass `simple=` to `set_chat_template` |
| `tuning/training/sft_training.py` | Skip `train_on_responses_only` when template is `"simple"` |
| `tuning/training/grpo_training.py` | Remove `_strip_chat_template` logic (template handles it) |

**No changes needed:**
- `tuning/training/dpo_training.py`
- `tuning/training/passk_callback.py`
- `tuning/training/eval_strategy.py`
- `tuning/data/test_dataset.py`
- `tuning/training/config_training.py`

## Testing

- Unit test: apply simple template to GSM8K messages → verify output matches `<bos>Question: ...\nAnswer: ...<eos>` (training) and `<bos>Question: ...\nAnswer:` (inference).
- Unit test: apply simple template to IFEval messages → verify no system prompt, raw instruction only.
- Unit test: verify `STOP_TOKENS["simple"]` is returned by `get_stop_tokens()`.
- Integration test: pipeline arg parsing accepts `--simple-template` / `--no-simple-template`, removed `--grpo-simple-template`.
- Existing tests: verify they still pass with default `simple_template=False` (no behavioral change).
