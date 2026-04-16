# Seed Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread the `--seed` CLI arg through all currently-hardcoded RNG in the unified early pipeline (training, LoRA init, data sampling, pass@k eval generation), add a `--eval_seed` override, and log both to W&B — all while preserving today's default behavior (42 everywhere).

**Architecture:** Pydantic configs (`TrainingArgumentsConfig`, `VLLMSamplingParamsConfig`, `PassAtKConfig`) gain `seed` fields. The pipeline reads `args.seed` and `args.eval_seed`, sets training/lora seeds via those configs, calls `random.seed(args.seed)` once before any data loading (replacing the module-level `random.seed(42)` calls in three data files), computes `effective_eval_seed = args.eval_seed if args.eval_seed is not None else args.seed`, and hands it to `PassAtKConfig.seed`. `PassAtKStoppingCallback` propagates that seed into every `VLLMSamplingParamsConfig` it constructs, including the subprocess data-parallel worker.

**Tech Stack:** Python, Pydantic, TRL (SFTConfig/DPOConfig/GRPOConfig), unsloth (FastLanguageModel.get_peft_model), vLLM (SamplingParams), pytest

**Spec:** `docs/superpowers/specs/2026-04-15-seed-wiring-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tuning/inference/config_inference.py` | Modify | Add `seed: Optional[int] = None` field to `VLLMSamplingParamsConfig` |
| `tuning/training/config_training.py` | Modify | Add `seed: int = 42` field to `TrainingArgumentsConfig`; `to_hf_args()` uses `self.seed`. Add `seed: Optional[int] = None` field to `PassAtKConfig` |
| `tuning/training/passk_callback.py` | Modify | `PassAtKStoppingCallback` reads `config.seed`; passes to every `VLLMSamplingParamsConfig` construction site (persistent, ephemeral, subprocess worker) |
| `tuning/data/utils.py` | Modify | Remove module-level `random.seed(42)` |
| `tuning/data/hf_dataset.py` | Modify | Remove module-level `random.seed(42)` |
| `tuning/data/test_dataset.py` | Modify | Remove module-level `random.seed(42)` |
| `tuning/training/unified_early_pipeline.py` | Modify | Add `--eval_seed` arg; in `run_sft`/`run_dpo`/`run_grpo`: call `random.seed(args.seed)`, set `lora_config.random_state = args.seed`, set `training_args.seed = args.seed`, set `passk_config.seed = effective_eval_seed`, log seeds to W&B config |
| `tests/test_vllm_sampling_params_global.py` | Modify | Add test: seed field exists with default None; seed is passed through to model_dump |
| `tests/test_grpo_config.py` | Modify | Add test: `TrainingArgumentsConfig(seed=7).to_hf_args(...)` injects 7 (not hardcoded 42) |
| `tests/test_unified_early_pipeline.py` | Modify | Add tests for `--seed` and `--eval_seed` parsing; effective-eval-seed helper |
| `tests/test_seed_wiring.py` | Create | Tests for `PassAtKConfig.seed` default; data-module import does not reseed global random |

---

### Task 1: Add `seed` field to `VLLMSamplingParamsConfig`

**Files:**
- Modify: `tuning/inference/config_inference.py:1-20`
- Test: `tests/test_vllm_sampling_params_global.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_vllm_sampling_params_global.py`:

```python
def test_vllm_sampling_params_seed_defaults_none():
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert config.seed is None


def test_vllm_sampling_params_seed_roundtrips_through_model_dump():
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig(seed=7)
    assert config.seed == 7
    assert config.model_dump()["seed"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vllm_sampling_params_global.py -v`
Expected: FAIL with `AttributeError: 'VLLMSamplingParamsConfig' object has no attribute 'seed'` for both new tests.

- [ ] **Step 3: Add `seed` field**

Edit `tuning/inference/config_inference.py` to read:

```python
from pydantic import BaseModel, model_validator
from typing import Optional


class VLLMSamplingParamsConfig(BaseModel):
    max_tokens: int = 4096
    temperature: float = 0.5
    top_k: int = 150
    top_p: float = 0.9
    stop: list[str] = []
    # stop_token_ids: list[int] = [128009, 128001]
    # repetition_penalty: float = 1.1
    n: int = 1
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _resolve_stop_tokens(self):
        from tuning.utils.utils import get_stop_tokens
        if not self.stop:
            self.stop = get_stop_tokens()
        return self

if __name__ == "__main__":
    print({**VLLMSamplingParamsConfig().model_dump()})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vllm_sampling_params_global.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tuning/inference/config_inference.py tests/test_vllm_sampling_params_global.py
git commit -m "Add optional seed field to VLLMSamplingParamsConfig"
```

---

### Task 2: Add `seed` field to `PassAtKConfig`

**Files:**
- Modify: `tuning/training/config_training.py:133-150`
- Create: `tests/test_seed_wiring.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_seed_wiring.py`:

```python
# ABOUTME: Tests for seed wiring — config seed fields and data module side effects.
# ABOUTME: Verifies PassAtKConfig.seed, and that data modules do not reseed global random on import.

import sys
from types import ModuleType


# Stub unsloth before importing config (crashes without GPU)
if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub


def test_passk_config_seed_defaults_none():
    from tuning.training.config_training import PassAtKConfig
    assert PassAtKConfig().seed is None


def test_passk_config_seed_set():
    from tuning.training.config_training import PassAtKConfig
    assert PassAtKConfig(seed=7).seed == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_seed_wiring.py -v`
Expected: FAIL for both new tests with validation error ("Extra inputs are not permitted" or "seed").

- [ ] **Step 3: Add `seed` field**

In `tuning/training/config_training.py`, modify the `PassAtKConfig` class (currently lines 133-150) to add `seed` after `initial_global_step`:

```python
class PassAtKConfig(BaseModel):
    """Configuration for generation-based evaluation callback."""
    target_pass_at_k: list[float] = [0.8]  # Target pass@k score to stop training (0.0 to 1.0)
    early_tuples: list[tuple[int, float]] | None = None  # Each tuple: (patience, min_increase)
    temperature: float = 0.5  # Sampling temperature for generation
    max_tokens: int = 4096  # Maximum tokens to generate per response
    enabled: bool = True  # Whether to enable the callback
    use_persistent_vllm: bool = True  # Keep vLLM engine alive between evals (saves cold-start time)
    vllm_gpu_memory_utilization: float = 0.4  # GPU memory fraction for vLLM (conservative for coexistence with training)
    num_inference_gpus: int = 1  # Number of GPUs for data-parallel vLLM inference (>1 forces ephemeral mode)
    max_checkpoint_gap: int | None = None  # Save a fallback checkpoint if no checkpoint for this many data points
    initial_global_step: int = 0  # Step offset for W&B logging continuity across chained runs
    seed: Optional[int] = None  # Seed for vLLM pass@k generation; None = stochastic (vLLM default)

    def __str__(self):
        lines = [f"[{self.__class__.__name__}]"]
        for name, value in self:
            lines.append(f"  {name}={value}")
        return "\n".join(lines)
```

(`Optional` is already imported at the top of the file.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_seed_wiring.py -v`
Expected: both new tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/config_training.py tests/test_seed_wiring.py
git commit -m "Add optional seed field to PassAtKConfig"
```

---

### Task 3: Make `TrainingArgumentsConfig.to_hf_args()` use `self.seed`

**Files:**
- Modify: `tuning/training/config_training.py:36-72`
- Test: `tests/test_grpo_config.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_grpo_config.py`:

```python
def test_training_arguments_config_seed_default():
    from tuning.training.config_training import TrainingArgumentsConfig
    d = TrainingArgumentsConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 42


def test_training_arguments_config_seed_override():
    from tuning.training.config_training import TrainingArgumentsConfig
    config = TrainingArgumentsConfig()
    config.seed = 7
    d = config.to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 7


def test_dpo_config_inherits_seed_override():
    from tuning.training.config_training import DPOTrainingConfig
    config = DPOTrainingConfig()
    config.seed = 13
    d = config.to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 13


def test_grpo_config_inherits_seed_override():
    from tuning.training.config_training import GRPOTrainingConfig
    config = GRPOTrainingConfig()
    config.seed = 99
    d = config.to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grpo_config.py -v`
Expected: the three `_override` tests FAIL with `42 != 7/13/99` (seed is hardcoded). The `_default` test passes.

- [ ] **Step 3: Add `seed` field and use it in `to_hf_args`**

In `tuning/training/config_training.py`, modify `TrainingArgumentsConfig` (lines 36-72):

1. Add `seed: int = 42` as a new field (e.g., right after `eval_accumulation_steps`).
2. Change the hardcoded assignment at line 71 from `d["seed"] = 42` to `d["seed"] = self.seed`.

Resulting relevant portion of `TrainingArgumentsConfig`:

```python
class TrainingArgumentsConfig(BaseModel):
    # sft training parameters
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = EFFECTIVE_BATCH_SIZE // per_device_train_batch_size
    per_device_eval_batch_size: int = 2
    eval_strategy: str = "steps"
    eval_steps: float = 4
    logging_steps: int = 1
    do_eval: bool = True
    warmup_ratio: int = 0.1
    num_train_epochs: int = 2
    learning_rate: float = 5e-5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    report_to: list[str] = ["wandb"]
    save_strategy: str = "no"
    save_steps: int = 4
    save_total_limit: int = 1
    load_best_model_at_end: bool = False
    dataloader_drop_last: bool = False
    dataloader_num_workers: int = 2
    eval_accumulation_steps: int = 1
    seed: int = 42

    def to_hf_args(self, output_dir: str) -> dict:
        """Return kwargs for TrainingArguments/DPOConfig constructor."""
        import torch
        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        d = self.model_dump()
        d.pop("beta", None)
        d["output_dir"] = output_dir
        d["fp16"] = not bf16_supported
        d["bf16"] = bf16_supported
        d["seed"] = self.seed
        return d
```

(Note: `seed` is already in `d` via `model_dump()` after this change, but the explicit `d["seed"] = self.seed` line is retained for clarity and symmetry with the other `d[...] = ...` assignments.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grpo_config.py -v`
Expected: all tests in the file PASS.

- [ ] **Step 5: Commit**

```bash
git add tuning/training/config_training.py tests/test_grpo_config.py
git commit -m "Thread seed through TrainingArgumentsConfig.to_hf_args"
```

---

### Task 4: Thread seed through `PassAtKStoppingCallback` into `VLLMSamplingParamsConfig`

**Files:**
- Modify: `tuning/training/passk_callback.py:45-114` (subprocess worker signature + body), `passk_callback.py:132-202` (callback init), `passk_callback.py:279-310` (persistent/ephemeral path), `passk_callback.py:386-407` (data-parallel subprocess spawn)
- Test: `tests/test_seed_wiring.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_seed_wiring.py`:

```python
def test_passk_callback_reads_seed_from_config(monkeypatch):
    """PassAtKStoppingCallback stores config.seed as self.seed."""
    # We cannot fully instantiate the callback (needs tokenizer + model), so
    # we mimic the attribute assignment the __init__ performs and assert
    # that a callback constructed from a PassAtKConfig with seed=13 would
    # end up with self.seed == 13. This is a contract test on the field.
    from tuning.training.config_training import PassAtKConfig
    config = PassAtKConfig(seed=13)
    # The getattr fallback mirrors the callback init pattern used for other
    # optional fields (lora_max_rank, max_checkpoint_gap, initial_global_step).
    assert getattr(config, "seed", None) == 13


def test_data_parallel_worker_signature_accepts_seed():
    """The subprocess worker must accept a seed kwarg so the parent can pass it."""
    import inspect
    from tuning.training.passk_callback import _data_parallel_worker
    sig = inspect.signature(_data_parallel_worker)
    assert "seed" in sig.parameters
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_seed_wiring.py::test_data_parallel_worker_signature_accepts_seed -v`
Expected: FAIL (seed not in worker signature).

Run: `pytest tests/test_seed_wiring.py::test_passk_callback_reads_seed_from_config -v`
Expected: PASS (Task 2 already added `seed` to `PassAtKConfig`), but kept as a guard against regression.

- [ ] **Step 3: Add `seed` kwarg to `_data_parallel_worker`**

In `tuning/training/passk_callback.py`, change the function signature at line 45-48 to add `seed=None`:

```python
def _data_parallel_worker(worker_id, cuda_device, messages_chunk, base_model_hf, adapter_path,
                          n_samples, temperature, max_tokens, chat_template,
                          lora_max_rank, gpu_memory_utilization, result_queue,
                          stop_tokens=None, seed=None):
```

And in the body (currently lines 75-82), change the `VLLMSamplingParamsConfig` construction to include `seed`:

```python
        from tuning.inference.config_inference import VLLMSamplingParamsConfig
        inference_config = VLLMSamplingParamsConfig(
            n=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
```

- [ ] **Step 4: Store `self.seed` in callback `__init__`**

Still in `passk_callback.py`, in `PassAtKStoppingCallback.__init__`, add a line near the other `getattr(config, ...)` reads (around line 153, after `self.max_checkpoint_gap = getattr(config, "max_checkpoint_gap", None)`):

```python
        self.seed = getattr(config, "seed", None)
```

- [ ] **Step 5: Pass `self.seed` in `_run_vllm_inference`**

In `passk_callback.py` `_run_vllm_inference` (currently lines 279-286), change:

```python
    def _run_vllm_inference(self, llm, eval_strategy: EvalStrategy, adapter_path: str = None) -> List[Dict]:
        """Run inference on a vLLM engine, optionally with a LoRA adapter."""
        inference_config = VLLMSamplingParamsConfig(
            n=eval_strategy.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        sampling_params = SamplingParams(**inference_config.model_dump())
```

- [ ] **Step 6: Pass `self.seed` to the data-parallel worker**

In `passk_callback.py` `_run_data_parallel_inference` (currently lines 394-406), pass `self.seed` as a keyword at the end of the args tuple. The worker signature was updated in Step 3 to accept `seed=None`; the parent should use the existing `args=(...)` tuple style. Change:

```python
        for i in range(actual_num_workers):
            p = ctx.Process(
                target=_data_parallel_worker,
                args=(
                    i, available_gpus[i], message_chunks[i], self.base_model_hf,
                    adapter_path, eval_strategy.n_samples, self.temperature, self.max_tokens,
                    self._chat_template, self.lora_max_rank,
                    self.vllm_gpu_memory_utilization, result_queue,
                    stop_tokens,
                    self.seed,  # added: passed as positional since ctx.Process uses args tuple
                ),
            )
            p.start()
            processes.append(p)
```

- [ ] **Step 7: Run tests to verify everything passes**

Run: `pytest tests/test_seed_wiring.py tests/test_vllm_sampling_params_global.py -v`
Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add tuning/training/passk_callback.py tests/test_seed_wiring.py
git commit -m "Thread seed through PassAtKStoppingCallback into vLLM SamplingParams"
```

---

### Task 5: Remove module-level `random.seed(42)` from data files

**Files:**
- Modify: `tuning/data/utils.py:1-4`
- Modify: `tuning/data/hf_dataset.py:10-12`
- Modify: `tuning/data/test_dataset.py:4-7`
- Test: `tests/test_seed_wiring.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_seed_wiring.py`:

```python
def test_data_modules_do_not_reseed_on_import():
    """Importing data modules must not mutate the global random state.

    The pipeline is the authoritative place to call random.seed(args.seed).
    Any module-level random.seed(42) would overwrite that if imports happened
    after the pipeline seeded.
    """
    import random
    import importlib

    random.seed(999)
    _ = random.random()
    expected_next = random.random()

    # Re-seed to reset the stream before importing
    random.seed(999)
    _ = random.random()

    # Force re-import of data modules
    for mod_name in ("tuning.data.utils", "tuning.data.hf_dataset", "tuning.data.test_dataset"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
        else:
            importlib.import_module(mod_name)

    # Global random must be untouched: the next call should produce expected_next
    assert random.random() == expected_next
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_seed_wiring.py::test_data_modules_do_not_reseed_on_import -v`
Expected: FAIL (module-level `random.seed(42)` overrides the stream).

- [ ] **Step 3: Remove the three module-level `random.seed(42)` calls**

Edit `tuning/data/utils.py` — delete line 4:

Before:
```python
from datasets import DatasetDict
import random

random.seed(42)


def get_random_train_subset(dataset: DatasetDict, train_size: int,
```

After:
```python
from datasets import DatasetDict
import random


def get_random_train_subset(dataset: DatasetDict, train_size: int,
```

Edit `tuning/data/hf_dataset.py` — delete line 12 and the blank line above it:

Before:
```python
from tuning.config import DATASETS_DIR


random.seed(42)

logger = logging.getLogger(__name__)
```

After:
```python
from tuning.config import DATASETS_DIR


logger = logging.getLogger(__name__)
```

Edit `tuning/data/test_dataset.py` — delete line 7:

Before:
```python
import random
import json

random.seed(42)
RESAMPLE = False
```

After:
```python
import random
import json

RESAMPLE = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_seed_wiring.py::test_data_modules_do_not_reseed_on_import -v`
Expected: PASS.

- [ ] **Step 5: Run existing data tests to check for regressions**

Run: `pytest tests/test_data_utils.py -v`
Expected: PASS (data subsetting still works; the pipeline is responsible for seeding).

- [ ] **Step 6: Commit**

```bash
git add tuning/data/utils.py tuning/data/hf_dataset.py tuning/data/test_dataset.py tests/test_seed_wiring.py
git commit -m "Remove module-level random.seed(42) from data modules"
```

---

### Task 6: Add `--eval_seed` CLI arg and wire seeds in `unified_early_pipeline.py`

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py:72-221` (arg parser), `343-415` (run_sft), `494-608` (run_dpo), `624-743` (run_grpo)
- Test: `tests/test_unified_early_pipeline.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_unified_early_pipeline.py`:

```python
class TestSeedArgs:
    def test_seed_default(self):
        args = _parse_args(["--model", "llama3-1B", "--wandb-project", "test"])
        assert args.seed == 42

    def test_seed_override(self):
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--seed", "7",
        ])
        assert args.seed == 7

    def test_eval_seed_default_none(self):
        args = _parse_args(["--model", "llama3-1B", "--wandb-project", "test"])
        assert args.eval_seed is None

    def test_eval_seed_override(self):
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--eval_seed", "99",
        ])
        assert args.eval_seed == 99


class TestEffectiveEvalSeed:
    def test_effective_eval_seed_falls_back_to_seed(self):
        from tuning.training.unified_early_pipeline import effective_eval_seed
        assert effective_eval_seed(seed=42, eval_seed=None) == 42

    def test_effective_eval_seed_override_wins(self):
        from tuning.training.unified_early_pipeline import effective_eval_seed
        assert effective_eval_seed(seed=42, eval_seed=7) == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_unified_early_pipeline.py -v`
Expected: the four new tests FAIL — `--eval_seed` not parsed; `effective_eval_seed` not defined.

- [ ] **Step 3: Add `--eval_seed` argument**

In `tuning/training/unified_early_pipeline.py`, in `_parse_args` right below the existing `--seed` line (currently line 110), add:

```python
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=None,
                        help="Override seed for pass@k eval generation. When None, uses --seed.")
```

- [ ] **Step 4: Add `effective_eval_seed` helper**

In `tuning/training/unified_early_pipeline.py`, add this helper function near the top-level helpers (e.g., right after `parse_early_tuple`, around line 69):

```python
def effective_eval_seed(seed: int, eval_seed: int | None) -> int:
    """Return eval_seed when set, else the master seed."""
    return eval_seed if eval_seed is not None else seed
```

- [ ] **Step 5: Wire `args.seed` and effective eval seed into `run_sft`**

In `tuning/training/unified_early_pipeline.py`, update `run_sft` (currently starting at line 343). Right after `set_chat_template(...)` at line 354, seed the global random and log the seeds. Then on the existing `lora_config` and `training_args` constructor calls, set their seeds. On the `passk_config` returned by `_build_eval_components`, set `seed`. Finally add the seed fields into the `wandb.init(config=...)` dict.

```python
def run_sft(args):
    """Run SFT stage, returning a list of metadata file paths written by callbacks."""
    import random
    import wandb
    from tuning.config import HF_MODEL_MAP, set_chat_template
    from tuning.training.config_training import (
        DatasetConfig, SFTRunConfig, ModelLoadConfig, LoraConfig,
        TrainingArgumentsConfig,
    )
    from tuning.training.sft_training import train_model_sft
    from tuning.utils.gpu import cleanup_gpu

    random.seed(args.seed)
    set_chat_template(args.model, simple=args.simple_template)
    gpu_util = MODEL_TO_GPU_1[args.model]

    sft_size = args.sft_data_size if args.sft_data_size is not None else args.train_size
    dataset_config = DatasetConfig(
        dataset=args.dataset,
        dataset_type="sft",
        train_size=sft_size,
    )
    run_config = SFTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[args.model],
        model_name=args.model,
        do_training=True,
        do_inference=False,
        do_evaluation=False,
        task_name=args.task_name,
    )
    lora_config = LoraConfig()
    lora_config.random_state = args.seed
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length
    training_args = TrainingArgumentsConfig()
    training_args.num_train_epochs = args.sft_num_epochs
    training_args.eval_steps = args.sft_eval_steps
    training_args.per_device_train_batch_size = args.sft_batch_size
    training_args.gradient_accumulation_steps = args.sft_grad_accum
    training_args.warmup_ratio = args.sft_warmup_ratio
    training_args.learning_rate = args.sft_learning_rate
    training_args.seed = args.seed

    eval_seed = effective_eval_seed(args.seed, args.eval_seed)
    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "sft", gpu_util)
    if passk_config is not None:
        passk_config.seed = eval_seed
    ppl_config = _sft_ppl_config(args)
    tags = _sft_tags(passk_config, ppl_config, primary_eval)

    with wandb.init(
        name=run_config.model_name,
        project=args.wandb_project,
        job_type="sft",
        tags=tags,
        config={"stage": "sft", "seed": args.seed, "eval_seed": eval_seed},
    ):
        model, tokenizer, trainer, callbacks = train_model_sft(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            passk_config=passk_config,
            perplexity_config=ppl_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
        )

    metadata_paths = [
        cb.metadata_path
        for cb in callbacks
        if getattr(cb, "metadata_path", None)
    ]

    del model, tokenizer, trainer, callbacks
    cleanup_gpu()
    print(subprocess.check_output("nvidia-smi").decode())
    print_metadata_paths(metadata_paths)
    return metadata_paths
```

- [ ] **Step 6: Wire `args.seed` and effective eval seed into `run_dpo`**

In `run_dpo` (currently starting at line 494), apply the same pattern. After `import wandb` and the other config imports (around line 516), add `import random` and `random.seed(args.seed)` right after `set_chat_template(...)` at line 524. After constructing `lora_config` and `training_args`, set their seeds. On the `passk_config`, set its `seed`. Include seeds in the `wandb.init(config=...)` dict.

Concretely, replace the block from `set_chat_template(...)` through the end of `wandb.init(...)` with:

```python
    import random
    random.seed(args.seed)
    set_chat_template(args.model, simple=args.simple_template)
    gpu_util = MODEL_TO_GPU_2[args.model]
    model_name = Path(checkpoint["checkpoint_path"]).name

    dataset_config = DatasetConfig(
        dataset=args.dataset,
        dataset_type="pt",
        train_size=dpo_size,
    )
    sft_run_config = SFTRunConfig(
        dataset_config=DatasetConfig(
            dataset=args.dataset,
            dataset_type="sft",
            train_size=checkpoint["data_points_seen"],
            dynamic_path=model_name,
        ),
        model_name=args.model,
        model_name_hf=HF_MODEL_MAP[args.model],
        task_name=args.task_name,
    )
    run_config = PTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[args.model],
        model_name=args.model,
        sft_run_config=sft_run_config,
        task_name=args.task_name,
        pft_method="dpo",
        do_training=True,
    )
    lora_config = LoraConfig()
    lora_config.random_state = args.seed
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length
    training_args = DPOTrainingConfig()
    training_args.num_train_epochs = args.dpo_num_epochs
    training_args.eval_steps = args.dpo_eval_steps
    training_args.per_device_train_batch_size = args.dpo_batch_size
    training_args.gradient_accumulation_steps = args.dpo_grad_accum
    training_args.learning_rate = args.dpo_learning_rate
    training_args.seed = args.seed

    eval_seed = effective_eval_seed(args.seed, args.eval_seed)
    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "dpo", gpu_util)
    if passk_config is not None:
        passk_config.seed = eval_seed
    ppl_config = _dpo_ppl_config(args)

    initial_step = checkpoint.get("global_step", 0)
    if passk_config is not None:
        passk_config.initial_global_step = initial_step
    if ppl_config is not None:
        ppl_config.initial_global_step = initial_step

    perplexity_test_dataset = None
    if ppl_config is not None:
        from tuning.data.train_dataset import get_train_dataset
        sft_dataset = get_train_dataset(sft_run_config)
        perplexity_test_dataset = sft_dataset["test"]

    tags = ["dpo", str(checkpoint["threshold_value"]), str(checkpoint["data_points_seen"])]
    if primary_eval is not None:
        tags.append(primary_eval.id)
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        tags.append(f"p{k_val}")
    if ppl_config is not None:
        tags.append("ppl")

    with wandb.init(
        name=run_config.model_name,
        project=args.wandb_project,
        job_type="dpo",
        tags=tags,
        config={"stage": "dpo", "seed": args.seed, "eval_seed": eval_seed},
        settings=wandb.Settings(init_timeout=300)
    ):
        train_model_dpo(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            passk_config=passk_config,
            perplexity_config=ppl_config,
            perplexity_test_dataset=perplexity_test_dataset,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
            initial_global_step=checkpoint.get("global_step")
        )

    mark_completed(metadata_file, checkpoint["checkpoint_path"])
```

- [ ] **Step 7: Wire `args.seed` and effective eval seed into `run_grpo`**

In `run_grpo` (currently starting at line 624), apply the same pattern. After `import wandb` and config imports (around line 646), add `import random` and `random.seed(args.seed)` right after `set_chat_template(...)` at line 654. After constructing `lora_config` and `training_args`, set their seeds. On `passk_config`, set `seed`. Include seeds in `wandb.init(config=...)` dict.

Replace the block from `set_chat_template(...)` through `wandb.init(...)` with:

```python
    import random
    random.seed(args.seed)
    set_chat_template(args.model, simple=args.simple_template)
    gpu_util = MODEL_TO_GPU_3[args.model]
    model_name = Path(checkpoint["checkpoint_path"]).name

    dataset_config = DatasetConfig(
        dataset=args.dataset,
        dataset_type="rlvr",
        train_size=grpo_size,
    )
    sft_run_config = SFTRunConfig(
        dataset_config=DatasetConfig(
            dataset=args.dataset,
            dataset_type="sft",
            train_size=checkpoint["data_points_seen"],
            dynamic_path=model_name,
        ),
        model_name=args.model,
        model_name_hf=HF_MODEL_MAP[args.model],
        task_name=args.task_name,
    )
    run_config = PTRunConfig(
        dataset_config=dataset_config,
        model_name_hf=HF_MODEL_MAP[args.model],
        model_name=args.model,
        sft_run_config=sft_run_config,
        task_name=args.task_name,
        pft_method="grpo",
        do_training=True,
        simple_template=args.simple_template,
    )
    lora_config = LoraConfig()
    if args.grpo_lora_target_modules is not None:
        lora_config.target_modules = args.grpo_lora_target_modules
    lora_config.random_state = args.seed
    model_load_config = ModelLoadConfig()
    model_load_config.max_seq_length = args.max_seq_length
    training_args = GRPOTrainingConfig()
    training_args.num_train_epochs = args.grpo_num_epochs
    training_args.eval_steps = args.grpo_eval_steps
    training_args.per_device_train_batch_size = args.grpo_batch_size
    training_args.gradient_accumulation_steps = args.grpo_grad_accum
    training_args.num_generations = args.grpo_num_generations
    training_args.max_completion_length = args.grpo_max_completion_length
    training_args.beta = args.grpo_beta
    training_args.temperature = args.grpo_temperature
    training_args.learning_rate = args.grpo_learning_rate
    training_args.loss_type = args.grpo_loss_type
    scale_rewards = args.grpo_scale_rewards
    training_args.scale_rewards = False if scale_rewards == "false" else scale_rewards
    training_args.vllm_gpu_memory_utilization = gpu_util
    training_args.seed = args.seed

    eval_seed = effective_eval_seed(args.seed, args.eval_seed)
    passk_config, primary_eval, monitor_evals = _build_eval_components(args, "grpo", gpu_util)
    if passk_config is not None:
        passk_config.seed = eval_seed
    reward_funcs = _build_reward_funcs(args)

    initial_step = checkpoint.get("global_step", 0)
    if passk_config is not None:
        passk_config.initial_global_step = initial_step

    tags = ["grpo", str(checkpoint["threshold_value"]), str(checkpoint["data_points_seen"])]
    if primary_eval is not None:
        tags.append(primary_eval.id)
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        tags.append(f"p{k_val}")


    with wandb.init(
        name=run_config.model_name,
        project=args.wandb_project,
        job_type="grpo",
        tags=tags,
        config={
            "stage": "grpo",
            "seed": args.seed,
            "eval_seed": eval_seed,
        },
        settings=wandb.Settings(init_timeout=300)
    ):
        train_model_grpo(
            run_config=run_config,
            lora_config=lora_config,
            model_load_config=model_load_config,
            training_args=training_args,
            reward_funcs=reward_funcs,
            passk_config=passk_config,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals,
            initial_global_step=checkpoint.get("global_step"),
            lora_layers_fraction=args.grpo_lora_layers_fraction,
        )

    mark_completed(metadata_file, checkpoint["checkpoint_path"])
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/test_unified_early_pipeline.py tests/test_seed_wiring.py tests/test_grpo_config.py tests/test_vllm_sampling_params_global.py -v`
Expected: all tests PASS.

- [ ] **Step 9: Sanity check the pipeline can still be invoked with defaults**

Run: `python -c "from tuning.training.unified_early_pipeline import _parse_args; a = _parse_args(['--model','llama3-1B','--wandb-project','x']); print('seed=', a.seed, 'eval_seed=', a.eval_seed)"`
Expected output: `seed= 42 eval_seed= None`

- [ ] **Step 10: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Wire --seed and --eval_seed through unified early pipeline"
```

---
