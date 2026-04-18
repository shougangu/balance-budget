# Seed Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread the `--seed` CLI arg through all currently-hardcoded RNG in the unified early pipeline (training, LoRA init, data sampling, pass@k eval generation), add a `--eval_seed` override, and log both to W&B — all while preserving today's default behavior (42 everywhere).

**Architecture:** Two globals in `tuning/config.py` — `DEFAULT_SEED` and `DEFAULT_EVAL_SEED` — mirror the existing `DEFAULT_CHAT_TEMPLATE` pattern. The pipeline calls `_init_seeds(args)` once per stage, which calls `set_seed()`, `set_eval_seed()`, and `random.seed()`. All downstream consumers resolve to these globals at runtime:

- `TrainingArgumentsConfig.to_hf_args()` reads `tuning.config.DEFAULT_SEED` (replacing hardcoded 42)
- `VLLMSamplingParamsConfig` resolves its `seed` field from `get_eval_seed()` via a `model_validator` (same pattern as `_resolve_stop_tokens` reads `DEFAULT_CHAT_TEMPLATE`)
- `LoraConfig` resolves `random_state` from `tuning.config.DEFAULT_SEED` via a `model_validator` (same pattern — no explicit per-object assignment needed in the pipeline)
- Module-level `random.seed(42)` calls in data files are removed; `_init_seeds` calls `random.seed()` once before any data loading
- Subprocess data-parallel workers: parent reads `tuning.config.get_eval_seed()` directly at spawn time and passes as an arg (globals aren't inherited across `spawn` context — same reason stop_tokens are passed explicitly)

**Tech Stack:** Python, Pydantic, TRL (SFTConfig/DPOConfig/GRPOConfig), unsloth (FastLanguageModel.get_peft_model), vLLM (SamplingParams), pytest

**Spec:** `docs/superpowers/specs/2026-04-15-seed-wiring-design.md`

**Spec deviations:** The spec describes seed passing via explicit callback parameters ("the callback receives the effective eval seed and injects it"). This plan uses global resolution via `model_validator` instead, matching the existing `DEFAULT_CHAT_TEMPLATE` → `get_stop_tokens()` pattern. This eliminates per-object seed wiring in the pipeline — `LoraConfig()`, `TrainingArgumentsConfig().to_hf_args()`, and `VLLMSamplingParamsConfig()` all resolve from globals automatically. Only the subprocess path needs explicit passing (same as `stop_tokens` today).

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `tuning/config.py` | Modify | Add `DEFAULT_SEED`, `DEFAULT_EVAL_SEED`, `set_seed()`, `set_eval_seed()`, `get_eval_seed()` |
| `tuning/inference/config_inference.py` | Modify | Add `seed: Optional[int] = None` field; model_validator resolves from `get_eval_seed()` when None |
| `tuning/training/config_training.py` | Modify | `to_hf_args()` reads `tuning.config.DEFAULT_SEED` instead of hardcoded 42; `LoraConfig` resolves `random_state` from global via model_validator |
| `tuning/training/passk_callback.py` | Modify | Subprocess worker accepts `seed` kwarg; parent reads `tuning.config.get_eval_seed()` directly at spawn time |
| `tuning/data/utils.py` | Modify | Remove module-level `random.seed(42)` |
| `tuning/data/hf_dataset.py` | Modify | Remove module-level `random.seed(42)` |
| `tuning/data/test_dataset.py` | Modify | Remove module-level `random.seed(42)` |
| `tuning/training/unified_early_pipeline.py` | Modify | Add `--eval_seed` arg; add `_init_seeds(args)` helper; call it once per stage; log seeds to W&B |
| `tests/test_seed_wiring.py` | Create | Tests for global seed config, VLLMSamplingParamsConfig resolver, data-module side-effects, worker signature |
| `tests/test_vllm_sampling_params_global.py` | Modify | Add seed resolver tests |
| `tests/test_grpo_config.py` | Modify | Add test: `to_hf_args()` follows `set_seed()` |
| `tests/test_unified_early_pipeline.py` | Modify | Add `--eval_seed` parsing tests; effective-eval-seed helper tests |

---

### Task 1: Add seed globals to `tuning/config.py`

**Files:**
- Modify: `tuning/config.py`
- Create: `tests/test_seed_wiring.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_seed_wiring.py`:

```python
# ABOUTME: Tests for seed wiring — global seed config, VLLMSamplingParamsConfig resolver,
# ABOUTME: data module side effects, and subprocess worker signature.

import sys
from types import ModuleType

# Stub unsloth before importing config (crashes without GPU)
if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub

import pytest
import tuning.config


@pytest.fixture(autouse=True)
def restore_seed_globals():
    """Reset seed globals after each test."""
    orig_seed = tuning.config.DEFAULT_SEED
    orig_eval = tuning.config.DEFAULT_EVAL_SEED
    yield
    tuning.config.DEFAULT_SEED = orig_seed
    tuning.config.DEFAULT_EVAL_SEED = orig_eval


def test_default_seed_is_42():
    assert tuning.config.DEFAULT_SEED == 42


def test_default_eval_seed_is_none():
    assert tuning.config.DEFAULT_EVAL_SEED is None


def test_set_seed_updates_global():
    tuning.config.set_seed(99)
    assert tuning.config.DEFAULT_SEED == 99


def test_set_eval_seed_updates_global():
    tuning.config.set_eval_seed(13)
    assert tuning.config.DEFAULT_EVAL_SEED == 13


def test_get_eval_seed_returns_eval_seed_when_set():
    tuning.config.set_seed(42)
    tuning.config.set_eval_seed(99)
    assert tuning.config.get_eval_seed() == 99


def test_get_eval_seed_falls_back_to_default_seed():
    tuning.config.set_seed(7)
    tuning.config.DEFAULT_EVAL_SEED = None
    assert tuning.config.get_eval_seed() == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_seed_wiring.py -v`
Expected: FAIL with `AttributeError: module 'tuning.config' has no attribute 'DEFAULT_SEED'`.

- [ ] **Step 3: Add seed globals to `tuning/config.py`**

At the end of `tuning/config.py` (after the `set_chat_template` function), add:

```python
DEFAULT_SEED = 42
DEFAULT_EVAL_SEED = None  # When None, eval uses DEFAULT_SEED


def set_seed(seed: int):
    """Set the global training seed. Call once at pipeline start, like set_chat_template()."""
    global DEFAULT_SEED
    DEFAULT_SEED = seed


def set_eval_seed(seed: int):
    """Set the global eval seed (pass@k generation). Call once at pipeline start."""
    global DEFAULT_EVAL_SEED
    DEFAULT_EVAL_SEED = seed


def get_eval_seed() -> int:
    """Return the eval seed: DEFAULT_EVAL_SEED if set, else DEFAULT_SEED."""
    return DEFAULT_EVAL_SEED if DEFAULT_EVAL_SEED is not None else DEFAULT_SEED
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_seed_wiring.py -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tuning/config.py tests/test_seed_wiring.py
git commit -m "Add DEFAULT_SEED and DEFAULT_EVAL_SEED globals to tuning/config.py"
```

---

### Task 2: Add `seed` field to `VLLMSamplingParamsConfig` with global resolver

**Files:**
- Modify: `tuning/inference/config_inference.py`
- Test: `tests/test_vllm_sampling_params_global.py`

The `seed` field defaults to `None`. The existing `_resolve_stop_tokens` validator is expanded into `_resolve_defaults`, which also resolves `seed` from `tuning.config.get_eval_seed()` when None — same pattern, same validator. When the subprocess passes `seed` explicitly, the validator sees a non-None value and skips resolution.

**Behavioral change:** Previously vLLM evals ran without a seed (stochastic). After this task, they default to `seed=42` via the global resolver. This is intentional — it makes evals reproducible by default, matching training's existing `seed=42` default. The spec describes the field as stochastic-when-None, but the resolver ensures None is never passed to vLLM.

- [ ] **Step 1: Write failing tests**

Add a `restore_seed_globals` fixture and new tests to `tests/test_vllm_sampling_params_global.py`:

```python
@pytest.fixture(autouse=True)
def restore_seed_globals():
    orig_seed = tuning.config.DEFAULT_SEED
    orig_eval = tuning.config.DEFAULT_EVAL_SEED
    yield
    tuning.config.DEFAULT_SEED = orig_seed
    tuning.config.DEFAULT_EVAL_SEED = orig_eval


def test_vllm_sampling_params_seed_resolves_from_global():
    """When seed is not set, it resolves from the global eval seed."""
    tuning.config.set_seed(7)
    tuning.config.DEFAULT_EVAL_SEED = None  # falls back to DEFAULT_SEED
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert config.seed == 7


def test_vllm_sampling_params_seed_resolves_eval_seed_override():
    """When DEFAULT_EVAL_SEED is set, it takes priority."""
    tuning.config.set_seed(42)
    tuning.config.set_eval_seed(99)
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig()
    assert config.seed == 99


def test_vllm_sampling_params_seed_explicit_overrides_global():
    """When seed is passed explicitly, global is ignored."""
    tuning.config.set_seed(42)
    tuning.config.set_eval_seed(99)
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig(seed=7)
    assert config.seed == 7


def test_vllm_sampling_params_seed_roundtrips_through_model_dump():
    tuning.config.set_seed(42)
    from tuning.inference.config_inference import VLLMSamplingParamsConfig

    config = VLLMSamplingParamsConfig(seed=7)
    assert config.model_dump()["seed"] == 7
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vllm_sampling_params_global.py -v`
Expected: FAIL — `VLLMSamplingParamsConfig` has no `seed` attribute.

- [ ] **Step 3: Add `seed` field and expand resolver**

Edit `tuning/inference/config_inference.py`:

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
    def _resolve_defaults(self):
        from tuning.utils.utils import get_stop_tokens
        import tuning.config
        if not self.stop:
            self.stop = get_stop_tokens()
        if self.seed is None:
            self.seed = tuning.config.get_eval_seed()
        return self

if __name__ == "__main__":
    print({**VLLMSamplingParamsConfig().model_dump()})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vllm_sampling_params_global.py -v`
Expected: all tests PASS (including existing stop-token tests).

- [ ] **Step 5: Commit**

```bash
git add tuning/inference/config_inference.py tests/test_vllm_sampling_params_global.py
git commit -m "Add seed field to VLLMSamplingParamsConfig with global resolver"
```

---

### Task 3: Make `TrainingArgumentsConfig.to_hf_args()` and `LoraConfig.random_state` read `DEFAULT_SEED`

**Files:**
- Modify: `tuning/training/config_training.py:24-34` (LoraConfig), `config_training.py:62-72` (to_hf_args)
- Test: `tests/test_grpo_config.py`

Both `to_hf_args()` and `LoraConfig.random_state` currently hardcode 42. Both are changed to resolve from `tuning.config.DEFAULT_SEED` at runtime:
- `to_hf_args()` reads the global directly (called at runtime, so it always gets the current value)
- `LoraConfig` uses a `model_validator` to resolve `random_state` from the global (same pattern as `VLLMSamplingParamsConfig` in Task 2)

- [ ] **Step 1: Write failing tests**

Add `import pytest` and `import tuning.config` to the top of `tests/test_grpo_config.py`, plus a fixture and new tests:

```python
import pytest
import tuning.config


@pytest.fixture(autouse=True)
def restore_seed_globals():
    orig = tuning.config.DEFAULT_SEED
    yield
    tuning.config.DEFAULT_SEED = orig


def test_training_arguments_config_seed_uses_global_default():
    tuning.config.set_seed(42)
    from tuning.training.config_training import TrainingArgumentsConfig
    d = TrainingArgumentsConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 42


def test_training_arguments_config_seed_follows_set_seed():
    tuning.config.set_seed(7)
    from tuning.training.config_training import TrainingArgumentsConfig
    d = TrainingArgumentsConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 7


def test_dpo_config_seed_follows_set_seed():
    tuning.config.set_seed(13)
    from tuning.training.config_training import DPOTrainingConfig
    d = DPOTrainingConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 13


def test_grpo_config_seed_follows_set_seed():
    tuning.config.set_seed(99)
    from tuning.training.config_training import GRPOTrainingConfig
    d = GRPOTrainingConfig().to_hf_args(output_dir="/tmp/test")
    assert d["seed"] == 99


def test_lora_config_random_state_resolves_from_global():
    tuning.config.set_seed(7)
    from tuning.training.config_training import LoraConfig
    config = LoraConfig()
    assert config.random_state == 7


def test_lora_config_random_state_default_is_42():
    tuning.config.set_seed(42)
    from tuning.training.config_training import LoraConfig
    config = LoraConfig()
    assert config.random_state == 42


def test_lora_config_random_state_explicit_overrides_global():
    tuning.config.set_seed(7)
    from tuning.training.config_training import LoraConfig
    config = LoraConfig(random_state=99)
    assert config.random_state == 99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grpo_config.py -v`
Expected: `_follows_set_seed` tests FAIL with `7 != 42` / `13 != 42` / `99 != 42`; `lora_config_random_state_resolves_from_global` FAIL with `42 != 7`.

- [ ] **Step 3: Add `import tuning.config` and model_validator to `LoraConfig`**

In `tuning/training/config_training.py`, add `import tuning.config` and `from pydantic import model_validator` at the top of the file (after the existing imports).

Change `LoraConfig` to use a sentinel default and resolve from the global:

```python
class LoraConfig(BaseModel):
    r: int = 32
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",]
    lora_alpha: int = 32
    lora_dropout: int = 0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: Optional[int] = None
    use_rslora: bool = False
    loftq_config: dict = {}

    @model_validator(mode="after")
    def _resolve_defaults(self):
        if self.random_state is None:
            self.random_state = tuning.config.DEFAULT_SEED
        return self
```

- [ ] **Step 4: Change `to_hf_args` to read the global**

In the same file, change line 71 from:

```python
        d["seed"] = 42
```

to:

```python
        d["seed"] = tuning.config.DEFAULT_SEED
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_grpo_config.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tuning/training/config_training.py tests/test_grpo_config.py
git commit -m "TrainingArgumentsConfig.to_hf_args and LoraConfig.random_state resolve from DEFAULT_SEED"
```

---

### Task 4: Thread eval seed through subprocess data-parallel worker

**Files:**
- Modify: `tuning/training/passk_callback.py:45-48` (worker signature), `passk_callback.py:74-82` (worker body), `passk_callback.py:386-406` (data-parallel spawn)
- Test: `tests/test_seed_wiring.py`

The persistent/ephemeral `_run_vllm_inference` path needs **no changes** — `VLLMSamplingParamsConfig()` resolves its seed from `tuning.config.get_eval_seed()` via the model_validator (Task 2). No seed is stored on the callback class.

The subprocess path needs explicit seed passing because globals aren't inherited across `spawn` context. The parent reads `tuning.config.get_eval_seed()` directly at spawn time (same as it reads `get_stop_tokens()`) and passes it to the worker.

- [ ] **Step 1: Write failing test**

Append to `tests/test_seed_wiring.py`:

```python
def test_data_parallel_worker_signature_accepts_seed():
    """The subprocess worker must accept a seed kwarg so the parent can pass it."""
    import inspect
    from tuning.training.passk_callback import _data_parallel_worker
    sig = inspect.signature(_data_parallel_worker)
    assert "seed" in sig.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_seed_wiring.py::test_data_parallel_worker_signature_accepts_seed -v`
Expected: FAIL (seed not in worker signature).

- [ ] **Step 3: Add `seed` kwarg to `_data_parallel_worker`**

In `tuning/training/passk_callback.py`, change the function signature at lines 45-48 to add `seed=None`:

```python
def _data_parallel_worker(worker_id, cuda_device, messages_chunk, base_model_hf, adapter_path,
                          n_samples, temperature, max_tokens, chat_template,
                          lora_max_rank, gpu_memory_utilization, result_queue,
                          stop_tokens=None, seed=None):
```

And in the body (lines 74-82), pass `seed` and `stop` to `VLLMSamplingParamsConfig` explicitly (overriding the global resolvers, which won't work in the subprocess — same reason `stop_tokens` is already passed explicitly today):

```python
        from tuning.inference.config_inference import VLLMSamplingParamsConfig
        inference_config = VLLMSamplingParamsConfig(
            n=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_tokens or [],
            seed=seed,
        )
```

- [ ] **Step 4: Pass eval seed to the data-parallel worker**

In `_run_data_parallel_inference` (lines 386-406), add `import tuning.config` to the file-level imports (changing the existing `from tuning.config import MODELS_METADATA_DIR` to also import the module), then read the eval seed alongside stop_tokens and pass it to the worker:

```python
        # Compute stop tokens and eval seed here since subprocess won't have the globals set
        from tuning.utils.utils import get_stop_tokens
        stop_tokens = get_stop_tokens()
        eval_seed = tuning.config.get_eval_seed()

        for i in range(actual_num_workers):
            p = ctx.Process(
                target=_data_parallel_worker,
                args=(
                    i, available_gpus[i], message_chunks[i], self.base_model_hf,
                    adapter_path, eval_strategy.n_samples, self.temperature, self.max_tokens,
                    self._chat_template, self.lora_max_rank,
                    self.vllm_gpu_memory_utilization, result_queue,
                    stop_tokens,
                    eval_seed,
                ),
            )
            p.start()
            processes.append(p)
```

- [ ] **Step 5: Run tests to verify everything passes**

Run: `pytest tests/test_seed_wiring.py tests/test_vllm_sampling_params_global.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tuning/training/passk_callback.py tests/test_seed_wiring.py
git commit -m "Thread eval seed through subprocess data-parallel worker (like stop_tokens)"
```

---

### Task 5: Remove module-level `random.seed(42)` from data files

**Files:**
- Modify: `tuning/data/utils.py:4`
- Modify: `tuning/data/hf_dataset.py:12`
- Modify: `tuning/data/test_dataset.py:7`
- Test: `tests/test_seed_wiring.py`

After removal, data modules no longer seed global random on import. The pipeline's `_init_seeds(args)` (Task 6) calls `random.seed(args.seed)` once before any data loading, so all `random.sample()` / `random.shuffle()` calls in data modules use the CLI-provided seed.

- [ ] **Step 1: Write failing test**

Append to `tests/test_seed_wiring.py`:

```python
def test_data_modules_do_not_reseed_on_import():
    """Importing data modules must not mutate the global random state.

    The pipeline calls random.seed(args.seed) once via _init_seeds() before
    any data loading. Module-level random.seed(42) would overwrite that if
    imports happened after the pipeline seeded.
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

Edit `tuning/data/utils.py` — delete line 4 (`random.seed(42)`).

Edit `tuning/data/hf_dataset.py` — delete line 12 (`random.seed(42)`).

Edit `tuning/data/test_dataset.py` — delete line 7 (`random.seed(42)`).

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

### Task 6: Add `--eval_seed` CLI arg, `_init_seeds()` helper, and wire in pipeline

**Files:**
- Modify: `tuning/training/unified_early_pipeline.py`
- Test: `tests/test_unified_early_pipeline.py`

This is the integration task. A single `_init_seeds(args)` helper sets all global seed state. Each `run_*` function calls `_init_seeds(args)` at the top — no per-object seed wiring needed. `LoraConfig`, `TrainingArgumentsConfig.to_hf_args()`, and `VLLMSamplingParamsConfig` all resolve from the globals automatically via their model_validators.

- [ ] **Step 1: Write failing tests**

Append to `tests/test_unified_early_pipeline.py`:

```python
import tuning.config


@pytest.fixture(autouse=True)
def restore_seed_globals():
    orig_seed = tuning.config.DEFAULT_SEED
    orig_eval = tuning.config.DEFAULT_EVAL_SEED
    yield
    tuning.config.DEFAULT_SEED = orig_seed
    tuning.config.DEFAULT_EVAL_SEED = orig_eval


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


class TestInitSeeds:
    def test_init_seeds_sets_globals(self):
        from tuning.training.unified_early_pipeline import _init_seeds
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--seed", "7", "--eval_seed", "99",
        ])
        _init_seeds(args)
        assert tuning.config.DEFAULT_SEED == 7
        assert tuning.config.DEFAULT_EVAL_SEED == 99
        assert tuning.config.get_eval_seed() == 99

    def test_init_seeds_eval_seed_falls_back(self):
        from tuning.training.unified_early_pipeline import _init_seeds
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--seed", "7",
        ])
        _init_seeds(args)
        assert tuning.config.DEFAULT_SEED == 7
        assert tuning.config.get_eval_seed() == 7

    def test_init_seeds_seeds_random(self):
        import random
        from tuning.training.unified_early_pipeline import _init_seeds
        args = _parse_args([
            "--model", "llama3-1B", "--wandb-project", "test",
            "--seed", "123",
        ])
        _init_seeds(args)
        val1 = random.random()
        # Re-seed and verify same stream
        random.seed(123)
        assert random.random() == val1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_unified_early_pipeline.py::TestSeedArgs -v`
Expected: `--eval_seed` tests FAIL (arg not defined). `--seed` tests pass.

Run: `pytest tests/test_unified_early_pipeline.py::TestEffectiveEvalSeed -v`
Expected: FAIL (`effective_eval_seed` not defined).

Run: `pytest tests/test_unified_early_pipeline.py::TestInitSeeds -v`
Expected: FAIL (`_init_seeds` not defined).

- [ ] **Step 3: Add `--eval_seed` argument**

In `tuning/training/unified_early_pipeline.py`, right below the existing `--seed` line (line 110), add:

```python
    parser.add_argument("--eval_seed", type=int, default=None,
                        help="Override seed for pass@k eval generation. When None, uses --seed.")
```

- [ ] **Step 4: Add `effective_eval_seed` and `_init_seeds` helpers**

In `tuning/training/unified_early_pipeline.py`, add `import random` and `import tuning.config` to the file-level imports (lines 3-8). Then near the top-level helpers (after `parse_early_tuple`, around line 69), add:

```python
def effective_eval_seed(seed: int, eval_seed: int | None) -> int:
    """Return eval_seed when set, else the master seed."""
    return eval_seed if eval_seed is not None else seed


def _init_seeds(args):
    """Set global seed state from CLI args. Call once per stage, like set_chat_template().

    Sets tuning.config.DEFAULT_SEED, tuning.config.DEFAULT_EVAL_SEED,
    and seeds the Python random module for data loading.

    Note: after this call DEFAULT_EVAL_SEED is always an int (either
    args.eval_seed or args.seed), so get_eval_seed()'s None-fallback
    path is only exercised before init or in tests.
    """
    from tuning.config import set_seed, set_eval_seed
    set_seed(args.seed)
    set_eval_seed(effective_eval_seed(args.seed, args.eval_seed))
    random.seed(args.seed)
```

- [ ] **Step 5: Wire seeds in `run_sft`**

In `run_sft` (line 343), add `_init_seeds(args)` right before `set_chat_template(...)`:

```python
    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)
```

Update `wandb.init(config=...)`:

```python
        config={"stage": "sft", "seed": args.seed, "eval_seed": tuning.config.get_eval_seed()},
```

No per-object seed wiring needed — `LoraConfig()`, `TrainingArgumentsConfig().to_hf_args()`, and `VLLMSamplingParamsConfig()` all resolve from the globals set by `_init_seeds`.

- [ ] **Step 6: Wire seeds in `run_dpo`**

Same pattern in `run_dpo` (line 494). Add `_init_seeds(args)` before `set_chat_template(...)`:

```python
    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)
```

Update `wandb.init(config=...)`:

```python
        config={"stage": "dpo", "seed": args.seed, "eval_seed": tuning.config.get_eval_seed()},
```

- [ ] **Step 7: Wire seeds in `run_grpo`**

Same pattern in `run_grpo` (line 624). Add `_init_seeds(args)` before `set_chat_template(...)`:

```python
    _init_seeds(args)
    set_chat_template(args.model, simple=args.simple_template)
```

Update `wandb.init(config=...)`:

```python
        config={"stage": "grpo", "seed": args.seed, "eval_seed": tuning.config.get_eval_seed()},
```

- [ ] **Step 8: Run all tests**

Run: `pytest tests/test_unified_early_pipeline.py tests/test_seed_wiring.py tests/test_grpo_config.py tests/test_vllm_sampling_params_global.py -v`
Expected: all tests PASS.

- [ ] **Step 9: Sanity check the pipeline can still be invoked with defaults**

Run: `python -c "from tuning.training.unified_early_pipeline import _parse_args, _init_seeds; a = _parse_args(['--model','llama3-1B','--wandb-project','x']); _init_seeds(a); import tuning.config; print('seed=', a.seed, 'eval_seed=', a.eval_seed, 'global=', tuning.config.DEFAULT_SEED, 'eval_global=', tuning.config.get_eval_seed())"`
Expected output: `seed= 42 eval_seed= None global= 42 eval_global= 42`

- [ ] **Step 10: Commit**

```bash
git add tuning/training/unified_early_pipeline.py tests/test_unified_early_pipeline.py
git commit -m "Wire --seed and --eval_seed through unified early pipeline via _init_seeds helper"
```

---
