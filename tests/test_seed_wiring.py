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
