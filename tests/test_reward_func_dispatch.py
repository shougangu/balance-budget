# ABOUTME: Tests dataset-name -> reward function dispatch in the post-training pipeline.

import sys
from types import ModuleType

if "unsloth" not in sys.modules:
    _stub = ModuleType("unsloth")
    _stub.is_bfloat16_supported = lambda: True
    sys.modules["unsloth"] = _stub

from tuning.training.pipeline.cli import _parse_args
from tuning.training.pipeline.stages import _build_reward_funcs


def _args(dataset):
    return _parse_args(["--model", "qwen2-3B", "--wandb-project", "test", "--dataset", dataset])


def test_dapo_uses_math500_reward():
    from tuning.training.reward_functions import math500_reward_func
    assert _build_reward_funcs(_args("dapo")) == [math500_reward_func]


def test_simplerl_uses_math500_reward():
    from tuning.training.reward_functions import math500_reward_func
    assert _build_reward_funcs(_args("simplerl")) == [math500_reward_func]


def test_mathmix_uses_math500_reward():
    from tuning.training.reward_functions import math500_reward_func
    assert _build_reward_funcs(_args("mathmix")) == [math500_reward_func]
