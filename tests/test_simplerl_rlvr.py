# ABOUTME: Tests for SimpleRL-Zoo RLVR dataset integration.
# ABOUTME: Covers alias resolution, CLI parsing of simplerl dataset choices, and reward function dispatch.

import argparse
import pytest

from tuning.training.unified_early_pipeline import (
    _parse_args,
    _resolve_simplerl_dataset,
    _build_reward_funcs,
    MODEL_TO_SIMPLERL_TIER,
)


REQUIRED = ["--model", "llama3-3B", "--wandb-project", "tuning"]


class TestResolveSimplerlDataset:
    def test_rewrites_simplerl_to_concrete_tier(self):
        args = argparse.Namespace(dataset="simplerl", model="llama3-8B")
        _resolve_simplerl_dataset(args)
        assert args.dataset == "simplerl-medium"

    def test_leaves_concrete_tier_unchanged(self):
        args = argparse.Namespace(dataset="simplerl-hard", model="llama3-8B")
        _resolve_simplerl_dataset(args)
        assert args.dataset == "simplerl-hard"

    def test_leaves_unrelated_dataset_unchanged(self):
        args = argparse.Namespace(dataset="gsm8k", model="llama3-8B")
        _resolve_simplerl_dataset(args)
        assert args.dataset == "gsm8k"

    def test_all_models_have_tier_mapping(self):
        expected_models = {"llama3-1B", "llama3-3B", "llama3-8B",
                           "qwen2-2B", "qwen2-3B", "qwen2-7B"}
        assert set(MODEL_TO_SIMPLERL_TIER.keys()) == expected_models

    def test_all_tier_values_are_valid(self):
        valid_tiers = {"easy", "medium", "hard"}
        for model, tier in MODEL_TO_SIMPLERL_TIER.items():
            assert tier in valid_tiers, f"{model} maps to invalid tier {tier!r}"


class TestParseArgsSimplerl:
    def test_simplerl_accepted(self):
        args = _parse_args(REQUIRED + ["--dataset", "simplerl"])
        assert args.dataset == "simplerl"

    def test_simplerl_easy_accepted(self):
        args = _parse_args(REQUIRED + ["--dataset", "simplerl-easy"])
        assert args.dataset == "simplerl-easy"

    def test_simplerl_medium_accepted(self):
        args = _parse_args(REQUIRED + ["--dataset", "simplerl-medium"])
        assert args.dataset == "simplerl-medium"

    def test_simplerl_hard_accepted(self):
        args = _parse_args(REQUIRED + ["--dataset", "simplerl-hard"])
        assert args.dataset == "simplerl-hard"


class TestBuildRewardFuncsSimplerl:
    def test_simplerl_easy_uses_math500_reward(self):
        from tuning.training.reward_functions import math500_reward_func
        args = argparse.Namespace(dataset="simplerl-easy")
        funcs = _build_reward_funcs(args)
        assert funcs == [math500_reward_func]

    def test_simplerl_medium_uses_math500_reward(self):
        from tuning.training.reward_functions import math500_reward_func
        args = argparse.Namespace(dataset="simplerl-medium")
        funcs = _build_reward_funcs(args)
        assert funcs == [math500_reward_func]

    def test_simplerl_hard_uses_math500_reward(self):
        from tuning.training.reward_functions import math500_reward_func
        args = argparse.Namespace(dataset="simplerl-hard")
        funcs = _build_reward_funcs(args)
        assert funcs == [math500_reward_func]

    def test_simplerl_alias_uses_math500_reward(self):
        from tuning.training.reward_functions import math500_reward_func
        args = argparse.Namespace(dataset="simplerl")
        funcs = _build_reward_funcs(args)
        assert funcs == [math500_reward_func]
