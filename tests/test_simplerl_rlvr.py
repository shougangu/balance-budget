# ABOUTME: Tests for SimpleRL-Zoo RLVR dataset integration.
# ABOUTME: Covers alias resolution, CLI parsing, reward function dispatch, and dataset loading.

import argparse
import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from tuning.training.pipeline.cli import (
    _parse_args,
    _resolve_simplerl_dataset,
    MODEL_TO_SIMPLERL_TIER,
)
from tuning.training.pipeline.stages import _build_reward_funcs


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


from tuning.data.simplerl_rlvr import SimpleRLRLVR, SIMPLERL_TIERS


class TestSimpleRLRLVR:
    @pytest.fixture(params=["easy", "medium", "hard"])
    def loaded_dataset(self, request):
        difficulty = request.param
        ds = SimpleRLRLVR(difficulty)
        ds.load_from_huggingface("hkust-nlp/SimpleRL-Zoo-Data")
        ds.format_dataset()
        return ds.get_dataset(), difficulty

    def test_has_train_and_test_splits(self, loaded_dataset):
        dataset, _ = loaded_dataset
        assert "train" in dataset
        assert "test" in dataset

    def test_train_has_prompt_and_reference_answer(self, loaded_dataset):
        dataset, _ = loaded_dataset
        train = dataset["train"]
        assert "prompt" in train.column_names
        assert "reference_answer" in train.column_names

    def test_prompt_is_system_user_pair(self, loaded_dataset):
        dataset, _ = loaded_dataset
        row = dataset["train"][0]
        assert isinstance(row["prompt"], list)
        assert len(row["prompt"]) == 2
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1]["role"] == "user"

    def test_reference_answer_is_nonempty_string(self, loaded_dataset):
        dataset, _ = loaded_dataset
        row = dataset["train"][0]
        assert isinstance(row["reference_answer"], str)
        assert len(row["reference_answer"]) > 0

    def test_prompts_are_deduplicated(self, loaded_dataset):
        dataset, _ = loaded_dataset
        prompts = dataset["train"]["prompt"]
        user_texts = [p[1]["content"] for p in prompts]
        assert len(user_texts) == len(set(user_texts))

    def test_uses_openmath_system_message(self, loaded_dataset):
        from tuning.data.config import SYSTEM_MESSAGE_OPENMATH
        dataset, _ = loaded_dataset
        row = dataset["train"][0]
        assert row["prompt"][0]["content"] == SYSTEM_MESSAGE_OPENMATH

    def test_uses_compmath_format(self, loaded_dataset):
        dataset, _ = loaded_dataset
        row = dataset["train"][0]
        assert row["prompt"][1]["content"].startswith("Problem:")

    def test_train_size_approximately_8k(self, loaded_dataset):
        dataset, difficulty = loaded_dataset
        train_size = len(dataset["train"])
        assert 7000 <= train_size <= 9000, f"{difficulty} has {train_size} train rows, expected ~8k"

    def test_test_split_is_nonempty(self, loaded_dataset):
        dataset, _ = loaded_dataset
        assert len(dataset["test"]) > 0


class TestSimplerlTiers:
    def test_tiers_dict_has_three_entries(self):
        assert set(SIMPLERL_TIERS.keys()) == {"easy", "medium", "hard"}

    def test_tier_subsets_are_abel_variants(self):
        for tier, subset in SIMPLERL_TIERS.items():
            assert "abel" in subset, f"{tier} subset should use abel variant"
