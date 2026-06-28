# ABOUTME: Tests for SimpleRL-Zoo RLVR dataset integration.
# ABOUTME: Covers CLI parsing, reward function dispatch, tier loading, and combined-dataset merge.

import argparse
import sys
from unittest.mock import MagicMock

import pytest
from datasets import Dataset, DatasetDict

sys.modules.setdefault("instruction_following_eval", MagicMock())
sys.modules.setdefault("instruction_following_eval.evaluation_lib", MagicMock())
sys.modules.setdefault("unsloth", MagicMock())

from tuning.training.pipeline.cli import _parse_args
from tuning.training.pipeline.stages import _build_reward_funcs


REQUIRED = ["--model", "llama3-3B", "--wandb-project", "tuning"]


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


from tuning.data.simplerl_rlvr import combine_tiers


def _tier(tier_name, n_train, n_test):
    """Build a formatted-style DatasetDict whose prompts are tagged with the tier name."""
    def rows(split, n):
        return [
            {
                "prompt": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": f"{tier_name}-{split}-{i}"},
                ],
                "reference_answer": f"{tier_name}-ans-{i}",
            }
            for i in range(n)
        ]
    return DatasetDict({
        "train": Dataset.from_list(rows("train", n_train)),
        "test": Dataset.from_list(rows("test", n_test)),
    })


def _user_texts(split):
    return [p[-1]["content"] for p in split["prompt"]]


class TestCombineTiers:
    def test_train_is_deduped_union_of_tiers(self):
        a = _tier("a", 10, 5)
        b = _tier("b", 10, 5)
        combined = combine_tiers([a, b])
        texts = _user_texts(combined["train"])
        assert len(texts) == len(set(texts))
        assert set(texts) == set(_user_texts(a["train"])) | set(_user_texts(b["train"]))

    def test_shared_prompt_appears_once_with_first_tier_answer(self):
        shared = {
            "prompt": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "shared"},
            ],
            "reference_answer": "first",
        }
        a = DatasetDict({
            "train": Dataset.from_list([shared]),
            "test": Dataset.from_list([]),
        })
        b = DatasetDict({
            "train": Dataset.from_list([{**shared, "reference_answer": "second"}]),
            "test": Dataset.from_list([]),
        })
        combined = combine_tiers([a, b])
        assert len(combined["train"]) == 1
        assert combined["train"][0]["reference_answer"] == "first"

    def test_test_split_capped_at_200(self):
        tiers = [_tier(name, 0, 150) for name in ("a", "b", "c")]
        combined = combine_tiers(tiers)
        assert len(combined["test"]) == 200

    def test_test_split_mixes_tiers(self):
        tiers = [_tier(name, 0, 150) for name in ("a", "b", "c")]
        combined = combine_tiers(tiers)
        seen_tiers = {t.split("-")[0] for t in _user_texts(combined["test"])}
        assert len(seen_tiers) >= 2

    def test_shuffle_is_deterministic(self):
        tiers = [_tier(name, 50, 0) for name in ("a", "b", "c")]
        first = _user_texts(combine_tiers(tiers)["train"])
        second = _user_texts(combine_tiers(tiers)["train"])
        assert first == second
