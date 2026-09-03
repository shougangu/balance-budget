# ABOUTME: Tests that SFT dataset preprocessing runs once on the main process with every
# ABOUTME: allocated CPU, and that the early process group outlives that preprocessing.

import contextlib
import os
from datetime import timedelta

from datasets import Dataset, DatasetDict

import tuning.training.pipeline.stages as stages
import tuning.training.sft_training as sft
from tuning.config import HF_MODEL_MAP
from tuning.utils.utils import LLAMA_31_SIMPLE_TEMPLATE, apply_chat_template, tokenize_sft_dataset

CONVOS = [
    [
        {"role": "system", "content": "You solve math problems."},
        {"role": "user", "content": "Problem: 2+2?\nAnswer:"},
        {"role": "assistant", "content": "<think>\nsum\n</think>\n\n\\boxed{4}"},
    ],
    [
        {"role": "system", "content": "You solve math problems."},
        {"role": "user", "content": "Problem: 3+3?\nAnswer:"},
        {"role": "assistant", "content": "Let us compute \\boxed{6}."},
    ],
]


def _tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_MAP["llama3-8B"])
    tokenizer.chat_template = LLAMA_31_SIMPLE_TEMPLATE
    return tokenizer


def _dataset():
    return DatasetDict({"train": Dataset.from_dict({"messages": CONVOS})})


class TestPreprocessSftDataset:
    def test_matches_the_two_pass_pipeline_on_real_tokens(self):
        tokenizer = _tokenizer()
        expected = tokenize_sft_dataset(
            tokenizer, apply_chat_template(tokenizer, _dataset(), mask_prompt=True),
            max_length=128, num_proc=None, mask_prompt=True,
        )
        got = sft.preprocess_sft_dataset(tokenizer, _dataset(), max_length=128, mask_prompt=True)
        for i in range(len(CONVOS)):
            assert got["train"][i]["input_ids"] == expected["train"][i]["input_ids"]
            assert got["train"][i]["completion_mask"] == expected["train"][i]["completion_mask"]
        assert "text" in got["train"].column_names

    def test_both_passes_run_inside_main_process_first(self, monkeypatch):
        events = []

        class FakeState:
            @contextlib.contextmanager
            def main_process_first(self):
                events.append("enter")
                yield
                events.append("exit")

        def fake_render(tokenizer, dataset, mask_prompt=False, num_proc=None):
            events.append(("render", num_proc))
            return dataset

        def fake_tokenize(tokenizer, dataset, max_length, num_proc=4, mask_prompt=False):
            events.append(("tokenize", num_proc, max_length, mask_prompt))
            return dataset

        monkeypatch.setattr(sft, "PartialState", FakeState)
        monkeypatch.setattr(sft, "apply_chat_template", fake_render)
        monkeypatch.setattr(sft, "tokenize_sft_dataset", fake_tokenize)
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4, 5, 6})

        sft.preprocess_sft_dataset(object(), _dataset(), max_length=64, mask_prompt=True)

        assert events == ["enter", ("render", 7), ("tokenize", 7, 64, True), "exit"]


class TestPreprocessingNumProc:
    def test_uses_the_cpus_this_process_may_run_on(self, monkeypatch):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {3, 9, 12})
        assert sft.preprocessing_num_proc() == 3

    def test_never_below_one(self, monkeypatch):
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set())
        assert sft.preprocessing_num_proc() == 1


class TestEarlyProcessGroup:
    def test_timeout_outlasts_rank_zero_preprocessing(self, monkeypatch):
        calls = {}
        monkeypatch.setenv("LOCAL_RANK", "0")
        monkeypatch.setattr(stages.dist, "is_initialized", lambda: False)
        monkeypatch.setattr(stages.dist, "init_process_group", lambda **kw: calls.update(kw))
        monkeypatch.setattr(stages.torch.cuda, "set_device", lambda idx: calls.update(device=idx))

        stages.init_process_group_early()

        assert calls["backend"] == "nccl"
        assert calls["timeout"] >= timedelta(hours=1)
        assert calls["device"] == 0

    def test_noop_outside_torchrun(self, monkeypatch):
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        monkeypatch.setattr(stages.dist, "init_process_group", lambda **kw: (_ for _ in ()).throw(AssertionError("must not init")))
        stages.init_process_group_early()
