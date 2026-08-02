# ABOUTME: Tests for OpenMath SFT and RLVR dataset loaders.
# ABOUTME: Validates dataset format, filtering, column names, and deduplication.

from datasets import Dataset
from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING


def test_openmath_system_message_exists():
    """SYSTEM_MESSAGE_OPENMATH should be a non-empty string mentioning boxed format."""
    assert isinstance(SYSTEM_MESSAGE_OPENMATH, str)
    assert len(SYSTEM_MESSAGE_OPENMATH) > 0
    assert "boxed" in SYSTEM_MESSAGE_OPENMATH


from tuning.data.openmath_sft import OpenMathSFT
from tuning.data.openmath_rlvr import OpenMathRLVR


def _make_fake_openmath_rows():
    """Build a small in-memory dataset with the same schema as nvidia/OpenMathInstruct-2."""
    return Dataset.from_list([
        {"problem": "What is 2+2?", "generated_solution": "2+2=4\n$\\boxed{4}$", "expected_answer": "4", "problem_source": "math"},
        {"problem": "What is 2+2?", "generated_solution": "The sum is 4.\n$\\boxed{4}$", "expected_answer": "4", "problem_source": "math"},
        {"problem": "What is 3+3?", "generated_solution": "3+3=6\n$\\boxed{6}$", "expected_answer": "6", "problem_source": "augmented_math"},
        {"problem": "What is 5*5?", "generated_solution": "5*5=25\n$\\boxed{25}$", "expected_answer": "25", "problem_source": "gsm8k"},
        {"problem": "What is 1+1?", "generated_solution": "1+1=2\n$\\boxed{2}$", "expected_answer": "2", "problem_source": "augmented_gsm8k"},
    ])


class TestOpenMathSFT:
    def _make_loader(self):
        loader = OpenMathSFT()
        loader._dataset = _make_fake_openmath_rows()
        return loader

    def test_filters_to_math_sources_only(self):
        """Only math and augmented_math rows should survive filtering."""
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        all_prompts = list(train["prompt"])
        # Should have exactly 3 rows (the math + augmented_math rows)
        assert len(all_prompts) == 3

    def test_sft_has_messages_and_prompt_columns(self):
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        assert "messages" in train.column_names
        assert "prompt" in train.column_names

    def test_sft_messages_format(self):
        """Messages should be [system, user, assistant] with correct roles."""
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        row = train[0]
        msgs = row["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert "boxed" in msgs[0]["content"]
        assert "Problem:" in msgs[1]["content"]

    def test_sft_keeps_all_solutions(self):
        """Multiple solutions for the same problem should all be kept."""
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        # "What is 2+2?" has 2 solutions in math source, both should be present
        all_prompts = list(train["prompt"])
        count_2plus2 = sum(1 for p in all_prompts if "2+2" in p)
        assert count_2plus2 == 2


class TestOpenMathRLVR:
    def _make_loader(self):
        loader = OpenMathRLVR()
        loader._dataset = _make_fake_openmath_rows()
        return loader

    def test_filters_to_math_sources_only(self):
        """Only math and augmented_math rows should survive filtering."""
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        test_split = dataset["test"]
        all_prompts = [p[1]["content"] for p in list(train["prompt"]) + list(test_split["prompt"])]
        # gsm8k source rows should be excluded
        assert not any("5*5" in p for p in all_prompts), "gsm8k source should be excluded"
        assert not any("1+1" in p for p in all_prompts), "augmented_gsm8k source should be excluded"

    def test_rlvr_has_prompt_and_reference_answer(self):
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        assert "prompt" in train.column_names
        assert "reference_answer" in train.column_names

    def test_rlvr_prompt_format(self):
        """Prompt should be [system, user] message list."""
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        row = train[0]
        assert isinstance(row["prompt"], list)
        assert len(row["prompt"]) == 2
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1]["role"] == "user"
        assert "boxed" in row["prompt"][0]["content"]
        assert "Problem:" in row["prompt"][1]["content"]

    def test_rlvr_deduplicates_by_problem(self):
        """RLVR should have one row per unique problem, not per solution."""
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        test_split = dataset["test"]
        all_prompts = [p[1]["content"] for p in list(train["prompt"]) + list(test_split["prompt"])]
        # "What is 2+2?" appears twice in math source but should be deduped to 1
        assert len(all_prompts) == len(set(all_prompts)), "RLVR should have unique prompts"
        # 2 unique math/augmented_math problems: "2+2" and "3+3"
        assert len(all_prompts) == 2

    def test_rlvr_reference_answer(self):
        """reference_answer should come from expected_answer field."""
        loader = self._make_loader()
        loader.format_dataset()
        dataset = loader.get_dataset()
        train = dataset["train"]
        test_split = dataset["test"]
        all_answers = list(train["reference_answer"]) + list(test_split["reference_answer"])
        assert all(isinstance(a, str) and len(a) > 0 for a in all_answers)
