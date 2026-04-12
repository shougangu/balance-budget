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
        test_split = dataset["test"]
        all_prompts = list(train["prompt"]) + list(test_split["prompt"])
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
        test = dataset["test"]
        # "What is 2+2?" has 2 solutions in math source, both should be present
        all_prompts = list(train["prompt"]) + list(test["prompt"])
        count_2plus2 = sum(1 for p in all_prompts if "2+2" in p)
        assert count_2plus2 == 2
