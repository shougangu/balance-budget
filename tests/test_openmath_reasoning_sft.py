# ABOUTME: Tests for OpenMathReasoning SFT dataset loader.
# ABOUTME: Validates qwq filtering, correctness checking, token-range filtering, and deduplication.

from datasets import Dataset


def _make_fake_cot_rows():
    """Build a small in-memory dataset matching nvidia/OpenMathReasoning cot schema."""

    def qwq_solution(answer, extra=""):
        return f"<think>\nLet me work this out step by step. {extra}\n</think>\n\nThe answer is $\\boxed{{{answer}}}$"

    def long_solution(answer):
        return f"<think>\n{'This is a detailed reasoning step. ' * 100}\n</think>\n\nThe answer is $\\boxed{{{answer}}}$"

    def tiny_solution(answer):
        return f"$\\boxed{{{answer}}}$"

    def wrong_solution():
        return "<think>\nI'll just guess.\n</think>\n\nThe answer is $\\boxed{999}$"

    return Dataset.from_list([
        # passes all filters (problem A)
        {"problem": "What is 2+2?", "expected_answer": "4", "generation_model": "QwQ-32B",
         "generated_solution": qwq_solution("4", "2 plus 2 equals 4.")},
        # wrong generation_model → excluded
        {"problem": "What is 3+3?", "expected_answer": "6", "generation_model": "deepseek-r1",
         "generated_solution": qwq_solution("6")},
        # duplicate of problem A → excluded by dedup
        {"problem": "What is 2+2?", "expected_answer": "4", "generation_model": "QwQ-32B",
         "generated_solution": qwq_solution("4", "2 and 2 together make 4.")},
        # wrong answer → excluded by correctness filter
        {"problem": "What is 5+5?", "expected_answer": "10", "generation_model": "QwQ-32B",
         "generated_solution": wrong_solution()},
        # empty expected_answer → excluded
        {"problem": "Explain the concept of infinity.", "expected_answer": "",
         "generation_model": "QwQ-32B", "generated_solution": qwq_solution("")},
        # too long → excluded by max token bound
        {"problem": "What is 7+7?", "expected_answer": "14", "generation_model": "QwQ-32B",
         "generated_solution": long_solution("14")},
        # too short → excluded by min token bound
        {"problem": "What is 8+8?", "expected_answer": "16", "generation_model": "QwQ-32B",
         "generated_solution": tiny_solution("16")},
        # passes all filters (problem B)
        {"problem": "What is 9+9?", "expected_answer": "18", "generation_model": "QwQ-32B",
         "generated_solution": qwq_solution("18", "9 and 9 together make 18.")},
    ])


class TestOpenMathReasoningSFT:
    @classmethod
    def setup_class(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    def _make_loader(self):
        from tuning.data.openmath_reasoning_sft import OpenMathReasoningSFT
        loader = OpenMathReasoningSFT(
            tokenizer=self.__class__.tokenizer,
            min_tokens=10,
            max_tokens=100,
        )
        loader._dataset = _make_fake_cot_rows()
        return loader

    def _all_rows(self, loader):
        """Every row built from the source corpus. The test split comes from elsewhere."""
        ds = loader.get_dataset()
        return list(ds["train"])

    def test_only_two_problems_pass_all_filters(self):
        """Only the 2+2 and 9+9 problems should survive all filters."""
        loader = self._make_loader()
        loader.format_dataset()
        assert len(self._all_rows(loader)) == 2

    def test_excludes_non_qwq_models(self):
        """Rows with generation_model != 'qwq' should be excluded."""
        loader = self._make_loader()
        loader.format_dataset()
        user_contents = [r["messages"][1]["content"] for r in self._all_rows(loader)]
        assert not any("3+3" in c for c in user_contents)

    def test_excludes_wrong_answers(self):
        """Solutions where extracted answer does not match expected_answer should be excluded."""
        loader = self._make_loader()
        loader.format_dataset()
        user_contents = [r["messages"][1]["content"] for r in self._all_rows(loader)]
        assert not any("5+5" in c for c in user_contents)

    def test_excludes_empty_expected_answer(self):
        """Rows with empty expected_answer should be excluded."""
        loader = self._make_loader()
        loader.format_dataset()
        user_contents = [r["messages"][1]["content"] for r in self._all_rows(loader)]
        assert not any("infinity" in c.lower() for c in user_contents)

    def test_excludes_solutions_exceeding_max_tokens(self):
        """Solutions exceeding max_tokens should be excluded."""
        loader = self._make_loader()
        loader.format_dataset()
        user_contents = [r["messages"][1]["content"] for r in self._all_rows(loader)]
        assert not any("7+7" in c for c in user_contents)

    def test_excludes_solutions_below_min_tokens(self):
        """Solutions below min_tokens should be excluded."""
        loader = self._make_loader()
        loader.format_dataset()
        user_contents = [r["messages"][1]["content"] for r in self._all_rows(loader)]
        assert not any("8+8" in c for c in user_contents)

    def test_deduplicates_by_problem(self):
        """Duplicate problems should appear only once (first occurrence kept)."""
        loader = self._make_loader()
        loader.format_dataset()
        user_contents = [r["messages"][1]["content"] for r in self._all_rows(loader)]
        assert sum(1 for c in user_contents if "2+2" in c) == 1

    def test_messages_has_system_user_assistant(self):
        """Each row should have [system, user, assistant] messages."""
        loader = self._make_loader()
        loader.format_dataset()
        row = self._all_rows(loader)[0]
        msgs = row["messages"]
        assert len(msgs) == 3
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"

    def test_system_message_mentions_boxed(self):
        loader = self._make_loader()
        loader.format_dataset()
        row = self._all_rows(loader)[0]
        assert "boxed" in row["messages"][0]["content"]

    def test_user_message_uses_compmath_format(self):
        """User message should use COMPMATH_STRING format starting with 'Problem:'."""
        loader = self._make_loader()
        loader.format_dataset()
        row = self._all_rows(loader)[0]
        assert row["messages"][1]["content"].startswith("Problem:")

    def test_assistant_message_is_raw_qwq_solution(self):
        """Assistant message should be the unmodified QwQ solution starting with <think>."""
        loader = self._make_loader()
        loader.format_dataset()
        row = self._all_rows(loader)[0]
        assert row["messages"][2]["content"].startswith("<think>")

    def test_prompt_matches_user_message_content(self):
        """prompt field should equal the user message content."""
        loader = self._make_loader()
        loader.format_dataset()
        row = self._all_rows(loader)[0]
        assert row["prompt"] == row["messages"][1]["content"]

    def test_output_has_train_and_test_splits(self):
        loader = self._make_loader()
        loader.format_dataset()
        ds = loader.get_dataset()
        assert "train" in ds
        assert "test" in ds
