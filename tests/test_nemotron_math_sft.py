# ABOUTME: Tests for the Nemotron-Math-v2 SFT dataset builder.
# ABOUTME: Validates tool-use filtering, one-solution-per-problem, think rendering, and token cap.

from unittest.mock import patch

from datasets import Dataset

from tuning.data.config import COMPMATH_STRING, SYSTEM_MESSAGE_OPENMATH


def _assistant(reasoning, content):
    return {"role": "assistant", "content": content, "reasoning_content": reasoning,
            "tool_calls": [], "tool_call_id": "", "name": ""}


def _user(problem):
    return {"role": "user", "content": f"Solve.\n\n{problem}", "reasoning_content": "",
            "tool_calls": [], "tool_call_id": "", "name": ""}


def _tool(output):
    return {"role": "tool", "content": output, "reasoning_content": "",
            "tool_calls": [], "tool_call_id": "call_0", "name": "stateful_python_code_exec"}


PYTHON_TOOL = [{"type": "function", "function": {"name": "stateful_python_code_exec"}}]


def _make_fake_rows():
    """Rows shaped like nvidia/Nemotron-Math-v2 `low`/`medium` parquet records."""
    return Dataset.from_list([
        # tool-free, first solution to problem A -> kept
        {"problem": "What is 2+2?", "expected_answer": "4", "data_source": "aops", "tools": [],
         "messages": [_user("What is 2+2?"),
                      _assistant("Add 2 and 2 to get 4.", "The answer is $\\boxed{4}$.")]},
        # second tool-free solution to problem A -> dropped, one solution per problem
        {"problem": "What is 2+2?", "expected_answer": "4", "data_source": "aops", "tools": [],
         "messages": [_user("What is 2+2?"),
                      _assistant("2 and 2 together make 4.", "$\\boxed{4}$")]},
        # tool-integrated solution -> dropped
        {"problem": "What is 3+3?", "expected_answer": "6", "data_source": "aops",
         "tools": PYTHON_TOOL,
         "messages": [_user("What is 3+3?"),
                      {**_assistant("Use python.", ""),
                       "tool_calls": [{"id": "call_0", "type": "function",
                                       "function": {"name": "stateful_python_code_exec",
                                                    "arguments": "{\"code\": \"3+3\"}"}}]},
                      _tool("6"),
                      _assistant("", "$\\boxed{6}$")]},
        # too long once rendered -> dropped by the token cap
        {"problem": "What is 7+7?", "expected_answer": "14", "data_source": "stackflow",
         "tools": [],
         "messages": [_user("What is 7+7?"),
                      _assistant("This is a detailed reasoning step. " * 100,
                                 "$\\boxed{14}$")]},
        # tool-free, StackExchange problem B -> kept
        {"problem": "What is 9+9?", "expected_answer": "18", "data_source": "stackflow",
         "tools": [],
         "messages": [_user("What is 9+9?"),
                      _assistant("9 plus 9 is 18.", "So $\\boxed{18}$.")]},
    ])


class TestNemotronMathSFT:
    @classmethod
    def setup_class(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("unsloth/Meta-Llama-3.1-8B")

    def _build(self, max_solutions=1):
        from tuning.data.nemotron_math_sft import NemotronMathSFT
        loader = NemotronMathSFT(tokenizer=self.__class__.tokenizer, max_tokens=120,
                                 max_solutions_per_problem=max_solutions)
        loader._dataset = _make_fake_rows()
        fake_eval = Dataset.from_dict({"prompt": ["held out"]})
        loader.format_dataset(eval_dataset=fake_eval)
        return loader.get_dataset()

    def _train_rows(self, max_solutions=1):
        return list(self._build(max_solutions)["train"])

    def test_keeps_several_solutions_per_problem_when_allowed(self):
        """A higher cap keeps repeat solutions to the same problem, in corpus order."""
        rows = self._train_rows(max_solutions=2)
        assert len(rows) == 3
        assert "Add 2 and 2" in rows[0]["messages"][2]["content"]
        assert "2 and 2 together" in rows[1]["messages"][2]["content"]

    def test_cap_never_exceeds_available_solutions(self):
        """Problems with fewer solutions than the cap contribute what they have."""
        rows = self._train_rows(max_solutions=8)
        prompts = [r["prompt"] for r in rows]
        assert sum("9+9" in p for p in prompts) == 1

    def test_keeps_one_tool_free_solution_per_problem(self):
        prompts = [r["prompt"] for r in self._train_rows()]
        assert prompts == [COMPMATH_STRING.format(problem="What is 2+2?"),
                           COMPMATH_STRING.format(problem="What is 9+9?")]

    def test_first_solution_wins_for_a_repeated_problem(self):
        row = self._train_rows()[0]
        assert "Add 2 and 2" in row["messages"][2]["content"]

    def test_renders_reasoning_inside_think_tags_before_the_answer(self):
        row = self._train_rows()[1]
        assert row["messages"][2]["content"] == (
            "<think>\n9 plus 9 is 18.\n</think>\n\nSo $\\boxed{18}$."
        )

    def test_messages_are_system_user_assistant_in_openmath_format(self):
        row = self._train_rows()[0]
        msgs = row["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert msgs[0]["content"] == SYSTEM_MESSAGE_OPENMATH
        assert msgs[1]["content"] == row["prompt"]

    def test_output_has_train_and_test_splits(self):
        ds = self._build()
        assert ds["test"][0]["prompt"] == "held out"


def test_render_solution_strips_surrounding_whitespace():
    from tuning.data.nemotron_math_sft import render_solution
    assert render_solution("  think \n", "\n answer ") == "<think>\nthink\n</think>\n\nanswer"


def test_build_loads_the_requested_effort_split():
    from tuning.data.nemotron_math_sft import build_nemotron_math_sft
    fake_eval = Dataset.from_dict({"prompt": ["held out"]})
    with patch("tuning.data.nemotron_math_sft.load_dataset",
               return_value=_make_fake_rows()) as load, \
         patch("tuning.data.nemotron_math_sft.build_heldout_math_eval",
               return_value=fake_eval), \
         patch("tuning.data.hf_dataset.HFDataset.save_dataset_to_disk") as save:
        build_nemotron_math_sft(effort="medium", save_name="sft-nemotron-math-medium",
                                max_tokens=8000)
    assert load.call_args.args[0] == "nvidia/Nemotron-Math-v2"
    assert load.call_args.kwargs["data_files"] == "data/medium.parquet"
    assert save.call_args.kwargs["save_name"] == "sft-nemotron-math-medium"
