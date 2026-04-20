# ABOUTME: Tests for evaluation_lib refactoring (injectable registry + null-filtering).
# ABOUTME: Ensures backward compatibility and new IFBench support.

from instruction_following_eval.evaluation_lib import (
    test_instruction_following_strict,
    test_instruction_following_loose,
    InputExample,
)


class FakeInstruction:
    """Minimal constraint checker for testing."""
    def __init__(self, instruction_id):
        self.id = instruction_id
        self._keyword = None

    def build_description(self, keyword=None):
        self._keyword = keyword

    def get_instruction_args(self):
        return {"keyword": self._keyword}

    def get_instruction_args_keys(self):
        return ["keyword"]

    def check_following(self, value):
        return self._keyword is not None and self._keyword in value


FAKE_REGISTRY = {"test:keyword_check": FakeInstruction}


def test_strict_accepts_custom_instruction_dict():
    inp = InputExample(
        key=0,
        instruction_id_list=["test:keyword_check"],
        prompt="Say hello with the keyword 'banana'.",
        kwargs=[{"keyword": "banana"}],
    )
    result = test_instruction_following_strict(
        inp,
        {inp.prompt: "Hello banana world!"},
        instruction_dict=FAKE_REGISTRY,
    )
    assert result.follow_all_instructions is True
    assert result.follow_instruction_list == [True]


def test_loose_accepts_custom_instruction_dict():
    inp = InputExample(
        key=0,
        instruction_id_list=["test:keyword_check"],
        prompt="Say hello with the keyword 'banana'.",
        kwargs=[{"keyword": "banana"}],
    )
    result = test_instruction_following_loose(
        inp,
        {inp.prompt: "Hello banana world!"},
        instruction_dict=FAKE_REGISTRY,
    )
    assert result.follow_all_instructions is True
    assert result.follow_instruction_list == [True]


def test_strict_defaults_to_builtin_registry():
    """When no instruction_dict is passed, the existing IFEval registry is used."""
    inp = InputExample(
        key=0,
        instruction_id_list=["keywords:existence"],
        prompt="Include the word 'hello' in your response.",
        kwargs=[{"keywords": ["hello"]}],
    )
    result = test_instruction_following_strict(
        inp,
        {inp.prompt: "hello world"},
    )
    assert result.follow_all_instructions is True


def test_strict_filters_none_kwargs():
    """None values in kwargs should be filtered out before build_description."""
    inp = InputExample(
        key=0,
        instruction_id_list=["test:keyword_check"],
        prompt="test",
        kwargs=[{"keyword": "banana", "irrelevant_param": None}],
    )
    result = test_instruction_following_strict(
        inp,
        {inp.prompt: "banana"},
        instruction_dict=FAKE_REGISTRY,
    )
    assert result.follow_all_instructions is True


def test_loose_filters_none_kwargs():
    """None values in kwargs should be filtered out before build_description."""
    inp = InputExample(
        key=0,
        instruction_id_list=["test:keyword_check"],
        prompt="test",
        kwargs=[{"keyword": "banana", "extra": None, "another": None}],
    )
    result = test_instruction_following_loose(
        inp,
        {inp.prompt: "banana"},
        instruction_dict=FAKE_REGISTRY,
    )
    assert result.follow_all_instructions is True
