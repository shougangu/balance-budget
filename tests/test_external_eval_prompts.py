# ABOUTME: Tests the prompt/template remapping used to re-evaluate external post-trained
# ABOUTME: models under both our protocol and the protocol their own papers report.

import pytest

from scripts.external_eval_calibration import (
    PROMPT_STYLES,
    build_messages,
    strip_prompt_wrapper,
)
from tuning.data.config import (
    COMPMATH_STRING,
    GSM8K_STRING,
    SYSTEM_MESSAGE_OPENMATH,
)


def test_strip_wrapper_recovers_math_problem():
    problem = "Convert the point $(0,3)$ to polar coordinates."
    wrapped = COMPMATH_STRING.format(problem=problem)
    assert strip_prompt_wrapper(wrapped) == problem


def test_strip_wrapper_recovers_gsm8k_question():
    question = "Janet has 3 apples. How many are left after eating 1?"
    wrapped = GSM8K_STRING.format(question=question)
    assert strip_prompt_wrapper(wrapped) == question


def test_strip_wrapper_passes_through_unwrapped_prompt():
    raw = "Write a poem about the sea in exactly four lines."
    assert strip_prompt_wrapper(raw) == raw


def test_ours_style_reproduces_repo_messages():
    problem = "What is $2+2$?"
    wrapped = COMPMATH_STRING.format(problem=problem)
    messages = build_messages("ours", "math500", wrapped)
    assert messages == [
        {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
        {"role": "user", "content": wrapped},
    ]


def test_boxed_style_matches_openmath_reference_prompt():
    problem = "What is $2+2$?"
    wrapped = COMPMATH_STRING.format(problem=problem)
    messages = build_messages("boxed", "math500", wrapped)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == (
        "Solve the following math problem. Make sure to put the answer "
        "(and only answer) inside \\boxed{}.\n\n" + problem
    )


def test_boxed_style_drops_our_answer_wrapper():
    wrapped = COMPMATH_STRING.format(problem="What is $2+2$?")
    content = build_messages("boxed", "math500", wrapped)[0]["content"]
    assert "Problem:" not in content
    assert not content.endswith("Answer:")


def test_plain_style_is_bare_user_turn():
    raw = "Write a limerick with no commas."
    assert build_messages("plain", "ifeval", raw) == [{"role": "user", "content": raw}]


def test_ifeval_ours_style_keeps_if_system_message():
    raw = "Write a limerick with no commas."
    messages = build_messages("ours", "ifeval", raw)
    assert messages[0]["role"] == "system"
    assert "carefully following the given instructions" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": raw}


def test_unknown_style_rejected():
    with pytest.raises(ValueError):
        build_messages("nonsense", "math500", "Problem: x\nAnswer:")


def test_all_declared_styles_are_buildable():
    wrapped = COMPMATH_STRING.format(problem="What is $2+2$?")
    for style in PROMPT_STYLES:
        messages = build_messages(style, "math500", wrapped)
        assert messages and messages[-1]["role"] == "user"
