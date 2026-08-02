# ABOUTME: Regression tests for the constraint-style IF SFT mix assembly.
# ABOUTME: Covers per-source normalisation, persona dedup, turn-1 cutting and seeded shuffling.

import pytest

from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
from tuning.data.ifmix_sft import (
    ARGILLA_SOURCE,
    NEMOTRON_PERSONA_SOURCE,
    NEMOTRON_SYNTHETIC_SOURCE,
    SHUFFLE_SEED,
    SMOL_SOURCE,
    _dedupe,
    argilla_rows,
    persona_rows,
    response_satisfies_constraints,
    shuffle_rows,
    smol_rows,
    synthetic_rows,
)


def _nemotron_record(seed_dataset, turns, reasoning="internal deliberation"):
    messages = [{"role": "system", "content": "nvidia system prompt"}]
    for user, assistant in turns:
        messages.append({"role": "user", "content": user})
        messages.append(
            {"role": "assistant", "content": assistant, "reasoning_content": reasoning}
        )
    return {"messages": messages, "metadata": {"seed_dataset": seed_dataset}}


def test_persona_rows_keep_one_response_per_prompt():
    records = [
        _nemotron_record(NEMOTRON_PERSONA_SOURCE, [("write a haiku", "first sample")]),
        _nemotron_record(NEMOTRON_PERSONA_SOURCE, [("write a haiku", "second sample")]),
        _nemotron_record(NEMOTRON_PERSONA_SOURCE, [("write a sonnet", "other prompt")]),
    ]

    rows = persona_rows(records)

    assert len(rows) == 2
    prompts = [row["messages"][1]["content"] for row in rows]
    assert sorted(prompts) == ["write a haiku", "write a sonnet"]
    assert rows[0]["messages"][2]["content"] == "first sample"


def test_persona_rows_ignore_synthetic_records():
    records = [_nemotron_record(NEMOTRON_SYNTHETIC_SOURCE, [("prompt", "response")])]

    assert persona_rows(records) == []


def test_synthetic_rows_keep_only_the_first_turn():
    records = [
        _nemotron_record(
            NEMOTRON_SYNTHETIC_SOURCE,
            [
                ("first question", "first answer"),
                ("follow up about that", "second answer"),
                ("another follow up", "third answer"),
            ],
        )
    ]

    rows = synthetic_rows(records)

    assert len(rows) == 1
    messages = rows[0]["messages"]
    assert len(messages) == 3
    assert messages[1]["content"] == "first question"
    assert messages[2]["content"] == "first answer"


def test_rows_strip_reasoning_content_and_use_the_if_system_message():
    records = [_nemotron_record(NEMOTRON_SYNTHETIC_SOURCE, [("prompt", "answer")])]

    messages = synthetic_rows(records)[0]["messages"]

    assert messages[0]["content"] == SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
    assert all("reasoning_content" not in message for message in messages)


def test_rows_record_their_source():
    persona = persona_rows([_nemotron_record(NEMOTRON_PERSONA_SOURCE, [("p", "a")])])
    synthetic = synthetic_rows([_nemotron_record(NEMOTRON_SYNTHETIC_SOURCE, [("p", "a")])])

    assert persona[0]["source"] == NEMOTRON_PERSONA_SOURCE
    assert synthetic[0]["source"] == NEMOTRON_SYNTHETIC_SOURCE


def test_smol_rows_normalise_two_message_conversations():
    records = [
        {
            "messages": [
                {"role": "user", "content": "write exactly 3 bullet points"},
                {"role": "assistant", "content": "* a\n* b\n* c"},
            ]
        }
    ]

    rows = smol_rows(records)

    assert len(rows) == 1
    assert rows[0]["source"] == SMOL_SOURCE
    assert rows[0]["messages"][0]["content"] == SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
    assert rows[0]["messages"][1]["content"] == "write exactly 3 bullet points"


def test_response_satisfies_constraints_accepts_a_compliant_answer():
    assert response_satisfies_constraints(
        "there are no capitals anywhere in this answer at all.",
        ["change_case:english_lowercase"],
        [{}],
    )


def test_response_satisfies_constraints_rejects_a_violating_answer():
    assert not response_satisfies_constraints(
        "This Answer Clearly Contains Capitals.",
        ["change_case:english_lowercase"],
        [{}],
    )


def test_response_satisfies_constraints_rejects_when_nothing_is_checkable():
    assert not response_satisfies_constraints("anything", [], [])


def test_response_satisfies_constraints_rejects_a_missing_parameter():
    """A constraint whose parameter never arrived cannot be checked against the data."""
    assert not response_satisfies_constraints(
        "a plain response with nothing unusual in it.",
        ["keywords:forbidden_words"],
        [{}],
    )


def test_response_satisfies_constraints_rejects_a_none_valued_parameter():
    """Argilla ships one shared mapping holding None for parameters it did not fill in."""
    assert not response_satisfies_constraints(
        "a plain response with nothing unusual in it.",
        ["keywords:forbidden_words"],
        [{"forbidden_words": None}],
    )


def test_response_satisfies_constraints_checks_a_supplied_parameter():
    """A constraint carrying its parameter is still evaluated against the response."""
    assert response_satisfies_constraints(
        "a plain response with nothing unusual in it.",
        ["keywords:forbidden_words"],
        [{"forbidden_words": ["zebra", "quasar"]}],
    )
    assert not response_satisfies_constraints(
        "a plain response mentioning a zebra.",
        ["keywords:forbidden_words"],
        [{"forbidden_words": ["zebra", "quasar"]}],
    )


def test_verification_verdict_is_repeatable():
    """Verification never depends on a value the checker invented for itself."""
    verdicts = {
        response_satisfies_constraints(
            "a plain response with nothing unusual in it.",
            ["keywords:forbidden_words"],
            [{}],
        )
        for _ in range(20)
    }
    assert len(verdicts) == 1


def test_response_satisfies_constraints_rejects_an_unsubstituted_placeholder():
    """Argilla ships rows whose template was never filled in, so the value is a stand-in.

    A checker rejects such a value and draws a real one, testing the response against
    a letter the prompt never asked for.
    """
    assert not response_satisfies_constraints(
        "the letter e appears here in several words, plainly and repeatedly.",
        ["keywords:letter_frequency"],
        [{"letter": "[letter]", "let_frequency": 3, "let_relation": "at least"}],
    )


def test_response_satisfies_constraints_rejects_a_letter_the_checker_cannot_express():
    """Constraints outside the checker's alphabet are unverifiable, not satisfied."""
    assert not response_satisfies_constraints(
        "问题问题问题",
        ["keywords:letter_frequency"],
        [{"letter": "问", "let_frequency": 3, "let_relation": "at least"}],
    )


def test_response_satisfies_constraints_rejects_an_empty_collection():
    """An empty keyword list is present and not None, yet states no constraint."""
    assert not response_satisfies_constraints(
        "a plain response with nothing unusual in it.",
        ["keywords:existence"],
        [{"keywords": []}],
    )


def test_verdict_on_a_substituted_parameter_is_repeatable():
    """The unverifiable row is rejected every time, not on a coin flip."""
    verdicts = {
        response_satisfies_constraints(
            "the letter e appears here in several words, plainly and repeatedly.",
            ["keywords:letter_frequency"],
            [{"letter": "[letter]", "let_frequency": 3, "let_relation": "at least"}],
        )
        for _ in range(20)
    }
    assert len(verdicts) == 1


def test_verification_leaves_the_shared_random_stream_untouched():
    """Row order must not shift the draws seen by anything else in the process."""
    import random

    random.seed(1234)
    baseline = [random.random() for _ in range(5)]

    random.seed(1234)
    response_satisfies_constraints(
        "the letter e appears here in several words.",
        ["keywords:letter_frequency"],
        [{"letter": "[letter]", "let_frequency": 3, "let_relation": "at least"}],
    )
    after = [random.random() for _ in range(5)]

    assert baseline == after


def _checker_raising(exception):
    """A checker class whose build_description fails with the given exception."""

    class Raising:
        def __init__(self, instruction_id):
            pass

        def build_description(self, num_sentences=None):
            raise exception

        def get_instruction_args(self):
            return {}

        def check_following(self, value):
            return True

    return Raising


def test_verification_reraises_a_missing_tokenizer(monkeypatch):
    """A checker that cannot run here says nothing about the response.

    Swallowing the failure would silently drop every row declaring that constraint,
    which is how one cluster built a third of the rows the others did.
    """
    from ifrlvr import instructions_registry

    monkeypatch.setitem(
        instructions_registry.FUNCTION_DICT,
        "length_constraints:number_sentences",
        _checker_raising(LookupError("Resource 'punkt' not found.")),
    )

    with pytest.raises(LookupError):
        response_satisfies_constraints(
            "one. two three. four five six.",
            ["length_constraints:number_sentences"],
            [{"num_sentences": 3}],
        )


def test_verification_rejects_a_row_the_checker_cannot_parse(monkeypatch):
    """A malformed value is a fact about the row, so the row is dropped as before.

    Checkers interpolate row values straight into regexes and format strings, so the
    error a bad value produces is open-ended and cannot be enumerated up front.
    """
    import re

    from ifrlvr import instructions_registry

    for error in (
        ValueError("bad value"),
        KeyError("absent"),
        IndexError("range"),
        re.error("missing ), unterminated subpattern"),
        RuntimeError("something the checker did not anticipate"),
    ):
        monkeypatch.setitem(
            instructions_registry.FUNCTION_DICT,
            "length_constraints:number_sentences",
            _checker_raising(error),
        )
        assert not response_satisfies_constraints(
            "one. two three.",
            ["length_constraints:number_sentences"],
            [{"num_sentences": 3}],
        )


def test_argilla_rows_drop_responses_that_fail_verification():
    records = [
        {
            "prompt": "Your entire response should be in English, and in all lowercase letters.",
            "response": "this one is entirely lowercase and therefore fine.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}],
        },
        {
            "prompt": "Your entire response should be in English, and in all lowercase letters.",
            "response": "This One SHOUTS and must be dropped.",
            "instruction_id_list": ["change_case:english_lowercase"],
            "kwargs": [{}],
        },
    ]

    rows = argilla_rows(records)

    assert len(rows) == 1
    assert rows[0]["source"] == ARGILLA_SOURCE
    assert rows[0]["messages"][2]["content"].startswith("this one")


def test_argilla_rows_accept_a_single_shared_kwargs_mapping():
    records = [
        {
            "instruction": "Answer using at least 2 placeholders such as [name].",
            "response": "Contact [name] at [address] for details.",
            "instruction_id_list": ["detectable_content:number_placeholders"],
            "kwargs": {"num_placeholders": 2},
        }
    ]

    assert len(argilla_rows(records)) == 1


def test_shuffle_rows_is_deterministic_for_the_pinned_seed():
    rows = [{"messages": [], "source": str(index)} for index in range(200)]

    first = shuffle_rows(list(rows))
    second = shuffle_rows(list(rows))

    assert [row["source"] for row in first] == [row["source"] for row in second]


def test_shuffle_rows_interleaves_the_sources():
    rows = [{"messages": [], "source": "a"} for _ in range(100)]
    rows += [{"messages": [], "source": "b"} for _ in range(100)]

    shuffled = shuffle_rows(rows)

    assert [row["source"] for row in shuffled] != [row["source"] for row in rows]
    assert sorted(row["source"] for row in shuffled) == sorted(row["source"] for row in rows)
    # Both sources must appear early once the order is randomised.
    assert len(set(row["source"] for row in shuffled[:40])) == 2


def test_shuffle_rows_does_not_depend_on_input_order():
    rows = [{"messages": [], "source": str(index)} for index in range(200)]

    from_forward = shuffle_rows(list(rows))
    from_reversed = shuffle_rows(list(reversed(rows)))

    assert [row["source"] for row in from_forward] == [row["source"] for row in from_reversed]


def test_shuffle_seed_is_pinned_to_42():
    assert SHUFFLE_SEED == 42


def test_dedupe_keeps_the_same_prompt_from_different_sources():
    rows = [
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Write exactly three bullets."},
                {"role": "assistant", "content": "Argilla response"},
            ],
            "source": ARGILLA_SOURCE,
        },
        {
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "Write exactly three bullets."},
                {"role": "assistant", "content": "Smol response"},
            ],
            "source": SMOL_SOURCE,
        },
    ]

    assert _dedupe(rows) == rows


def test_dedupe_normalises_and_drops_repeated_prompts_within_a_source():
    first = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Write exactly three bullets."},
            {"role": "assistant", "content": "First response"},
        ],
        "source": SMOL_SOURCE,
    }
    duplicate = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "  write EXACTLY   three bullets.  "},
            {"role": "assistant", "content": "Second response"},
        ],
        "source": SMOL_SOURCE,
    }

    assert _dedupe([first, duplicate]) == [first]
