# ABOUTME: Regression tests for TuluMix source tagging and single-turn filtering.
# ABOUTME: Ensures Persona IF provenance survives as a compact boolean marker.

from datasets import Dataset

from tuning.data.tulumix_sft import (
    PERSONA_IF_COLUMN,
    PERSONA_IF_SOURCE,
    TuluMixSFT,
)


def test_tulumix_marks_persona_if_before_removing_source_column():
    dataset = Dataset.from_dict(
        {
            "source": [PERSONA_IF_SOURCE, "another/source", PERSONA_IF_SOURCE],
            "messages": [
                [
                    {"role": "user", "content": "persona prompt"},
                    {"role": "assistant", "content": "persona response"},
                ],
                [
                    {"role": "user", "content": "other prompt"},
                    {"role": "assistant", "content": "other response"},
                ],
                [
                    {"role": "user", "content": "turn one"},
                    {"role": "assistant", "content": "turn one response"},
                    {"role": "user", "content": "turn two"},
                    {"role": "assistant", "content": "turn two response"},
                ],
            ],
        }
    )

    formatted = TuluMixSFT()._format_rows(dataset)

    assert len(formatted) == 2
    assert formatted[PERSONA_IF_COLUMN] == [True, False]
    assert "source" not in formatted.column_names
