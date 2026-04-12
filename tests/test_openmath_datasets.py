# ABOUTME: Tests for OpenMath SFT and RLVR dataset loaders.
# ABOUTME: Validates dataset format, filtering, column names, and deduplication.

from tuning.data.config import SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING


def test_openmath_system_message_exists():
    """SYSTEM_MESSAGE_OPENMATH should be a non-empty string mentioning boxed format."""
    assert isinstance(SYSTEM_MESSAGE_OPENMATH, str)
    assert len(SYSTEM_MESSAGE_OPENMATH) > 0
    assert "boxed" in SYSTEM_MESSAGE_OPENMATH
