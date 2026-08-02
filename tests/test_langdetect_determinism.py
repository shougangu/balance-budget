# ABOUTME: Tests that language detection in the IF instruction checkers is repeatable.
# ABOUTME: langdetect samples n-grams at random unless its factory seed is pinned.

import langdetect

import ifrlvr.instructions  # noqa: F401  - imported for its seed pinning side effect

# Short multilingual text sits near the decision boundary, where unseeded
# langdetect averages 7 randomised trials into different answers per call.
AMBIGUOUS = "Merci beaucoup mon ami"


def test_detector_factory_seed_is_pinned():
    """Importing the instruction checkers pins langdetect's sampling seed."""
    assert langdetect.DetectorFactory.seed is not None


def test_repeated_detection_is_stable():
    """The same text detects to the same language on every call."""
    results = {langdetect.detect(AMBIGUOUS) for _ in range(25)}
    assert len(results) == 1, f"langdetect returned {results} for one input"


def test_language_instruction_verdict_is_stable():
    """A response-language check returns the same verdict across repeated calls."""
    checker = ifrlvr.instructions.ResponseLanguageChecker("language:response_language")
    checker.build_description(language="fr")
    verdicts = {checker.check_following(AMBIGUOUS) for _ in range(25)}
    assert len(verdicts) == 1, f"checker flip-flopped: {verdicts}"
