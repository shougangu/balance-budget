# ABOUTME: Tests for near-duplicate detection and decontaminated train/test splitting.
# ABOUTME: Covers shingling, MinHash/LSH recall, exact Jaccard, and holdout selection.

import pytest

from tuning.data.near_duplicates import (
    NearDuplicateIndex,
    exact_jaccard,
    holdout_without_near_duplicates,
    nearest_neighbours,
    normalize,
    shingles,
    signature,
)


def test_normalize_collapses_whitespace_case_and_punctuation():
    assert normalize("Hello,   WORLD!\n") == normalize("hello world")


def test_shingles_of_short_text_is_non_empty():
    assert len(shingles("two words")) > 0


def test_identical_texts_have_jaccard_one():
    a = "write a poem about the sea in exactly three sentences"
    assert exact_jaccard(shingles(a), shingles(a)) == pytest.approx(1.0)


def test_unrelated_texts_have_low_jaccard():
    a = shingles("write a poem about the sea in exactly three sentences")
    b = shingles("implement a red black tree in rust with unit tests")
    assert exact_jaccard(a, b) < 0.05


TEMPLATE = (
    "Your ENTIRE response should be in en language, no other language is allowed. "
    "Your response should contain at least {} sentences. Your answer must contain a "
    "title, wrapped in double angular brackets, such as <<poem of joy>>. Highlight at "
    "least 2 sections in your answer with markdown, i.e. *highlighted section*. At the "
    "end of your response, please explicitly add a postscript starting with P.S."
)


def test_template_twins_are_near_duplicates():
    """The failure mode found in ifmix: same prompt, one integer changed."""
    assert exact_jaccard(shingles(TEMPLATE.format(3)), shingles(TEMPLATE.format(5))) > 0.5


def test_digits_are_content_not_noise():
    """Masking digits would collapse distinct maths problems into one row."""
    a = shingles("What is the remainder when 2 raised to the power 100 is divided by 7")
    b = shingles("What is the remainder when 3 raised to the power 250 is divided by 11")
    assert exact_jaccard(a, b) < 0.6


def test_index_retrieves_a_near_duplicate_as_candidate():
    index = NearDuplicateIndex()
    index.add("a", signature(shingles(TEMPLATE.format(3))))
    assert "a" in index.query(signature(shingles(TEMPLATE.format(5))))


def test_index_does_not_retrieve_unrelated_text():
    index = NearDuplicateIndex()
    index.add("a", signature(shingles("write a poem about the sea")))
    unrelated = signature(shingles("implement a red black tree in rust"))
    assert index.query(unrelated) == set()


def test_nearest_neighbours_scores_each_query_against_the_corpus():
    corpus = [
        TEMPLATE.format(3),
        "implement a red black tree in rust with unit tests",
    ]
    queries = [
        TEMPLATE.format(5),
        "explain photosynthesis to a child using simple words",
    ]
    scores = nearest_neighbours(corpus, queries)
    assert scores[0][0] > 0.5
    assert scores[0][1] == 0
    assert scores[1][0] < 0.2


def test_nearest_neighbours_can_exclude_self_matches():
    corpus = ["write a poem about the sea", "implement a red black tree in rust"]
    scores = nearest_neighbours(corpus, corpus, exclude_self=True)
    # Each row's only exact match is itself, which is excluded.
    assert all(score < 0.5 for score, _ in scores)


DISTINCT = [
    "explain how photosynthesis converts sunlight into chemical energy in plant cells",
    "write a short story about a lighthouse keeper who befriends a migrating whale",
    "summarise the causes of the fall of the western roman empire for a student",
    "implement a red black tree in rust including rotation and rebalancing helpers",
    "compare the monetary policy tools available to a central bank during deflation",
    "describe the rules of cricket to somebody who has only ever watched baseball",
    "draft a polite email declining a vendor proposal while leaving the door open",
    "outline a training plan for a first marathon over eighteen weeks of running",
    "explain why the sky appears blue and why sunsets appear red near the horizon",
    "review the tradeoffs between message queues and direct synchronous http calls",
]


def test_holdout_excludes_rows_with_a_near_duplicate_in_train():
    texts = [TEMPLATE.format(n) for n in range(20)] + DISTINCT
    train_idx, test_idx = holdout_without_near_duplicates(texts, test_size=5, threshold=0.5)

    assert len(test_idx) == 5
    assert not (set(train_idx) & set(test_idx))
    assert set(train_idx) | set(test_idx) == set(range(len(texts)))
    # Every held-out row must be far from everything left in train.
    held = [texts[i] for i in test_idx]
    kept = [texts[i] for i in train_idx]
    assert all(score < 0.5 for score, _ in nearest_neighbours(kept, held))


def test_holdout_is_deterministic_for_a_fixed_seed():
    first = holdout_without_near_duplicates(DISTINCT, test_size=3, threshold=0.9, seed=7)
    second = holdout_without_near_duplicates(DISTINCT, test_size=3, threshold=0.9, seed=7)
    assert first == second


def test_holdout_raises_when_too_few_clean_rows_exist():
    """All rows are twins, so no clean holdout of the requested size exists."""
    texts = [TEMPLATE.format(n) for n in range(10)]
    with pytest.raises(ValueError, match="clean"):
        holdout_without_near_duplicates(texts, test_size=5, threshold=0.5)


def test_holdout_respects_group_quotas():
    """Stratified holdout keeps each group's share of the test split fixed."""
    texts = DISTINCT + [t.upper() + " extra tail words to differ" for t in DISTINCT]
    groups = ["a"] * len(DISTINCT) + ["b"] * len(DISTINCT)

    train_idx, test_idx = holdout_without_near_duplicates(
        texts, test_size=6, threshold=0.9, groups=groups, quotas={"a": 4, "b": 2}
    )

    held = [groups[i] for i in test_idx]
    assert held.count("a") == 4
    assert held.count("b") == 2
    assert len(train_idx) == len(texts) - 6


def test_holdout_with_quotas_still_excludes_near_duplicates():
    """A quota never forces a contaminated row into the holdout."""
    twins = [TEMPLATE.format(n) for n in range(10)]
    texts = DISTINCT + twins
    groups = ["clean"] * len(DISTINCT) + ["twins"] * len(twins)

    with pytest.raises(ValueError, match="twins"):
        holdout_without_near_duplicates(
            texts, test_size=5, threshold=0.5, groups=groups,
            quotas={"clean": 3, "twins": 2},
        )


def test_holdout_quotas_must_sum_to_test_size():
    with pytest.raises(ValueError, match="quotas"):
        holdout_without_near_duplicates(
            DISTINCT, test_size=5, threshold=0.9,
            groups=["a"] * len(DISTINCT), quotas={"a": 4},
        )
