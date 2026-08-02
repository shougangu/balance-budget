# ABOUTME: Tests the 50/50 simplerl+openmath mixed RLVR dataset builder.

from datasets import Dataset, DatasetDict

from tuning.data.mathmix_rlvr import build_mathmix


def _fake(prefix, n_train, n_test=4):
    def rows(tag, n):
        return [{"prompt": [{"role": "system", "content": "s"},
                            {"role": "user", "content": f"{prefix}-{tag}-{i}"}],
                 "reference_answer": str(i)} for i in range(n)]
    return DatasetDict({
        "train": Dataset.from_list(rows("train", n_train)),
        "test": Dataset.from_list(rows("test", n_test)),
    })


def test_mathmix_is_half_simplerl_half_openmath():
    simplerl = _fake("srl", 10)
    openmath = _fake("om", 100)
    mixed = build_mathmix(simplerl, openmath, seed=42)

    assert mixed["train"].num_rows == 20
    sources = [r["prompt"][-1]["content"].split("-")[0] for r in mixed["train"]]
    assert sources.count("srl") == 10
    assert sources.count("om") == 10


def test_mathmix_train_is_shuffled_not_concatenated():
    mixed = build_mathmix(_fake("srl", 10), _fake("om", 100), seed=42)
    sources = [r["prompt"][-1]["content"].split("-")[0] for r in mixed["train"]]
    assert sources[:10] != ["srl"] * 10


def test_mathmix_skips_openmath_prompts_already_in_simplerl():
    simplerl = _fake("dup", 5)
    openmath = _fake("dup", 5)  # identical prompts
    extra = _fake("om", 20)
    from datasets import concatenate_datasets
    openmath = DatasetDict({
        "train": concatenate_datasets([openmath["train"], extra["train"]]),
        "test": extra["test"],
    })
    mixed = build_mathmix(simplerl, openmath, seed=42)
    contents = [r["prompt"][-1]["content"] for r in mixed["train"]]
    assert len(contents) == len(set(contents)) == 10


def test_mathmix_test_split_is_half_and_half():
    mixed = build_mathmix(_fake("srl", 10, n_test=8), _fake("om", 100, n_test=8), seed=42)
    sources = [r["prompt"][-1]["content"].split("-")[0] for r in mixed["test"]]
    assert sources.count("srl") == 4
    assert sources.count("om") == 4


def test_mathmix_test_excludes_problems_the_other_corpus_trains_on():
    """Both corpora derive from MATH, so one's test problem can sit in the other's train pool."""
    simplerl = _fake("srl", 10, n_test=8)
    openmath = _fake("om", 100, n_test=8)

    shared = simplerl["train"][0]["prompt"][-1]["content"]
    openmath_test = openmath["test"].to_list()
    openmath_test[0]["prompt"][-1]["content"] = shared
    openmath = DatasetDict(
        {"train": openmath["train"], "test": Dataset.from_list(openmath_test)}
    )

    mixed = build_mathmix(simplerl, openmath, seed=42)

    test_contents = [r["prompt"][-1]["content"] for r in mixed["test"]]
    train_contents = {r["prompt"][-1]["content"] for r in mixed["train"]}
    assert shared not in test_contents
    assert not train_contents & set(test_contents)
