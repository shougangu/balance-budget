# ABOUTME: Reports test-vs-train contamination and train-internal near-duplication per dataset.
# ABOUTME: Run over the built datasets in DATASETS_DIR to see how self-similar each split is.

import argparse
import json
import random
from collections import Counter

from datasets import load_from_disk

from tuning.config import DATASETS_DIR
from tuning.data.near_duplicates import (
    NearDuplicateIndex,
    exact_jaccard,
    shingles,
    signature,
)

THRESHOLDS = (0.99, 0.9, 0.8, 0.7, 0.5)
TRAIN_SAMPLE = 2000


def _user_content(messages):
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content") or ""
    return ""


def prompt_of(row):
    """The user-visible prompt, whichever column shape the dataset uses.

    Datasets store it as a plain string, as a chat message list under `prompt`,
    or only inside `messages`.
    """
    value = row.get("prompt")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, list):
        return _user_content(value)
    return _user_content(row.get("messages"))


def _profile(corpus_shingles, index, queries, exclude):
    """Best similarity of each query against the indexed corpus."""
    scores = []
    for position, query_shingles in queries:
        best = 0.0
        for candidate in index.query(signature(query_shingles)):
            if exclude and candidate == position:
                continue
            score = exact_jaccard(query_shingles, corpus_shingles[candidate])
            if score > best:
                best = score
        scores.append(best)
    return scores


def summarise(scores, labels=None):
    total = len(scores) or 1
    out = {}
    for threshold in THRESHOLDS:
        hit = [i for i, s in enumerate(scores) if s >= threshold]
        entry = {"count": len(hit), "pct": round(100 * len(hit) / total, 1)}
        if labels:
            entry["by_source"] = dict(Counter(labels[i] for i in hit))
        out[str(threshold)] = entry
    return out


def audit(name, sample=TRAIN_SAMPLE, seed=0):
    dataset = load_from_disk(f"{DATASETS_DIR}/{name}")
    if "train" not in dataset or "test" not in dataset:
        return None

    train_rows = dataset["train"]
    test_rows = dataset["test"]
    labels = train_rows["source"] if "source" in train_rows.column_names else None

    train_prompts = [prompt_of(r) for r in train_rows]
    test_prompts = [prompt_of(r) for r in test_rows]

    train_shingles = [shingles(p) for p in train_prompts]
    index = NearDuplicateIndex()
    for position, row_shingles in enumerate(train_shingles):
        index.add(position, signature(row_shingles))

    test_queries = [(-1, shingles(p)) for p in test_prompts]
    test_scores = _profile(train_shingles, index, test_queries, exclude=False)

    picks = random.Random(seed).sample(range(len(train_prompts)), min(sample, len(train_prompts)))
    train_queries = [(i, train_shingles[i]) for i in picks]
    train_scores = _profile(train_shingles, index, train_queries, exclude=True)

    exact = len({" ".join(p.split()).lower() for p in test_prompts} &
                {" ".join(p.split()).lower() for p in train_prompts})

    return {
        "dataset": name,
        "train_rows": len(train_prompts),
        "test_rows": len(test_prompts),
        "exact_test_prompts_in_train": exact,
        "test_vs_train": summarise(test_scores),
        "train_vs_train": summarise(
            train_scores, labels=[labels[i] for i in picks] if labels else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", help="dataset directory names; default all")
    parser.add_argument("--sample", type=int, default=TRAIN_SAMPLE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    import os

    names = args.datasets or sorted(
        d for d in os.listdir(DATASETS_DIR)
        if os.path.isfile(os.path.join(DATASETS_DIR, d, "dataset_dict.json"))
    )

    reports = []
    for name in names:
        print(f"auditing {name} ...", flush=True)
        try:
            report = audit(name, sample=args.sample)
        except Exception as error:
            print(f"  skipped: {type(error).__name__}: {error}", flush=True)
            continue
        if report is None:
            print("  skipped: no train/test split", flush=True)
            continue
        reports.append(report)
        t = report["test_vs_train"]
        r = report["train_vs_train"]
        print(
            f"  test->train  exact={report['exact_test_prompts_in_train']} "
            f">=0.9:{t['0.9']['pct']}% >=0.8:{t['0.8']['pct']}%  |  "
            f"train->train >=0.9:{r['0.9']['pct']}% >=0.8:{r['0.8']['pct']}%",
            flush=True,
        )

    if args.output:
        with open(args.output, "w") as handle:
            json.dump(reports, handle, indent=2)
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
