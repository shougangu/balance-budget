# ABOUTME: Compares GSM8K scoring between lm-eval harness filters and gsm8k_scoring.is_correct.
# ABOUTME: Uses existing lm-eval sample outputs to check for discrepancies.

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tuning.evaluation.gsm8k_scoring import (
    is_correct,
    extract_gsm8k_answer_strict,
    extract_gsm8k_answer_flexible,
    normalize_answer,
)

BASE = Path(__file__).resolve().parent.parent / "tuning" / "outputs" / "gsm8k"

MODELS = {
    "qwen2-2B": "samples_gsm8k_matched_2026-03-09T10-23-43.934392.jsonl",
    "llama3-3B": "samples_gsm8k_matched_2026-03-09T10-15-13.364014.jsonl",
    "qwen2-7B": "samples_gsm8k_matched_2026-03-09T10-28-34.175768.jsonl",
}


def load_samples(model_name, filename):
    """Load lm-eval samples, keyed by (doc_id, filter)."""
    # Find the subdirectory (the long path-encoded dir name)
    model_dir = BASE / model_name / "responses.jsonl"
    subdirs = list(model_dir.iterdir())
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories in {model_dir}")
    sample_file = subdirs[0] / filename

    by_filter = defaultdict(dict)  # filter -> {doc_id: record}
    with open(sample_file) as f:
        for line in f:
            d = json.loads(line)
            by_filter[d["filter"]][d["doc_id"]] = d
    return by_filter


def compare_model(model_name, filename):
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    samples = load_samples(model_name, filename)
    flexible_samples = samples["flexible-extract"]
    strict_samples = samples["strict-match"]

    doc_ids = sorted(flexible_samples.keys())
    n = len(doc_ids)

    # Counters
    lm_flex_correct = 0
    lm_strict_correct = 0
    our_correct = 0
    disagree_flex = []  # (doc_id, lm_flex, ours)
    disagree_strict = []

    for doc_id in doc_ids:
        flex_rec = flexible_samples[doc_id]
        strict_rec = strict_samples[doc_id]

        ref_answer = flex_rec["doc"]["answer"]  # full answer text
        ref_num = ref_answer.split("####")[-1].strip()

        # Raw model response (same for both filter records)
        raw_response = flex_rec["resps"][0][0]

        # lm-eval scores
        lm_flex_score = flex_rec["exact_match"]
        lm_strict_score = strict_rec["exact_match"]
        lm_flex_correct += lm_flex_score
        lm_strict_correct += lm_strict_score

        # Our is_correct scoring
        our_score = 1.0 if is_correct(raw_response, ref_answer) else 0.0
        our_correct += our_score

        if lm_flex_score != our_score:
            disagree_flex.append({
                "doc_id": doc_id,
                "lm_flex": lm_flex_score,
                "ours": our_score,
                "ref": ref_num,
                "lm_extracted": flex_rec["filtered_resps"][0],
                "our_strict": extract_gsm8k_answer_strict(raw_response),
                "our_flexible": extract_gsm8k_answer_flexible(raw_response),
            })

        if lm_strict_score != our_score:
            disagree_strict.append({
                "doc_id": doc_id,
                "lm_strict": lm_strict_score,
                "ours": our_score,
                "ref": ref_num,
                "lm_extracted": strict_rec["filtered_resps"][0],
                "our_strict": extract_gsm8k_answer_strict(raw_response),
                "our_flexible": extract_gsm8k_answer_flexible(raw_response),
            })

    print(f"\nTotal samples: {n}")
    print(f"lm-eval flexible-extract accuracy: {lm_flex_correct/n:.4f} ({int(lm_flex_correct)}/{n})")
    print(f"lm-eval strict-match accuracy:     {lm_strict_correct/n:.4f} ({int(lm_strict_correct)}/{n})")
    print(f"is_correct() accuracy:             {our_correct/n:.4f} ({int(our_correct)}/{n})")

    print(f"\nDisagreements vs flexible-extract: {len(disagree_flex)}")
    for d in disagree_flex[:10]:
        print(f"  doc_id={d['doc_id']}: lm_flex={d['lm_flex']} ours={d['ours']} "
              f"ref={d['ref']} lm_extracted='{d['lm_extracted']}' "
              f"our_strict={d['our_strict']} our_flexible={d['our_flexible']}")
    if len(disagree_flex) > 10:
        print(f"  ... and {len(disagree_flex) - 10} more")

    print(f"\nDisagreements vs strict-match: {len(disagree_strict)}")
    for d in disagree_strict[:10]:
        print(f"  doc_id={d['doc_id']}: lm_strict={d['lm_strict']} ours={d['ours']} "
              f"ref={d['ref']} lm_extracted='{d['lm_extracted']}' "
              f"our_strict={d['our_strict']} our_flexible={d['our_flexible']}")
    if len(disagree_strict) > 10:
        print(f"  ... and {len(disagree_strict) - 10} more")


if __name__ == "__main__":
    for model, fname in MODELS.items():
        compare_model(model, fname)
