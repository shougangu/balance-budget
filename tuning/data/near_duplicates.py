# ABOUTME: Near-duplicate detection over prompt text via shingling, MinHash and LSH banding.
# ABOUTME: Backs both dataset overlap auditing and decontaminated train/test splitting.

import random
import re

import numpy as np
import xxhash

SHINGLE_SIZE = 5
NUM_PERM = 128
BANDS = 32

_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
_SPACES = re.compile(r"\s+")

_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def normalize(text):
    """Case-folded, punctuation-free form used for all comparisons.

    Digits are kept: they are the whole content of a maths prompt, and a
    constraint prompt long enough to matter still overlaps heavily when only
    one of its numbers changes.
    """
    return _SPACES.sub(" ", _NON_ALNUM.sub(" ", text.lower())).strip()


def shingles(text, size=SHINGLE_SIZE):
    """Hashes of every `size`-word window, as the set representation of the text."""
    words = normalize(text).split()
    if not words:
        return np.empty(0, dtype=np.uint64)
    if len(words) <= size:
        windows = [" ".join(words)]
    else:
        windows = [" ".join(words[i : i + size]) for i in range(len(words) - size + 1)]
    # Not the builtin hash: it is salted per process, so signatures would differ
    # between the run that builds a split and the run that audits it.
    hashed = {np.uint64(xxhash.xxh64_intdigest(w) & _MAX_HASH) for w in windows}
    return np.sort(np.fromiter(hashed, dtype=np.uint64, count=len(hashed)))


def _permutations(num_perm, seed=1):
    # Pinned so a signature computed today matches one computed in a later run.
    rng = np.random.RandomState(seed)
    a = rng.randint(1, _MERSENNE_PRIME, size=num_perm, dtype=np.int64).astype(np.uint64)
    b = rng.randint(0, _MERSENNE_PRIME, size=num_perm, dtype=np.int64).astype(np.uint64)
    return a, b


_PERM_CACHE = {}


def _perms(num_perm):
    if num_perm not in _PERM_CACHE:
        _PERM_CACHE[num_perm] = _permutations(num_perm)
    return _PERM_CACHE[num_perm]


def signature(row_shingles, num_perm=NUM_PERM):
    """MinHash signature; equal-signature fraction estimates Jaccard similarity."""
    a, b = _perms(num_perm)
    if len(row_shingles) == 0:
        return np.full(num_perm, _MAX_HASH, dtype=np.uint64)
    hashed = (row_shingles[:, None] * a[None, :] + b[None, :]) % _MERSENNE_PRIME
    return (hashed & np.uint64(_MAX_HASH)).min(axis=0)


def exact_jaccard(left, right):
    """Exact Jaccard over the two shingle sets, not the MinHash estimate."""
    if len(left) == 0 and len(right) == 0:
        return 1.0
    if len(left) == 0 or len(right) == 0:
        return 0.0
    intersection = np.intersect1d(left, right, assume_unique=True).size
    union = len(left) + len(right) - intersection
    return intersection / union


class NearDuplicateIndex:
    """LSH over MinHash signatures: retrieves candidates sharing any band."""

    def __init__(self, num_perm=NUM_PERM, bands=BANDS):
        if num_perm % bands:
            raise ValueError(f"num_perm {num_perm} must be divisible by bands {bands}")
        self.num_perm = num_perm
        self.bands = bands
        self.rows_per_band = num_perm // bands
        self._buckets = [{} for _ in range(bands)]

    def _band_keys(self, sig):
        reshaped = sig.reshape(self.bands, self.rows_per_band)
        return [band.tobytes() for band in reshaped]

    def add(self, key, sig):
        for bucket, band_key in zip(self._buckets, self._band_keys(sig)):
            bucket.setdefault(band_key, []).append(key)

    def query(self, sig):
        found = set()
        for bucket, band_key in zip(self._buckets, self._band_keys(sig)):
            found.update(bucket.get(band_key, ()))
        return found


def nearest_neighbours(corpus, queries, exclude_self=False, num_perm=NUM_PERM, bands=BANDS):
    """For each query, the (similarity, corpus index) of its closest corpus entry.

    Similarity is exact Jaccard over shingles; LSH only narrows the candidates.
    """
    corpus_shingles = [shingles(text) for text in corpus]
    index = NearDuplicateIndex(num_perm=num_perm, bands=bands)
    for position, row_shingles in enumerate(corpus_shingles):
        index.add(position, signature(row_shingles, num_perm=num_perm))

    results = []
    for query_position, text in enumerate(queries):
        query_shingles = shingles(text)
        best_score, best_index = 0.0, -1
        for candidate in index.query(signature(query_shingles, num_perm=num_perm)):
            if exclude_self and candidate == query_position:
                continue
            score = exact_jaccard(query_shingles, corpus_shingles[candidate])
            if score > best_score:
                best_score, best_index = score, candidate
        results.append((best_score, best_index))
    return results


def holdout_without_near_duplicates(
    texts, test_size, threshold, seed=42, num_perm=NUM_PERM, bands=BANDS,
    groups=None, quotas=None,
):
    """Split indices so no held-out row has a near duplicate among the kept rows.

    Candidates are considered in a seeded order and a row joins the holdout only
    when nothing else in the corpus resembles it, so the rows it would have leaked
    from stay in train rather than being discarded.

    Passing `groups` (one label per row) with `quotas` (rows to hold out per label)
    stratifies the holdout, keeping each label's share of the test split fixed
    while the near-duplicate check stays global across every label.
    """
    if (groups is None) != (quotas is None):
        raise ValueError("groups and quotas must be given together")
    if quotas is not None:
        if sum(quotas.values()) != test_size:
            raise ValueError(
                f"quotas sum to {sum(quotas.values())}, expected test_size {test_size}"
            )
        if len(groups) != len(texts):
            raise ValueError(f"groups has {len(groups)} labels for {len(texts)} texts")

    all_shingles = [shingles(text) for text in texts]
    index = NearDuplicateIndex(num_perm=num_perm, bands=bands)
    signatures = []
    for position, row_shingles in enumerate(all_shingles):
        sig = signature(row_shingles, num_perm=num_perm)
        signatures.append(sig)
        index.add(position, sig)

    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)

    remaining = dict(quotas) if quotas else None
    test_idx = []
    for position in order:
        if len(test_idx) == test_size:
            break
        if remaining is not None and remaining.get(groups[position], 0) == 0:
            continue
        neighbours = index.query(signatures[position])
        if any(
            candidate != position
            and exact_jaccard(all_shingles[position], all_shingles[candidate]) >= threshold
            for candidate in neighbours
        ):
            continue
        if remaining is not None:
            remaining[groups[position]] -= 1
        test_idx.append(position)

    if len(test_idx) < test_size:
        short = (
            ", ".join(f"{label} short by {count}" for label, count in remaining.items() if count)
            if remaining
            else f"only {len(test_idx)} clean rows"
        )
        raise ValueError(
            f"{short}: no clean holdout of {test_size} at threshold {threshold}; "
            f"the corpus is too self-similar"
        )

    held = set(test_idx)
    train_idx = [position for position in range(len(texts)) if position not in held]
    return train_idx, sorted(test_idx)
