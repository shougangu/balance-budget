# ABOUTME: Single-turn constraint-style SFT mix for instruction following (IFEval/IFBench targets).
# ABOUTME: Merges verified Argilla IFEval-like data with both Nemotron v3 IF subsets.

# The mix holds five properties, each enforced in one place:
#   - Every row is one turn carrying the IF system message and the name of its source
#     (`_row`, `persona_rows`, `synthetic_rows`).
#   - A row survives only if its response satisfies every constraint the prompt declares
#     (`response_satisfies_constraints`).
#   - That verification is reproducible: checkers draw from the module RNG, so each is
#     built under a pinned seed and the stream is restored afterwards, leaving the result
#     independent of both cluster and row order (`_built_checker`, `PROBE_SEEDS`).
#   - A constraint whose arguments the checker will not take as given is dropped rather
#     than counted as passing (`_instruction_kwargs`, the `accepted` check).
#   - Duplicate prompts are dropped per source, so a prompt two sources both carry keeps
#     both responses (`_dedupe`), and the test split holds out only rows nothing in train
#     resembles, at each source's share of the mix (`split_rows`, `source_quotas`).

import inspect
import json
import random

from datasets import Dataset, DatasetDict, load_dataset

from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
from tuning.data.hf_dataset import HFDataset
from tuning.data.near_duplicates import holdout_without_near_duplicates, normalize

ARGILLA_REPO = "argilla/ifeval-like-data"
# The repo's other config is an IFEval-passing subset of this one, carrying no prompt it
# does not already hold; every row is verified here anyway.
ARGILLA_CONFIG = "default"
NEMOTRON_REPO = "nvidia/Nemotron-SFT-Instruction-Following-Chat-v3"
NEMOTRON_FILE = "data/instruction_following.jsonl"

ARGILLA_SOURCE = "argilla-ifeval-like"
NEMOTRON_PERSONA_SOURCE = "allenai/tulu-3-sft-personas-instruction-following"
NEMOTRON_SYNTHETIC_SOURCE = "synthetic"

SHUFFLE_SEED = 42
TEST_SIZE = 500

# Jaccard over 5-word shingles. The mix is assembled from constraint templates, so
# prompts that differ only in one of their numbers score around here; holding such a
# pair across the split would score the model on text it was trained on.
NEAR_DUPLICATE_THRESHOLD = 0.7


def _row(prompt, response, source):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "source": source,
    }


def _turns(messages):
    """Yield (user, assistant) content pairs in order, skipping empty turns."""
    pending = None
    for message in messages:
        if message.get("role") == "user":
            pending = message.get("content")
        elif message.get("role") == "assistant" and pending is not None:
            yield pending, message.get("content")
            pending = None


def _seed_dataset(record):
    return (record.get("metadata") or {}).get("seed_dataset", "")


def persona_rows(records):
    """Nemotron rows seeded from Persona IF, deduplicated to one response per prompt.

    The split ships roughly four unfiltered samples per prompt; keeping all of them
    would weight prompts unevenly by how many samples they happen to carry.
    """
    rows = []
    seen = set()
    for record in records:
        if _seed_dataset(record) != NEMOTRON_PERSONA_SOURCE:
            continue
        for prompt, response in _turns(record["messages"]):
            if not prompt or not response or prompt in seen:
                break
            seen.add(prompt)
            rows.append(_row(prompt, response, NEMOTRON_PERSONA_SOURCE))
            break
    return rows


def synthetic_rows(records):
    """Nemotron synthetic conversations cut to their opening turn.

    Later user turns refer back to earlier answers, so only the first turn stands
    alone as a single-turn training example.
    """
    rows = []
    for record in records:
        if _seed_dataset(record) == NEMOTRON_PERSONA_SOURCE:
            continue
        for prompt, response in _turns(record["messages"]):
            if prompt and response:
                rows.append(_row(prompt, response, NEMOTRON_SYNTHETIC_SOURCE))
            break
    return rows


def _instruction_kwargs(instruction_ids, raw_kwargs):
    """Align per-instruction kwargs; the unfiltered config ships one shared mapping."""
    if isinstance(raw_kwargs, str):
        try:
            raw_kwargs = json.loads(raw_kwargs)
        except json.JSONDecodeError:
            raw_kwargs = None
    if isinstance(raw_kwargs, list):
        aligned = list(raw_kwargs) + [{}] * len(instruction_ids)
        return aligned[: len(instruction_ids)]
    if isinstance(raw_kwargs, dict):
        return [raw_kwargs] * len(instruction_ids)
    return [{}] * len(instruction_ids)


# A checker that dislikes one of its arguments quietly replaces it with a value drawn
# from the shared random module. Building twice under fixed, differing seeds exposes
# that: a constraint the data fully stated resolves the same way both times.
PROBE_SEEDS = (1, 999)


def _built_checker(checker_class, instruction_id, arguments, seed):
    """Instantiate a checker with the module RNG pinned, and report what it resolved to."""
    random.seed(seed)
    checker = checker_class(instruction_id)
    checker.build_description(**arguments)
    return checker, checker.get_instruction_args()


def response_satisfies_constraints(response, instruction_ids, kwargs_list):
    """True when every checkable constraint on the prompt is satisfied.

    Responses whose constraints cannot be instantiated are rejected rather than
    assumed correct, so an unverifiable row never enters the mix.
    """
    from ifrlvr.instructions_registry import FUNCTION_DICT

    # Checkers draw from the module RNG, so verifying a row would otherwise shift the
    # stream for every row after it and make the whole mix depend on its own order.
    rng_state = random.getstate()
    try:
        checked = 0
        for instruction_id, kwargs in zip(instruction_ids, kwargs_list):
            checker_class = FUNCTION_DICT.get(instruction_id)
            if checker_class is None:
                return False
            signature = inspect.signature(checker_class.build_description)
            accepted = [
                name
                for name, parameter in signature.parameters.items()
                if name != "self"
                and parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
            ]
            arguments = {
                name: value
                for name, value in (kwargs or {}).items()
                if name in accepted and value is not None
            }
            # A parameter the data never supplied has no constraint behind it to check.
            if len(arguments) != len(accepted):
                return False
            try:
                first, first_args = _built_checker(
                    checker_class, instruction_id, arguments, PROBE_SEEDS[0]
                )
                _, second_args = _built_checker(
                    checker_class, instruction_id, arguments, PROBE_SEEDS[1]
                )
                # The checker rejected what the data supplied and invented a replacement,
                # so the constraint it would test is not the one the prompt states.
                if first_args != second_args:
                    return False
                if not first.check_following(response):
                    return False
            except (IndexError, KeyError):
                # Container lookups on a malformed row, not a missing resource.
                return False
            except (LookupError, ImportError, OSError):
                # A checker missing a tokenizer or a package says nothing about the
                # response. Silently rejecting here once cost a cluster two thirds of
                # its verified rows, so an unusable environment must stop the build.
                raise
            except Exception:
                # Row values reach regexes and format strings unescaped, so a bad one
                # can surface as almost any error. Whatever it is, it describes the row.
                return False
            checked += 1
        return checked > 0
    finally:
        random.setstate(rng_state)


def argilla_rows(records):
    """Argilla IFEval-like rows whose response passes every constraint it declares."""
    rows = []
    for record in records:
        prompt = record.get("prompt") or record.get("instruction")
        response = record.get("response")
        if not prompt or not response:
            continue
        instruction_ids = record.get("instruction_id_list") or []
        kwargs_list = _instruction_kwargs(instruction_ids, record.get("kwargs"))
        if not response_satisfies_constraints(response, instruction_ids, kwargs_list):
            continue
        rows.append(_row(prompt, response, ARGILLA_SOURCE))
    return rows


def shuffle_rows(rows):
    """Interleave the sources under a pinned seed, independent of input order."""
    ordered = sorted(rows, key=lambda row: (row["source"], _row_key(row)))
    random.Random(SHUFFLE_SEED).shuffle(ordered)
    return ordered


def _row_key(row):
    messages = row.get("messages") or []
    return "".join(message.get("content") or "" for message in messages)


def _dedupe(rows):
    """Drop repeated prompts within a source, keeping the first occurrence.

    Prompts are compared under the same normalisation the near-duplicate check
    uses, so a prompt that recurs with only its punctuation changed counts as the
    repeat it is rather than reaching the mix twice.

    The same prompt may legitimately occur in more than one source with a
    different response. Keep those cross-source examples so an earlier-loaded
    component cannot silently erase a later component from the mix.
    """
    seen = set()
    unique = []
    for row in rows:
        prompt = row["messages"][1]["content"]
        key = (row["source"], normalize(prompt))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def source_quotas(rows, test_size):
    """Test-split places per source, in proportion to each source's share of the mix.

    Largest-remainder allocation, so the places sum to exactly `test_size`.
    """
    counts = {}
    for row in rows:
        counts[row["source"]] = counts.get(row["source"], 0) + 1

    exact = {source: test_size * count / len(rows) for source, count in counts.items()}
    quotas = {source: int(share) for source, share in exact.items()}
    order = sorted(exact, key=lambda source: (exact[source] - quotas[source], source), reverse=True)
    for source in order[: test_size - sum(quotas.values())]:
        quotas[source] += 1
    return quotas


def split_rows(rows, test_size=TEST_SIZE, threshold=NEAR_DUPLICATE_THRESHOLD):
    """Hold out rows that nothing else in the mix resembles.

    The sources are template-generated, so a random slice puts near-identical
    prompts on both sides. Candidates are drawn under a pinned seed and a row is
    held out only when no other row is within `threshold`; rejected candidates
    stay in train, so nothing is discarded. Sources are held to their share of the
    mix, so decontaminating the split does not also reweight it.
    """
    prompts = [row["messages"][1]["content"] for row in rows]
    sources = [row["source"] for row in rows]
    train_idx, test_idx = holdout_without_near_duplicates(
        prompts,
        test_size=test_size,
        threshold=threshold,
        seed=SHUFFLE_SEED,
        groups=sources,
        quotas=source_quotas(rows, test_size),
    )
    return DatasetDict(
        {
            "train": Dataset.from_list([rows[i] for i in train_idx]),
            "test": Dataset.from_list([rows[i] for i in test_idx]),
        }
    )


class IFMixSFT(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="ifmix")

    def _load_nemotron(self):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(NEMOTRON_REPO, NEMOTRON_FILE, repo_type="dataset")
        with open(path) as handle:
            for line in handle:
                yield json.loads(line)

    def load_sources(self):
        rows = []

        print(f"Loading {ARGILLA_REPO} (verifying every declared constraint)", flush=True)
        # Downloaded rather than streamed: a stream that dies mid-iteration ends the
        # loop silently, which cost this split two thirds of its rows on one cluster.
        records = load_dataset(ARGILLA_REPO, ARGILLA_CONFIG, split="train")
        verified = argilla_rows(records)
        print(f"  kept {len(verified):,} verified rows of {len(records):,}", flush=True)
        rows.extend(verified)

        # Streamed twice rather than materialised; the split is 3 GB of parsed JSON.
        print(f"Loading {NEMOTRON_REPO}", flush=True)
        persona = persona_rows(self._load_nemotron())
        synthetic = synthetic_rows(self._load_nemotron())
        print(f"  persona-seeded: kept {len(persona):,} rows (deduplicated)", flush=True)
        print(f"  synthetic: kept {len(synthetic):,} first-turn rows", flush=True)
        rows.extend(persona)
        rows.extend(synthetic)

        return rows

    def format_dataset(self):
        rows = _dedupe(self.load_sources())
        rows = shuffle_rows(rows)

        counts = {}
        for row in rows:
            counts[row["source"]] = counts.get(row["source"], 0) + 1
        print(f"IF mix composition: {counts}")

        formatted_dataset = split_rows(rows, test_size=min(TEST_SIZE, len(rows) - 1))
        print(f"IF Mix SFT Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    ifmix = IFMixSFT()
    ifmix.format_dataset()
    ifmix.clear_old_datasets(prefix="sft-ifmix")
    ifmix.save_dataset_to_disk(save_name="sft-ifmix")
