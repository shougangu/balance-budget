# ABOUTME: Measures problem/reasoning/answer token lengths of every tool-free row in the Nemotron
# ABOUTME: math corpora (v2 low/medium/high, v3, v4) and reports length percentiles and problem overlap.

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# corpus -> (HF repo, file format, files inside the repo)
CORPORA = {
    "v2-low": ("nvidia/Nemotron-Math-v2", "parquet", ["data/low.parquet"]),
    "v2-medium": ("nvidia/Nemotron-Math-v2", "parquet", ["data/medium.parquet"]),
    "v2-high": ("nvidia/Nemotron-Math-v2", "parquet",
                [f"data/high_part{i:02d}.parquet" for i in range(3)]),
    "v3": ("nvidia/Nemotron-SFT-Math-v3", "jsonl", ["data/train.jsonl"]),
    "v4": ("nvidia/Nemotron-SFT-Math-v4", "parquet",
           [f"data/train-{i:05d}-of-00012.parquet" for i in range(12)]),
}
TOKENIZER_NAME = "unsloth/Meta-Llama-3.1-8B"
CAPS = (4096, 8192, 16384, 32768, 65536)
PERCENTILES = (50, 90, 95, 99, 99.99)
DEFAULT_OUT = Path("/project/6105902/shougan/balance-budget/outputs/nemotron_length_stats")
BATCH_ROWS = 2000
# Per-row corpus label columns, checked against our own classification.
SOURCE_COLUMNS = ("data_source", "source")
LABEL_COLUMNS = ("tool_usage", "subset")

# Reasoning-effort x tool-use configurations the v2 teacher was sampled under, 8 tries each.
REGIMES = tuple(f"{effort}_{tool}" for effort in ("low", "medium", "high")
                for tool in ("no_tool", "with_tool"))
V2_TIERS = ("v2-low", "v2-medium", "v2-high")

ROW_SCHEMA = pa.schema([
    ("problem_hash", pa.int64()),
    ("uuid_hash", pa.int64()),
    ("has_tools", pa.bool_()),
])

RECORD_SCHEMA = pa.schema([
    ("source", pa.string()),
    ("label", pa.string()),
    ("problem_hash", pa.int64()),
    ("problem_chars", pa.int32()),
    ("reasoning_chars", pa.int32()),
    ("answer_chars", pa.int32()),
    ("problem_tokens", pa.int32()),
    ("reasoning_tokens", pa.int32()),
    ("answer_tokens", pa.int32()),
])


def classify_row(tools, messages):
    """'cot' for a plain user -> assistant exchange with reasoning and an answer, 'tir' when
    Python tools are defined or called, 'other' for every remaining shape."""
    if tools:
        return "tir"
    turns = list(messages)
    if turns and turns[0].get("role") == "system":
        turns = turns[1:]
    if any(m.get("role") == "tool" or m.get("tool_calls") for m in turns):
        return "tir"
    if [m.get("role") for m in turns] != ["user", "assistant"]:
        return "other"
    assistant = turns[1]
    if not (assistant.get("reasoning_content") or "").strip():
        return "other"
    if not (assistant.get("content") or "").strip():
        return "other"
    return "cot"


def normalize_source(raw):
    return "aops" if "aops" in (raw or "").lower() else "stackexchange"


def problem_hash(problem):
    digest = hashlib.blake2b(problem.strip().encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=True)


def iter_jsonl_shard(path, shard, num_shards):
    """Yield the rows whose line starts inside this shard's byte range of the file."""
    size = os.path.getsize(path)
    start = size * shard // num_shards
    end = size * (shard + 1) // num_shards
    with open(path, "rb") as f:
        if start > 0:
            f.seek(start - 1)
            if f.read(1) != b"\n":
                f.readline()
        while f.tell() < end:
            line = f.readline()
            if not line:
                break
            if line.strip():
                yield json.loads(line)


def iter_parquet_shard(paths, shard, num_shards, columns):
    """Yield row-group batches (as lists of dicts) round-robined across shards."""
    group_index = 0
    for path in paths:
        reader = pq.ParquetFile(path)
        available = [c for c in columns if c in reader.schema_arrow.names]
        for g in range(reader.metadata.num_row_groups):
            if group_index % num_shards == shard:
                yield reader.read_row_group(g, columns=available).to_pylist()
            group_index += 1


def batched(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def resolve_local_files(corpus):
    from huggingface_hub import hf_hub_download

    repo, fmt, files = CORPORA[corpus]
    return fmt, [hf_hub_download(repo, f, repo_type="dataset", local_files_only=True) for f in files]


def iter_corpus_batches_columns(corpus, shard, num_shards, columns):
    fmt, paths = resolve_local_files(corpus)
    if fmt == "parquet":
        yield from iter_parquet_shard(paths, shard, num_shards, columns)
    else:
        yield from batched(iter_jsonl_shard(paths[0], shard, num_shards), BATCH_ROWS)


def iter_corpus_batches(corpus, shard, num_shards):
    columns = ["problem", "messages", "tools", *SOURCE_COLUMNS, *LABEL_COLUMNS]
    yield from iter_corpus_batches_columns(corpus, shard, num_shards, columns)


def first_present(row, keys):
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return ""


def counts_path(parquet_path):
    """Row-count sidecar of a shard parquet; shard names contain dots, so suffix swaps won't do."""
    return Path(str(parquet_path)[: -len(".parquet")] + ".counts.json")


def token_lengths(tokenizer, texts):
    if not texts:
        return []
    return [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]


class ShardMeasurer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.columns = {name: [] for name in RECORD_SCHEMA.names}
        self.counts = Counter()
        self.label_crosstab = Counter()
        self.raw_sources = Counter()

    def add_batch(self, rows):
        problems, reasonings, answers = [], [], []
        for row in rows:
            kind = classify_row(row.get("tools"), row["messages"])
            self.counts["rows"] += 1
            self.counts[kind] += 1
            raw_label = first_present(row, LABEL_COLUMNS)
            self.label_crosstab[f"{raw_label}|{kind}"] += 1
            if kind != "cot":
                continue
            turns = row["messages"]
            assistant = turns[-1]
            raw_source = first_present(row, SOURCE_COLUMNS)
            self.raw_sources[raw_source] += 1
            self.columns["source"].append(normalize_source(raw_source))
            self.columns["label"].append(raw_label)
            self.columns["problem_hash"].append(problem_hash(row["problem"]))
            problems.append(row["problem"])
            reasonings.append(assistant["reasoning_content"])
            answers.append(assistant["content"])
        for name, texts in (("problem", problems), ("reasoning", reasonings), ("answer", answers)):
            lengths = token_lengths(self.tokenizer, texts)
            self.columns[f"{name}_chars"].extend(len(t) for t in texts)
            self.columns[f"{name}_tokens"].extend(lengths)
            self.counts["tokens"] += sum(lengths)

    def write(self, out_dir, corpus, shard, num_shards):
        stem = out_dir / f"{corpus}.shard{shard:02d}-of-{num_shards:02d}"
        table = pa.table({name: pa.array(values, type=RECORD_SCHEMA.field(name).type)
                          for name, values in self.columns.items()})
        pq.write_table(table, f"{stem}.parquet")
        with open(counts_path(f"{stem}.parquet"), "w") as f:
            json.dump({"counts": dict(self.counts),
                       "label_crosstab": dict(self.label_crosstab),
                       "raw_sources": dict(self.raw_sources)}, f, indent=2)
        return stem


def measure(corpus, shard, num_shards, out_dir, tokenizer_name, limit_batches=None):
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    measurer = ShardMeasurer(tokenizer)
    started = time.time()
    for i, batch in enumerate(iter_corpus_batches(corpus, shard, num_shards)):
        if limit_batches is not None and i >= limit_batches:
            break
        measurer.add_batch(batch)
        if i % 50 == 0:
            elapsed = time.time() - started
            print(f"[{corpus} shard {shard}/{num_shards}] batches={i + 1} {dict(measurer.counts)} "
                  f"{elapsed:.0f}s {measurer.counts['tokens'] / max(elapsed, 1e-9) / 1e6:.2f} Mtok/s",
                  flush=True)
    stem = measurer.write(out_dir, corpus, shard, num_shards)
    print(f"[{corpus} shard {shard}/{num_shards}] done {dict(measurer.counts)} "
          f"in {time.time() - started:.0f}s -> {stem}.parquet")


def flatten_pass_rates(metadata):
    """Per-regime (count, verified pass) pairs out of a v2 row's problem-level metadata struct."""
    flat = {}
    for regime in REGIMES:
        entry = (metadata or {}).get(f"reason_{regime}") or {}
        flat[f"{regime}_count"] = int(entry.get("count") or 0)
        flat[f"{regime}_pass"] = int(entry.get("pass") or 0)
    return flat


def tier_agreement(kept, verified):
    """Compare solutions a tier actually ships per problem against solutions it verified."""
    agreement = Counter()
    for problem, count in kept.items():
        if count > 0:
            agreement["problems_kept"] += 1
            passed = verified.get(problem, 0)
            if passed == 0:
                agreement["kept_but_not_verified"] += 1
            elif count == passed:
                agreement["rows_equal_pass"] += 1
            elif count > passed:
                agreement["rows_exceed_pass"] += 1
            else:
                agreement["rows_below_pass"] += 1
    for problem, passed in verified.items():
        if passed > 0:
            agreement["problems_verified"] += 1
            if kept.get(problem, 0) == 0:
                agreement["verified_but_not_kept"] += 1
    return agreement


def measure_problems(corpus, shard, num_shards, out_dir):
    """Light pass over one corpus shard: per-row identity and per-problem verification counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    problem_hashes, uuid_hashes, has_tools = [], [], []
    pass_rates = {}
    for batch in iter_corpus_batches_columns(
            corpus, shard, num_shards, ["uuid", "problem", "tools", "metadata"]):
        for row in batch:
            ph = problem_hash(row["problem"])
            problem_hashes.append(ph)
            uuid_hashes.append(problem_hash(row.get("uuid") or ""))
            has_tools.append(bool(row["tools"]))
            if ph not in pass_rates:
                pass_rates[ph] = flatten_pass_rates(row.get("metadata"))
    rows = pa.table({"problem_hash": pa.array(problem_hashes, pa.int64()),
                     "uuid_hash": pa.array(uuid_hashes, pa.int64()),
                     "has_tools": pa.array(has_tools, pa.bool_())})
    stem = out_dir / f"{corpus}.rows.shard{shard:02d}-of-{num_shards:02d}"
    pq.write_table(rows, f"{stem}.parquet")
    columns = {"problem_hash": pa.array(list(pass_rates), pa.int64())}
    for field in [f"{r}_{k}" for r in REGIMES for k in ("count", "pass")]:
        columns[field] = pa.array([v[field] for v in pass_rates.values()], pa.int32())
    stem = out_dir / f"{corpus}.passrates.shard{shard:02d}-of-{num_shards:02d}"
    pq.write_table(pa.table(columns), f"{stem}.parquet")
    print(f"[{corpus} problems shard {shard}/{num_shards}] rows={len(problem_hashes)} "
          f"problems={len(pass_rates)} in {time.time() - started:.0f}s -> {stem}.parquet")


def summarize_lengths(lengths, caps=CAPS):
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    summary = {
        "count": int(lengths.size),
        "mean": float(lengths.mean()),
        "median": float(np.percentile(lengths, 50)),
        "max": int(lengths.max()),
        "total_tokens": total,
        "rows_within": {},
        "tokens_within": {},
    }
    for p in PERCENTILES:
        summary[f"p{p:g}"] = float(np.percentile(lengths, p))
    for cap in caps:
        within = lengths <= cap
        summary["rows_within"][cap] = float(within.mean())
        summary["tokens_within"][cap] = float(lengths[within].sum() / total)
    return summary


def load_corpus(out_dir, corpus):
    """Concatenate every shard of a corpus; returns (table, counts, expected_shards, found_shards)."""
    parts = sorted(out_dir.glob(f"{corpus}.shard*-of-*.parquet"))
    if not parts:
        return None, None, 0, 0
    expected = int(parts[0].stem.split("-of-")[1])
    table = pa.concat_tables([pq.read_table(p) for p in parts])
    counts, crosstab, sources = Counter(), Counter(), Counter()
    for p in parts:
        with open(counts_path(p)) as f:
            payload = json.load(f)
        counts.update(payload["counts"])
        crosstab.update(payload["label_crosstab"])
        sources.update(payload["raw_sources"])
    return table, {"counts": counts, "label_crosstab": crosstab, "raw_sources": sources}, \
        expected, len(parts)


def corpus_report(table, meta):
    cols = {name: table.column(name).to_numpy() for name in table.column_names
            if name not in ("source", "label")}
    source = np.asarray(table.column("source").to_pylist())
    response = cols["reasoning_tokens"] + cols["answer_tokens"]
    sequence = cols["problem_tokens"] + response
    hashes = cols["problem_hash"]
    unique_problems = np.unique(hashes)
    aops_rows = source == "aops"
    aops_problems = np.unique(hashes[aops_rows])
    counts = meta["counts"]
    by_source = {}
    for name in ("aops", "stackexchange"):
        rows = source == name
        if rows.any():
            by_source[name] = {
                "rows": int(rows.sum()),
                "problems": int(np.unique(hashes[rows]).size),
                "response_tokens": summarize_lengths(response[rows], caps=()),
            }
    return {
        "rows_total": counts["rows"],
        "rows_cot": counts.get("cot", 0),
        "rows_tir": counts.get("tir", 0),
        "rows_other": counts.get("other", 0),
        "label_crosstab": dict(meta["label_crosstab"]),
        "raw_sources": dict(meta["raw_sources"]),
        "unique_problems": int(unique_problems.size),
        "solutions_per_problem": float(hashes.size / unique_problems.size),
        "aops_row_share": float(aops_rows.mean()),
        "aops_problem_share": float(aops_problems.size / unique_problems.size),
        "by_source": by_source,
        "problem_tokens": summarize_lengths(cols["problem_tokens"], caps=()),
        "answer_tokens": summarize_lengths(cols["answer_tokens"], caps=()),
        "response_tokens": summarize_lengths(response),
        "sequence_tokens": summarize_lengths(sequence),
        "chars_per_token": float((cols["reasoning_chars"].sum() + cols["answer_chars"].sum())
                                 / response.sum()),
        "_hashes": unique_problems,
    }


def overlap_report(reports):
    """Pairwise shared-problem counts across corpora, plus each corpus against the v2 union."""
    sets = {name: r["_hashes"] for name, r in reports.items()}
    v2 = [s for name, s in sets.items() if name.startswith("v2-")]
    if v2:
        sets["v2-all"] = np.unique(np.concatenate(v2))
    names = list(sets)
    matrix = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            matrix[a][b] = int(np.intersect1d(sets[a], sets[b], assume_unique=True).size)
    return {"unique_problems": {n: int(s.size) for n, s in sets.items()}, "shared": matrix}


def fmt_k(value):
    return f"{value / 1000:,.1f}k" if value >= 1000 else f"{value:,.0f}"


def render_markdown(reports, overlap):
    names = list(reports)
    lines = []
    lines.append("### Non-TIR rows, Llama-3.1 tokens (reasoning + answer)")
    header = "| corpus | rows (all) | non-TIR rows | unique problems | sols/problem | AoPS % (problems) | mean | median | p90 | p95 | p99 | p99.99 | max | total tokens |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for n in names:
        r = reports[n]
        s = r["response_tokens"]
        lines.append(
            f"| {n} | {r['rows_total']:,} | {r['rows_cot']:,} ({r['rows_cot'] / r['rows_total']:.0%}) | "
            f"{r['unique_problems']:,} | {r['solutions_per_problem']:.2f} | {r['aops_problem_share']:.0%} | "
            f"{s['mean']:,.0f} | {s['median']:,.0f} | {s['p90']:,.0f} | {s['p95']:,.0f} | "
            f"{s['p99']:,.0f} | {s['p99.99']:,.0f} | {s['max']:,} | {s['total_tokens'] / 1e9:.2f}B |")
    lines.append("")
    lines.append("### Same, split by problem source")
    header = "| corpus | source | rows | problems | mean | median | p90 | p95 | p99 | p99.99 |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for n in names:
        for src, b in reports[n]["by_source"].items():
            s = b["response_tokens"]
            lines.append(f"| {n} | {src} | {b['rows']:,} | {b['problems']:,} | {s['mean']:,.0f} | "
                         f"{s['median']:,.0f} | {s['p90']:,.0f} | {s['p95']:,.0f} | {s['p99']:,.0f} | "
                         f"{s['p99.99']:,.0f} |")
    lines.append("")
    lines.append("### Rows / tokens kept under a sequence cap (problem + reasoning + answer)")
    header = "| corpus | " + " | ".join(f"≤{c // 1024}k rows / tokens" for c in CAPS) + " | problem tok mean | answer tok mean | chars/token |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for n in names:
        r = reports[n]
        s = r["sequence_tokens"]
        cells = " | ".join(f"{s['rows_within'][c]:.1%} / {s['tokens_within'][c]:.1%}" for c in CAPS)
        lines.append(f"| {n} | {cells} | {r['problem_tokens']['mean']:.0f} | "
                     f"{r['answer_tokens']['mean']:.0f} | {r['chars_per_token']:.2f} |")
    lines.append("")
    lines.append("### Shared problems (exact match after strip), non-TIR rows")
    onames = list(overlap["unique_problems"])
    lines.append("| | " + " | ".join(onames) + " |")
    lines.append("|" + "---|" * (len(onames) + 1))
    for a in onames:
        lines.append(f"| {a} ({overlap['unique_problems'][a]:,}) | " +
                     " | ".join(f"{overlap['shared'][a][b]:,}" for b in onames) + " |")
    return "\n".join(lines)


def report(out_dir):
    reports = {}
    for corpus in CORPORA:
        table, meta, expected, found = load_corpus(out_dir, corpus)
        if table is None:
            print(f"[report] {corpus}: no shards yet, skipped")
            continue
        if found != expected:
            print(f"[report] WARNING {corpus}: {found}/{expected} shards present, numbers are partial")
        reports[corpus] = corpus_report(table, meta)
    overlap = overlap_report(reports)
    markdown = render_markdown(reports, overlap)
    print(markdown)
    payload = {name: {k: v for k, v in r.items() if k != "_hashes"} for name, r in reports.items()}
    payload["overlap"] = overlap
    with open(out_dir / "report.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    (out_dir / "report.md").write_text(markdown + "\n")


def load_tier_rows(out_dir, corpus):
    parts = sorted(out_dir.glob(f"{corpus}.rows.shard*-of-*.parquet"))
    if not parts:
        return None
    return pa.concat_tables([pq.read_table(p) for p in parts])


def load_pass_rates(out_dir, corpora):
    """Union of every corpus' per-problem verification counts, keyed by problem hash."""
    tables = []
    for corpus in corpora:
        tables += [pq.read_table(p)
                   for p in sorted(out_dir.glob(f"{corpus}.passrates.shard*-of-*.parquet"))]
    if not tables:
        return None
    table = pa.concat_tables(tables)
    hashes = table.column("problem_hash").to_numpy()
    _, first = np.unique(hashes, return_index=True)
    return table.take(pa.array(first))


def counts_by_problem(hashes):
    unique, counts = np.unique(hashes, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def kept_counts_by_mode(cot, all_rows):
    """Split a tier's shipped rows per problem into tool-free and tool-integrated.

    The `tools` column alone understates tool use — v2 rows call the Python tool without
    declaring it — so the tool-free counts come from the full message-shape classification.
    """
    with_tool = {}
    for problem, total in all_rows.items():
        remainder = total - cot.get(problem, 0)
        if remainder > 0:
            with_tool[problem] = remainder
    return dict(cot), with_tool


def tier_membership_report(out_dir):
    pass_rates = load_pass_rates(out_dir, V2_TIERS)
    if pass_rates is None:
        raise SystemExit("no pass-rate shards found; run `problems` for the v2 tiers first")
    verified = {regime: dict(zip(pass_rates.column("problem_hash").to_numpy().tolist(),
                                 pass_rates.column(f"{regime}_pass").to_numpy().tolist()))
                for regime in REGIMES}
    report = {"problems_in_pool": pass_rates.num_rows, "tiers": {}}
    for corpus in V2_TIERS:
        table = load_tier_rows(out_dir, corpus)
        if table is None:
            continue
        hashes = table.column("problem_hash").to_numpy()
        uuids = table.column("uuid_hash").to_numpy()
        effort = corpus.split("-")[1]
        lengths, _, _, _ = load_corpus(out_dir, corpus)
        if lengths is None:
            raise SystemExit(f"{corpus}: run `measure` first, tool-free counts come from it")
        cot = counts_by_problem(lengths.column("problem_hash").to_numpy())
        modes = dict(zip(("no_tool", "with_tool"),
                         kept_counts_by_mode(cot, counts_by_problem(hashes))))
        entry = {"rows": int(hashes.size),
                 "distinct_uuids": int(np.unique(uuids).size),
                 "distinct_problem_strings": int(np.unique(hashes).size)}
        for mode, kept in modes.items():
            entry[mode] = dict(tier_agreement(kept, verified[f"{effort}_{mode}"]))
            entry[mode]["rows"] = int(sum(kept.values()))
            entry[mode]["rows_per_problem"] = float(np.mean(list(kept.values()))) if kept else 0.0
        report["tiers"][corpus] = entry
    return report


def render_tier_markdown(report):
    lines = ["### Which v2 problems land in which effort tier",
             f"Problem pool across the three tier files: {report['problems_in_pool']:,} distinct "
             "problem strings (metadata is problem-level and identical across tiers).", ""]
    header = ("| tier | regime | problems verified (pass>0) | problems shipped | shipped rows | "
              "rows == verified passes | shipped w/o a verified pass | verified but absent | "
              "rows/problem |")
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for corpus, entry in report["tiers"].items():
        for mode in ("no_tool", "with_tool"):
            m = entry[mode]
            equal = m.get("rows_equal_pass", 0)
            shipped = m.get("problems_kept", 0)
            lines.append(
                f"| {corpus} | {mode} | {m.get('problems_verified', 0):,} | {shipped:,} | "
                f"{m['rows']:,} | {equal:,} ({equal / max(shipped, 1):.1%}) | "
                f"{m.get('kept_but_not_verified', 0):,} | {m.get('verified_but_not_kept', 0):,} | "
                f"{m['rows_per_problem']:.2f} |")
    lines.append("")
    lines.append("`uuid` identifies a trajectory, not a problem, so it cannot detect repeated "
                 "problem strings (and v2 high ships it empty).")
    header = "| tier | rows | distinct uuids | distinct problem strings |"
    lines += [header, "|" + "---|" * (header.count("|") - 1)]
    for corpus, entry in report["tiers"].items():
        lines.append(f"| {corpus} | {entry['rows']:,} | {entry['distinct_uuids']:,} | "
                     f"{entry['distinct_problem_strings']:,} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="Tokenize one shard of one corpus")
    m.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    m.add_argument("--shard", type=int, default=0)
    m.add_argument("--num-shards", type=int, default=1)
    m.add_argument("--out", type=Path, default=DEFAULT_OUT)
    m.add_argument("--tokenizer", default=TOKENIZER_NAME)
    m.add_argument("--limit-batches", type=int, default=None, help="Smoke-test cutoff")
    p = sub.add_parser("problems", help="Read one shard's per-problem verification counts")
    p.add_argument("--corpus", required=True, choices=sorted(CORPORA))
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    r = sub.add_parser("report", help="Summarize every measured shard")
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    t = sub.add_parser("tiers", help="Explain which problems land in which v2 effort tier")
    t.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.command == "measure":
        measure(args.corpus, args.shard, args.num_shards, args.out, args.tokenizer,
                args.limit_batches)
    elif args.command == "problems":
        measure_problems(args.corpus, args.shard, args.num_shards, args.out)
    elif args.command == "tiers":
        tier_report = tier_membership_report(args.out)
        markdown = render_tier_markdown(tier_report)
        print(markdown)
        with open(args.out / "tiers.json", "w") as f:
            json.dump(tier_report, f, indent=2, default=str)
        (args.out / "tiers.md").write_text(markdown + "\n")
    else:
        report(args.out)


if __name__ == "__main__":
    main()
