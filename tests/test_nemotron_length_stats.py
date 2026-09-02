# ABOUTME: Tests for the Nemotron math corpus length measurement script.
# ABOUTME: Covers tool-free row classification across the v2/v3/v4 schemas, jsonl byte-range sharding, and summary stats.

import json

import numpy as np

from scripts.nemotron_length_stats import classify_row, iter_jsonl_shard, summarize_lengths


def _assistant(reasoning, content, tool_calls=None):
    return {"role": "assistant", "content": content, "reasoning_content": reasoning,
            "tool_calls": tool_calls}


def test_classify_row_keeps_plain_user_assistant_exchange():
    messages = [{"role": "user", "content": "Solve 2+2"}, _assistant("2 and 2", "$\\boxed{4}$")]
    assert classify_row([], messages) == "cot"


def test_classify_row_flags_tool_definitions_and_tool_calls_as_tir():
    tool = [{"type": "function", "function": {"name": "stateful_python_code_exec"}}]
    plain = [{"role": "user", "content": "Solve"}, _assistant("think", "$\\boxed{4}$")]
    assert classify_row(tool, plain) == "tir"
    called = [{"role": "user", "content": "Solve"},
              _assistant("think", "", tool_calls=[{"id": "c0", "type": "function",
                                                   "function": {"name": "x", "arguments": "{}"}}]),
              {"role": "tool", "content": "4", "tool_call_id": "c0"},
              _assistant("", "$\\boxed{4}$")]
    assert classify_row([], called) == "tir"


def test_classify_row_accepts_v3_messages_without_tool_call_keys_and_leading_system():
    messages = [{"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Solve"},
                {"role": "assistant", "content": "$\\boxed{4}$", "reasoning_content": "think"}]
    assert classify_row([], messages) == "cot"


def test_classify_row_marks_empty_reasoning_or_answer_as_other():
    no_reasoning = [{"role": "user", "content": "Solve"}, _assistant("", "$\\boxed{4}$")]
    assert classify_row([], no_reasoning) == "other"
    no_answer = [{"role": "user", "content": "Solve"}, _assistant("think", None)]
    assert classify_row([], no_answer) == "other"


def test_iter_jsonl_shard_covers_every_line_exactly_once(tmp_path):
    path = tmp_path / "rows.jsonl"
    lines = [json.dumps({"i": i, "pad": "x" * (i * 7 % 50)}) for i in range(200)]
    path.write_text("\n".join(lines) + "\n")
    seen = []
    for shard in range(3):
        seen.extend(row["i"] for row in iter_jsonl_shard(path, shard, 3))
    assert sorted(seen) == list(range(200))


def test_iter_jsonl_shard_handles_missing_trailing_newline(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps({"i": i}) for i in range(5)))
    seen = [row["i"] for shard in range(2) for row in iter_jsonl_shard(path, shard, 2)]
    assert sorted(seen) == [0, 1, 2, 3, 4]


def test_summarize_lengths_reports_percentiles_and_cap_retention():
    lengths = np.arange(1, 10001)
    summary = summarize_lengths(lengths, caps=(5000,))
    assert summary["count"] == 10000
    assert summary["mean"] == 5000.5
    assert summary["median"] == 5000.5
    assert abs(summary["p90"] - 9000.1) < 1
    assert abs(summary["p99.99"] - 9999) < 1.5
    assert summary["max"] == 10000
    assert summary["total_tokens"] == 50005000
    assert summary["rows_within"][5000] == 0.5
    assert abs(summary["tokens_within"][5000] - (5000 * 5001 / 2) / 50005000) < 1e-9


def test_shard_files_keep_shard_suffix_and_reload_together(tmp_path):
    from scripts.nemotron_length_stats import ShardMeasurer, load_corpus

    class CountingTokenizer:
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [t.split() for t in texts]}

    rows = [{"problem": "p", "tools": [], "data_source": "aops",
             "messages": [{"role": "user", "content": "p"},
                          _assistant("a b c", "d")]}]
    for shard in range(2):
        measurer = ShardMeasurer(CountingTokenizer())
        measurer.add_batch(rows)
        measurer.write(tmp_path, "v2-low", shard, 2)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "v2-low.shard00-of-02.counts.json", "v2-low.shard00-of-02.parquet",
        "v2-low.shard01-of-02.counts.json", "v2-low.shard01-of-02.parquet"]
    table, meta, expected, found = load_corpus(tmp_path, "v2-low")
    assert (expected, found) == (2, 2)
    assert table.num_rows == 2
    assert meta["counts"]["cot"] == 2
    assert table.column("reasoning_tokens").to_pylist() == [3, 3]


def test_corpus_report_breaks_response_lengths_down_by_source():
    import pyarrow as pa
    from collections import Counter
    from scripts.nemotron_length_stats import corpus_report

    table = pa.table({
        "source": ["aops", "aops", "stackexchange"],
        "label": ["", "", ""],
        "problem_hash": [1, 1, 2],
        "problem_chars": [10, 10, 10], "reasoning_chars": [10, 10, 10], "answer_chars": [5, 5, 5],
        "problem_tokens": [4, 4, 4], "reasoning_tokens": [100, 200, 30], "answer_tokens": [10, 10, 10],
    })
    meta = {"counts": Counter({"rows": 3, "cot": 3}), "label_crosstab": Counter(), "raw_sources": Counter()}
    report = corpus_report(table, meta)
    assert report["by_source"]["aops"]["rows"] == 2
    assert report["by_source"]["aops"]["problems"] == 1
    assert report["by_source"]["aops"]["response_tokens"]["mean"] == 160
    assert report["by_source"]["stackexchange"]["response_tokens"]["median"] == 40
