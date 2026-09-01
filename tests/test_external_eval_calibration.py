# ABOUTME: Tests for scripts/external_eval_calibration.py: prompt remapping for the
# ABOUTME: paper-protocol benchmarks and serving LoRA adapter checkpoints.

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import external_eval_calibration as cal  # noqa: E402


def test_paper_benchmarks_are_math_benchmarks():
    assert {"math", "aime24"} <= set(cal.MATH_BENCHMARKS)


def test_boxed_style_strips_wrapper_for_math():
    msgs = cal.build_messages("boxed", "math", "Problem: 1+1?\nAnswer:")
    assert msgs == [{"role": "user", "content": f"{cal.BOXED_INSTRUCTION}\n\n1+1?"}]


def test_build_strategy_math_and_aime24():
    assert cal.build_strategy("math", 1, [1], 2, strict={}).id == "math"
    assert cal.build_strategy("aime24", 1, [1], None, strict={}).id == "aime24"


def test_hard_benchmarks_are_math_benchmarks():
    assert {"aime25", "aime26", "hmmt_feb25", "olympiadbench"} <= set(cal.MATH_BENCHMARKS)


def test_build_strategy_hard_benchmarks():
    for benchmark in ("aime25", "aime26", "hmmt_feb25", "olympiadbench"):
        assert cal.build_strategy(benchmark, 1, [1], 2, strict={}).id == benchmark


def test_thirty_problem_sets_take_the_aime_sample_count():
    args = cal.parse_args([
        "--model", "m", "--model-family", "qwen3-8B",
        "--n-samples", "4", "--aime-n-samples", "32", "--out", "o.json",
    ])
    for benchmark in ("aime24", "aime25", "aime26", "hmmt_feb25"):
        assert cal.benchmark_n_samples(benchmark, args) == 32
    assert cal.benchmark_n_samples("math500", args) == 4
    # 581 problems: the shared sample count already gives a usable SE.
    assert cal.benchmark_n_samples("olympiadbench", args) == 4


def test_resolve_model_full_checkpoint(tmp_path):
    assert cal.resolve_model(str(tmp_path)) == (str(tmp_path), None, None)


def test_resolve_model_adapter_checkpoint(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Meta-Llama-3.1-8B", "r": 128}))
    assert cal.resolve_model(str(tmp_path)) == ("unsloth/Meta-Llama-3.1-8B", str(tmp_path), 128)


def test_canonical_answer_uses_last_boxed():
    assert cal.canonical_answer(r"so \boxed{1} no wait \boxed{\frac{1}{2}}") == \
        cal.canonical_answer(r"final: \boxed{0.5}")
    assert cal.canonical_answer("I give up") is None


def test_canonical_answer_falls_back_to_hash_marker():
    assert cal.canonical_answer("so the total is #### 42") == cal.canonical_answer(r"\boxed{42}")


def test_majority_vote_scores_takes_mode_of_first_k():
    results = [
        # k=1 votes the wrong first sample; k=4 has 2 right vs 1 wrong + 1 unparsable.
        {"prompt": "p1", "responses": [r"\boxed{3}", r"\boxed{2}", "nothing", r"\boxed{2}"]},
        # Ties break toward the earliest answer seen.
        {"prompt": "p2", "responses": [r"\boxed{7}", r"\boxed{8}", r"\boxed{8}", r"\boxed{7}"]},
    ]
    refs = {"p1": "2", "p2": "7"}
    scores = cal.majority_vote_scores(results, refs, k_values=[1, 2, 4, 8])
    assert scores == {"maj_at_1": 0.5, "maj_at_2": 0.5, "maj_at_4": 1.0}


def test_majority_vote_all_unparsable_is_wrong():
    results = [{"prompt": "p", "responses": ["a", "b"]}]
    assert cal.majority_vote_scores(results, {"p": "1"}, k_values=[2]) == {"maj_at_2": 0.0}


def _write_adapter(path, base, output_dir, grpo):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base, "r": 128}))
    config = {"output_dir": output_dir}
    if grpo:
        config["num_generations"] = 8
    (path / "training_config.json").write_text(json.dumps(config))


def test_sft_parent_of_grpo_adapter(tmp_path):
    parent = tmp_path / "l8b_960m_sft-1475072_oz7gxjiw"
    _write_adapter(parent, "unsloth/Meta-Llama-3.1-8B",
                   "/elsewhere/l8b_sft-openmath-14000000_oz7gxjiw", grpo=False)
    grpo = tmp_path / "l8b_3840m_sft-205568_71xr216b"
    _write_adapter(grpo, "unsloth/Meta-Llama-3.1-8B",
                   "/elsewhere/l8b_960m_sft-1475072_oz7gxjiw_rlvr-simplerl-25000_grpo_71xr216b",
                   grpo=True)
    assert cal.sft_parent_adapter(str(grpo)) == str(parent)
    assert cal.sft_parent_adapter(str(parent)) is None


def test_sft_parent_missing_dir_is_an_error(tmp_path):
    grpo = tmp_path / "grpo"
    _write_adapter(grpo, "base", "/x/orphan_rlvr-simplerl-25000_grpo_abc", grpo=True)
    with pytest.raises(FileNotFoundError):
        cal.sft_parent_adapter(str(grpo))


def test_resolve_model_merges_sft_parent_for_grpo_adapter(tmp_path, monkeypatch):
    parent = tmp_path / "l8b_960m_sft-1475072_oz7gxjiw"
    _write_adapter(parent, "unsloth/Meta-Llama-3.1-8B", "/elsewhere/sft", grpo=False)
    grpo = tmp_path / "l8b_3840m_sft-205568_71xr216b"
    _write_adapter(grpo, "unsloth/Meta-Llama-3.1-8B",
                   "/elsewhere/l8b_960m_sft-1475072_oz7gxjiw_rlvr-simplerl-25000_grpo_71xr216b",
                   grpo=True)
    merges = []
    monkeypatch.setattr(cal, "merge_adapter_into_base",
                        lambda base, adapter, out: merges.append((base, adapter, out)) or out)
    merged = tmp_path / "root" / parent.name
    assert cal.resolve_model(str(grpo), merge_root=str(tmp_path / "root")) == \
        (str(merged), str(grpo), 128)
    assert merges == [("unsloth/Meta-Llama-3.1-8B", str(parent), str(merged))]


def test_merge_reuses_a_completed_merge_dir(tmp_path):
    out = tmp_path / "merged"
    out.mkdir()
    (out / cal.MERGE_DONE_MARKER).touch()
    assert cal.merge_adapter_into_base("base", "adapter", str(out)) == str(out)


def test_ifbench_takes_its_own_sample_count():
    args = cal.parse_args([
        "--model", "m", "--model-family", "gemma3-12B", "--benchmarks", "ifeval,ifbench",
        "--n-samples", "4", "--ifbench-n-samples", "8", "--out", "o.json",
    ])
    assert cal.benchmark_n_samples("ifeval", args) == 4
    assert cal.benchmark_n_samples("ifbench", args) == 8


def test_ifbench_defaults_to_the_shared_sample_count():
    args = cal.parse_args(["--model", "m", "--model-family", "gemma3-12B",
                           "--n-samples", "4", "--out", "o.json"])
    assert cal.benchmark_n_samples("ifbench", args) == 4
    assert cal.benchmark_n_samples("amc", args) == 8

def test_merge_carries_the_processor_config_a_multimodal_base_needs(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "preprocessor_config.json").write_text('{"image_seq_length": 256}')
    base = tmp_path / "base"
    base.mkdir()
    (base / "preprocessor_config.json").write_text('{"image_seq_length": 999}')
    (base / "processor_config.json").write_text('{"processor_class": "Gemma3Processor"}')
    out = tmp_path / "merged"
    out.mkdir()
    cal.copy_processor_assets(str(adapter), str(base), str(out))
    assert (out / "processor_config.json").exists()
    assert json.loads((out / "preprocessor_config.json").read_text())["image_seq_length"] == 256


def test_merge_of_a_text_only_base_copies_nothing(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    out = tmp_path / "merged"
    out.mkdir()
    cal.copy_processor_assets(str(adapter), str(base), str(out))
    assert list(out.iterdir()) == []


def test_majority_vote_grades_the_winner_with_the_accept_either_verifier():
    """maj@k must use the same grader as pass@k, or the two columns disagree."""
    # The numeric path reads the 7 out of this box; math-verify alone rejects the \text{}.
    results = [{"prompt": "p", "responses": [r"so \\boxed{\\text{7 apples}}"] * 2}]
    assert cal.majority_vote_scores(results, {"p": "7"}, k_values=[2]) == {"maj_at_2": 1.0}
