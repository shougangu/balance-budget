# ABOUTME: Tests for scripts/submit_eval_calibration.py: the arm/model grid it expands
# ABOUTME: into calibration jobs, including our own lineage checkpoints and maj@k arms.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import submit_eval_calibration as sub  # noqa: E402


def test_maj256_arm_samples_256_on_math500_and_64_elsewhere():
    flags = sub.MATH_ARMS["ours_maj256"]
    assert flags[flags.index("--n-samples") + 1] == "256"
    assert flags[flags.index("--amc-n-samples") + 1] == "64"
    assert flags[flags.index("--gsm8k-n-samples") + 1] == "64"
    assert flags[flags.index("--k-values") + 1:][:5] == ["1", "4", "16", "64", "256"]
    assert flags[flags.index("--template") + 1] == "simple"
    assert flags[flags.index("--prompt-style") + 1] == "ours"


def test_ours_greedy_arm_is_single_sample_greedy_on_our_protocol():
    flags = sub.MATH_ARMS["ours_greedy"]
    assert flags[flags.index("--temperature") + 1] == "0.0"
    assert flags[flags.index("--n-samples") + 1] == "1"
    assert flags[flags.index("--template") + 1] == "simple"


def test_lineage_checkpoints_are_adapter_dirs_of_the_llama_family():
    for key in ("l8b-64h-100", "l8b-64h-25"):
        model, family = sub.MODELS[key]
        assert family == "llama3-8B"
        assert model.startswith("tuning/models/llama3-8B_math500-p@1-3840m_sft-")


def test_smoke_cells_shrink_prompts_and_samples():
    cell = next(sub.cells(["l8b-64h-100"], ["math"], arms=["ours_maj256"], smoke=True))
    flags = cell[-1]
    assert flags[flags.index("--num-prompts") + 1] == "20"
    last_n = len(flags) - 1 - flags[::-1].index("--n-samples")
    assert flags[last_n + 1] == "8"
