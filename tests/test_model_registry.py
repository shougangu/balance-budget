# ABOUTME: Guards that every registered model appears across all registry maps, so a
# ABOUTME: model can't be half-registered (in HF map but missing a GPU or tier mapping).

from tuning.config import HF_MODEL_MAP, MODEL_CHAT_TEMPLATE_MAP
from tuning.training.pipeline.cli import (
    MODEL_TO_GPU_1,
    MODEL_TO_GPU_2,
    MODEL_TO_GPU_3,
)


def test_all_registry_maps_share_model_set():
    """Every registry map must cover exactly the same set of model names."""
    reference = set(HF_MODEL_MAP)
    for name, mapping in [
        ("MODEL_CHAT_TEMPLATE_MAP", MODEL_CHAT_TEMPLATE_MAP),
        ("MODEL_TO_GPU_1", MODEL_TO_GPU_1),
        ("MODEL_TO_GPU_2", MODEL_TO_GPU_2),
        ("MODEL_TO_GPU_3", MODEL_TO_GPU_3),
    ]:
        assert set(mapping) == reference, (
            f"{name} model set differs from HF_MODEL_MAP; "
            f"missing={reference - set(mapping)}, extra={set(mapping) - reference}"
        )


def test_qwen2_14b_fully_registered():
    """qwen2-14B and its instruct variant must be selectable and fully mapped."""
    for model in ["qwen2-14B", "qwen2-14B-instruct"]:
        assert model in HF_MODEL_MAP
        assert model in MODEL_CHAT_TEMPLATE_MAP
        assert model in MODEL_TO_GPU_1
        assert model in MODEL_TO_GPU_2
        assert model in MODEL_TO_GPU_3


def test_qwen3_base_models_fully_registered():
    """The long-CoT campaign lineages must be selectable and fully mapped."""
    expected_repos = {
        "qwen3-4B": "Qwen/Qwen3-4B-Base",
        "qwen3-8B": "Qwen/Qwen3-8B-Base",
        "qwen3-14B": "Qwen/Qwen3-14B-Base",
    }
    for model, repo in expected_repos.items():
        assert HF_MODEL_MAP.get(model) == repo
        assert model in MODEL_CHAT_TEMPLATE_MAP
        assert model in MODEL_TO_GPU_1
        assert model in MODEL_TO_GPU_2
        assert model in MODEL_TO_GPU_3
