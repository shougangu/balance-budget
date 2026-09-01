# ABOUTME: Tests the HF-export fix-ups applied when an RL mark is banked: the export carries
# ABOUTME: its SFT parent's config so the eval venv's transformers reads the same architecture.

import json

from tuning.verl.export import carry_parent_config


def test_export_takes_the_parents_config_verbatim(tmp_path):
    """transformers 5 (verl venv) rewrites rope_theta as rope_parameters, which
    transformers 4 (eval venv) reads as the 10000 default: the parent's file is
    the one every other checkpoint in the lineage is evaluated with."""
    parent = tmp_path / "parent"
    parent.mkdir()
    parent_config = {"model_type": "qwen3", "rope_theta": 1000000, "torch_dtype": "bfloat16"}
    (parent / "config.json").write_text(json.dumps(parent_config))
    export = tmp_path / "export"
    export.mkdir()
    (export / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "rope_parameters": {"rope_theta": 1000000}, "dtype": "bfloat16",
    }))

    carry_parent_config(str(parent), str(export))

    assert json.loads((export / "config.json").read_text()) == parent_config
