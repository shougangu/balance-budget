# ABOUTME: Fix-ups for the HF export verl writes when an RL mark is banked, so the export is
# ABOUTME: read by the eval venv exactly like the SFT marks of its lineage. No verl imports.

import os
import shutil


def carry_parent_config(parent_dir: str, export_dir: str) -> None:
    """Replace the export's config.json with the SFT parent's.

    RL changes weights only, so the parent's architecture config still
    describes the export. verl's venv (transformers 5) re-serializes it in a
    newer format (rope_parameters in place of rope_theta) that the eval venv's
    transformers 4 reads with default values, which silently degrades every
    RL score; the parent's file is what every other checkpoint in the lineage
    is evaluated with.
    """
    shutil.copyfile(os.path.join(parent_dir, "config.json"), os.path.join(export_dir, "config.json"))
