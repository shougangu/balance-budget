# ABOUTME: Tests that an SFT resume picks the newest checkpoint that finished writing,
# ABOUTME: skipping a directory a wall-time kill cut off part-way through the save.

from tuning.training.sft_training import latest_complete_checkpoint


def test_resume_skips_a_checkpoint_cut_off_mid_save(tmp_path):
    """The Trainer writes trainer_state.json last, so a directory without it holds
    a partial save; resuming from it raises FileNotFoundError inside the Trainer."""
    complete = tmp_path / "checkpoint-60"
    complete.mkdir()
    (complete / "trainer_state.json").write_text("{}")
    torn = tmp_path / "checkpoint-80"
    torn.mkdir()
    (torn / "config.json").write_text("{}")

    assert latest_complete_checkpoint(str(tmp_path)) == str(complete)


def test_resume_picks_the_highest_complete_checkpoint(tmp_path):
    for step in (20, 100, 60):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text("{}")

    assert latest_complete_checkpoint(str(tmp_path)).endswith("checkpoint-100")


def test_resume_returns_none_without_a_complete_checkpoint(tmp_path):
    (tmp_path / "checkpoint-80").mkdir()

    assert latest_complete_checkpoint(str(tmp_path)) is None
    assert latest_complete_checkpoint(str(tmp_path / "missing")) is None
