# ABOUTME: Tests that claiming a checkpoint row never destroys the metadata file.
# ABOUTME: A malformed line must raise before the file is rewritten, leaving it intact.

import json

import pytest

from tuning.training.pipeline.checkpoint_metadata import (
    append_metadata_row,
    claim_checkpoint,
    claim_next_checkpoint,
)


def test_append_metadata_row_writes_exactly_one_line(tmp_path):
    path = tmp_path / "rows.json"
    append_metadata_row(str(path), {"checkpoint_path": "/ckpt/a", "total_minutes": 2.0})
    append_metadata_row(str(path), {"checkpoint_path": "/ckpt/b", "total_minutes": 5.0})
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["checkpoint_path"] == "/ckpt/b"


def test_appended_rows_are_claimable(tmp_path):
    path = tmp_path / "rows.json"
    append_metadata_row(str(path), {"checkpoint_path": "/ckpt/a"})
    claimed = claim_next_checkpoint(str(path))
    assert claimed["checkpoint_path"] == "/ckpt/a"
    assert claimed["claimed"] is True


def test_malformed_row_leaves_the_file_untouched(tmp_path):
    path = tmp_path / "rows.json"
    original = '{\n  "checkpoint_path": "/ckpt/a",\n  "claimed": false\n}\n'
    path.write_text(original)
    with pytest.raises(json.JSONDecodeError):
        claim_checkpoint(str(path), "/ckpt/a")
    assert path.read_text() == original


def test_claim_rewrites_only_the_matching_row(tmp_path):
    path = tmp_path / "rows.json"
    rows = [{"checkpoint_path": "/ckpt/a", "claimed": False}, {"checkpoint_path": "/ckpt/b", "claimed": False}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    claimed = claim_checkpoint(str(path), "/ckpt/b")
    assert claimed["claimed"] is True
    rewritten = [json.loads(l) for l in path.read_text().splitlines()]
    assert rewritten == [{"checkpoint_path": "/ckpt/a", "claimed": False}, {"checkpoint_path": "/ckpt/b", "claimed": True}]


def test_claim_checkpoint_reclaims_a_claimed_row(tmp_path):
    """A resumed worker re-claims its own pinned row; completion is checked by callers."""
    path = tmp_path / "rows.json"
    append_metadata_row(str(path), {"checkpoint_path": "/ckpt/a", "claimed": True})
    assert claim_checkpoint(str(path), "/ckpt/a")["checkpoint_path"] == "/ckpt/a"
    assert claim_checkpoint(str(path), "/ckpt/missing") is None


def test_mark_eval_submitted_flags_only_that_row(tmp_path):
    from tuning.training.pipeline.checkpoint_metadata import mark_eval_submitted

    path = tmp_path / "rows.json"
    append_metadata_row(str(path), {"checkpoint_path": "/ckpt/a"})
    append_metadata_row(str(path), {"checkpoint_path": "/ckpt/b"})
    mark_eval_submitted(str(path), "/ckpt/b")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0].get("eval_submitted") is None
    assert rows[1]["eval_submitted"] is True
