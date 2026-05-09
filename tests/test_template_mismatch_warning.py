# ABOUTME: Verifies warn_if_template_mismatch flags --simple-template/SFT-checkpoint
# ABOUTME: disagreement before vLLM-backed post-training silently uses the wrong template.

import json
import warnings
from pathlib import Path

import pytest

from tuning.utils.utils import SIMPLE_TEMPLATE, warn_if_template_mismatch


CHATML_TEMPLATE = (
    "{% if 'role' in messages[0] %}{% for message in messages %}"
    "{% if message['role'] == 'user' %}{{'<|im_start|>user\n' + message['content'] + '<|im_end|>\n'}}"
    "{% elif message['role'] == 'assistant' %}{{'<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{% else %}{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}{% endif %}"
)


def _write_jinja(checkpoint_dir: Path, template: str) -> None:
    (checkpoint_dir / "chat_template.jinja").write_text(template)


def _write_tokenizer_config(checkpoint_dir: Path, template: str | None) -> None:
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast"}
    if template is not None:
        cfg["chat_template"] = template
    (checkpoint_dir / "tokenizer_config.json").write_text(json.dumps(cfg))


def _capture(simple_requested, checkpoint_dir):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_template_mismatch(str(checkpoint_dir), simple_requested)
    return [str(w.message) for w in caught]


def test_simple_requested_simple_on_disk_no_warning(tmp_path):
    _write_jinja(tmp_path, SIMPLE_TEMPLATE)
    assert _capture(simple_requested=True, checkpoint_dir=tmp_path) == []


def test_simple_requested_chatml_on_disk_warns(tmp_path):
    _write_jinja(tmp_path, CHATML_TEMPLATE)
    msgs = _capture(simple_requested=True, checkpoint_dir=tmp_path)
    assert len(msgs) == 1
    assert "simple" in msgs[0].lower()
    assert str(tmp_path) in msgs[0]


def test_simple_not_requested_chatml_on_disk_no_warning(tmp_path):
    _write_jinja(tmp_path, CHATML_TEMPLATE)
    assert _capture(simple_requested=False, checkpoint_dir=tmp_path) == []


def test_simple_not_requested_simple_on_disk_warns(tmp_path):
    _write_jinja(tmp_path, SIMPLE_TEMPLATE)
    msgs = _capture(simple_requested=False, checkpoint_dir=tmp_path)
    assert len(msgs) == 1
    assert "simple" in msgs[0].lower()
    assert str(tmp_path) in msgs[0]


def test_falls_back_to_tokenizer_config_when_jinja_missing(tmp_path):
    _write_tokenizer_config(tmp_path, CHATML_TEMPLATE)
    msgs = _capture(simple_requested=True, checkpoint_dir=tmp_path)
    assert len(msgs) == 1
    assert "simple" in msgs[0].lower()


def test_jinja_takes_precedence_over_tokenizer_config(tmp_path):
    _write_jinja(tmp_path, SIMPLE_TEMPLATE)
    _write_tokenizer_config(tmp_path, CHATML_TEMPLATE)
    assert _capture(simple_requested=True, checkpoint_dir=tmp_path) == []


def test_no_template_files_warns_unverifiable(tmp_path):
    msgs = _capture(simple_requested=True, checkpoint_dir=tmp_path)
    assert len(msgs) == 1
    assert "could not verify" in msgs[0].lower() or "no chat_template" in msgs[0].lower()


def test_missing_checkpoint_dir_warns_unverifiable(tmp_path):
    missing = tmp_path / "does_not_exist"
    msgs = _capture(simple_requested=True, checkpoint_dir=missing)
    assert len(msgs) == 1


def test_whitespace_differences_dont_trigger_false_positive(tmp_path):
    _write_jinja(tmp_path, SIMPLE_TEMPLATE + "\n\n")
    assert _capture(simple_requested=True, checkpoint_dir=tmp_path) == []
