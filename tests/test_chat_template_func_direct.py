import sys
import types

from tuning.utils import utils


def test_chat_template_func_direct_import_and_apply(monkeypatch):
    calls = []

    def fake_get_chat_template(tokenizer, chat_template, mapping, map_eos_token):
        calls.append(
            {
                "chat_template": chat_template,
                "mapping": mapping,
                "map_eos_token": map_eos_token,
            }
        )
        tokenizer.chat_template = f"applied:{chat_template}"
        return tokenizer

    fake_unsloth = types.ModuleType("unsloth")
    fake_chat_templates = types.ModuleType("unsloth.chat_templates")
    fake_chat_templates.get_chat_template = fake_get_chat_template
    fake_unsloth.chat_templates = fake_chat_templates

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.chat_templates", fake_chat_templates)

    tokenizer = types.SimpleNamespace()
    out = utils.chat_template_func(tokenizer, chat_template="llama")

    assert out is tokenizer
    assert out.chat_template == "applied:llama"
    assert len(calls) == 1
    assert calls[0]["chat_template"] == "llama"
    assert calls[0]["map_eos_token"] is False
