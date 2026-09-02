# ABOUTME: Regression test that SFT leaves the collator for TRL to build.
# ABOUTME: A caller-supplied collator makes Unsloth force padding_free off, halving throughput.

from types import SimpleNamespace

import tuning.training.sft_training as sft_training


def _run_pipeline(monkeypatch, tmp_path, mask_prompt):
    """Drive train_model_sft with every heavy dependency stubbed out."""
    rendered_dataset = {
        "train": [{"text": "<bos>train<eos>"}],
        "test": [{"text": "<bos>test<eos>"}],
    }
    tokenized_dataset = {
        "train": [{"text": "<bos>train<eos>", "input_ids": [1, 2, 3]}],
        "test": [{"text": "<bos>test<eos>", "input_ids": [1, 4, 3]}],
    }
    model = SimpleNamespace(config=SimpleNamespace(use_cache=True))
    tokenizer = SimpleNamespace(pad_token_id=0)
    captured = {}

    monkeypatch.setattr(sft_training, "get_train_dataset", lambda _config: rendered_dataset)
    monkeypatch.setattr(
        sft_training,
        "load_model_with_lora",
        lambda **_kwargs: (model, tokenizer),
    )
    monkeypatch.setattr(sft_training, "chat_template_func", lambda value: value)
    monkeypatch.setattr(
        sft_training,
        "apply_chat_template",
        lambda _tokenizer, _dataset, mask_prompt=False: rendered_dataset,
    )
    monkeypatch.setattr(
        sft_training,
        "tokenize_sft_dataset",
        lambda *_args, **_kwargs: tokenized_dataset,
    )
    monkeypatch.setattr(
        sft_training,
        "SFTConfig",
        lambda **kwargs: SimpleNamespace(
            gradient_accumulation_steps=1,
            to_dict=lambda: kwargs,
            **kwargs,
        ),
    )

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.args = kwargs["args"]

        def train(self, *, resume_from_checkpoint):
            assert resume_from_checkpoint is None

    monkeypatch.setattr(sft_training, "SFTTrainer", FakeTrainer)
    monkeypatch.setattr(sft_training, "OffsetAwareWandbCallback", lambda **_kwargs: object())
    monkeypatch.setattr(sft_training, "disable_loss_kwargs_if_unsupported", lambda _trainer: False)
    monkeypatch.setattr(sft_training, "remove_default_wandb_callback", lambda _trainer: None)
    monkeypatch.setattr(sft_training, "save_trained_model", lambda *_args: None)

    sft_training.train_model_sft(
        run_config=SimpleNamespace(
            model_name_hf="test-model",
            model_name="llama3-8B",
            output_dir=str(tmp_path / "output"),
        ),
        lora_config=object(),
        model_load_config=SimpleNamespace(max_seq_length=1024),
        training_args=SimpleNamespace(
            full_finetune=False,
            mask_prompt_tokens=mask_prompt,
            resume_from_checkpoint=False,
            fsdp="",
            gpu_minute_multiplier=None,
            do_eval=True,
            packing=False,
            packing_strategy="bfd",
            padding_free=False,
            to_hf_args=lambda **_kwargs: {},
        ),
    )
    return captured


def test_sft_leaves_collator_to_trl(monkeypatch, tmp_path):
    """Unsloth blocks padding-free whenever the caller supplies a collator."""
    captured = _run_pipeline(monkeypatch, tmp_path, mask_prompt=True)

    assert captured.get("data_collator") is None


def test_sft_config_carries_completion_only_loss(monkeypatch, tmp_path):
    """Masking has to survive on SFTConfig, since TRL now builds the collator."""
    for mask_prompt in (True, False):
        captured = _run_pipeline(monkeypatch, tmp_path, mask_prompt=mask_prompt)
        assert captured["args"].completion_only_loss is mask_prompt


def test_padding_free_status_reports_active():
    args = SimpleNamespace(padding_free=True)
    model = SimpleNamespace(config=SimpleNamespace())

    assert sft_training.padding_free_status(model, args, full_finetune=False) == "on"


def test_padding_free_status_names_the_full_finetune_path():
    args = SimpleNamespace(padding_free=False)
    model = SimpleNamespace(config=SimpleNamespace())

    status = sft_training.padding_free_status(model, args, full_finetune=True)

    assert status.startswith("off")
    assert "full fine-tune" in status


def test_padding_free_status_reports_packing_on():
    """BFD packing removes padding entirely, full fine-tune included."""
    args = SimpleNamespace(padding_free=False, packing=True)

    status = sft_training.padding_free_status(None, args, full_finetune=True)

    assert status.startswith("on")


def test_padding_free_status_names_vision_language_models():
    args = SimpleNamespace(padding_free=False)
    model = SimpleNamespace(
        config=SimpleNamespace(architectures=["Gemma3ForConditionalGeneration"])
    )

    status = sft_training.padding_free_status(model, args, full_finetune=False)

    assert status.startswith("off")
    assert "vision-language" in status
    assert "TRL" not in status


def test_padding_free_status_flags_an_unexplained_loss():
    """The collator regression looked exactly like this: off with no supported reason."""
    args = SimpleNamespace(padding_free=False)
    model = SimpleNamespace(config=SimpleNamespace(architectures=["LlamaForCausalLM"]))

    status = sft_training.padding_free_status(model, args, full_finetune=False)

    assert status.startswith("off")
    assert "unexpected" in status


def test_padding_free_defaults_to_unset_so_trl_can_auto_enable():
    """An explicit False reaches SFTConfig and defeats Unsloth's auto-enable, so the
    default must stay unset."""
    from tests.test_unified_early_pipeline import _parse_args, REQUIRED
    assert _parse_args(REQUIRED).sft_padding_free is None


def test_padding_free_flag_forces_it_on_and_off():
    from tests.test_unified_early_pipeline import _parse_args, REQUIRED
    assert _parse_args(REQUIRED + ["--sft-padding-free"]).sft_padding_free is True
    assert _parse_args(REQUIRED + ["--no-sft-padding-free"]).sft_padding_free is False


def test_training_config_leaves_padding_free_unset_by_default():
    from tuning.training.config_training import TrainingArgumentsConfig
    assert TrainingArgumentsConfig().padding_free is None
