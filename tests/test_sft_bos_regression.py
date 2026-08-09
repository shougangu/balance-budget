# ABOUTME: Regression test that SFT hands TRL pre-tokenized ids, not rendered chat text.
# ABOUTME: Re-tokenizing rendered text prepends a second BOS, which costs math pass@1.

from types import SimpleNamespace

import tuning.training.sft_training as sft_training


def test_sft_pipeline_pretokenizes_rendered_text_before_trl(monkeypatch, tmp_path):
    """Keep TRL from adding a second BOS to already-rendered chat text."""
    raw_dataset = {
        "train": [{"messages": ["train"]}],
        "test": [{"messages": ["test"]}],
    }
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
    calls = []

    monkeypatch.setattr(sft_training, "get_train_dataset", lambda _config: raw_dataset)
    monkeypatch.setattr(
        sft_training,
        "load_model_with_lora",
        lambda **_kwargs: (model, tokenizer),
    )
    monkeypatch.setattr(sft_training, "chat_template_func", lambda value: value)
    monkeypatch.setattr(
        sft_training,
        "apply_chat_template",
        lambda passed_tokenizer, dataset, mask_prompt=False: rendered_dataset,
    )

    def fake_tokenize(passed_tokenizer, dataset, max_length, num_proc, mask_prompt=False):
        calls.append((passed_tokenizer, dataset, max_length, num_proc))
        return tokenized_dataset

    monkeypatch.setattr(sft_training, "tokenize_sft_dataset", fake_tokenize)
    monkeypatch.setattr(
        sft_training,
        "SFTConfig",
        lambda **kwargs: SimpleNamespace(
            gradient_accumulation_steps=1,
            to_dict=lambda: kwargs,
        ),
    )

    class FakeTrainer:
        def __init__(self, *, train_dataset, eval_dataset, args, **_kwargs):
            assert train_dataset is tokenized_dataset["train"]
            assert eval_dataset is tokenized_dataset["test"]
            assert "input_ids" in train_dataset[0]
            self.args = args

        def train(self, *, resume_from_checkpoint):
            assert resume_from_checkpoint is None

    monkeypatch.setattr(sft_training, "SFTTrainer", FakeTrainer)
    monkeypatch.setattr(sft_training, "OffsetAwareWandbCallback", object)
    monkeypatch.setattr(sft_training, "disable_loss_kwargs_if_unsupported", lambda _trainer: False)
    monkeypatch.setattr(sft_training, "remove_default_wandb_callback", lambda _trainer: None)
    monkeypatch.setattr(sft_training, "save_trained_model", lambda *_args: None)
    run_config = SimpleNamespace(
        model_name_hf="test-model",
        model_name="llama3-8B",
        output_dir=str(tmp_path / "output"),
    )
    training_args = SimpleNamespace(
        full_finetune=False,
        mask_prompt_tokens=False,
        resume_from_checkpoint=False,
        to_hf_args=lambda **_kwargs: {},
    )
    model_load_config = SimpleNamespace(max_seq_length=1024)

    sft_training.train_model_sft(
        run_config=run_config,
        lora_config=object(),
        model_load_config=model_load_config,
        training_args=training_args,
    )

    assert calls == [(tokenizer, rendered_dataset, 1024, 4)]
    assert model.config.use_cache is False
