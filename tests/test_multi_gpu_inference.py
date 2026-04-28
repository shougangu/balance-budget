"""Unit tests for multi-GPU data-parallel inference (no GPU required)."""

from unittest.mock import MagicMock

from tuning.training.config_training import PassAtKConfig
from tuning.training.passk.data_parallel import partition_prompts


# --- PassAtKConfig tests ---

class TestPassAtKConfigNumInferenceGpus:
    def test_default_is_one(self):
        config = PassAtKConfig()
        assert config.num_inference_gpus == 1

    def test_custom_value(self):
        config = PassAtKConfig(num_inference_gpus=4)
        assert config.num_inference_gpus == 4

    def test_single_gpu_value(self):
        config = PassAtKConfig(num_inference_gpus=1)
        assert config.num_inference_gpus == 1


# --- Persistent mode override tests ---

class TestPersistentModeOverride:
    @staticmethod
    def _make_callback(num_gpus, persistent):
        from tuning.training.passk.callback import PassAtKStoppingCallback
        config = PassAtKConfig(
            num_inference_gpus=num_gpus,
            use_persistent_vllm=persistent,
            enabled=True,
        )
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = "test_template"
        mock_eval = MagicMock()
        mock_eval.n_samples = 1
        mock_eval.get_test_messages.return_value = [[{"role": "user", "content": "x"}]]
        return PassAtKStoppingCallback(
            config=config,
            tokenizer=mock_tokenizer,
            model_name="test-model",
            base_model_hf="test/model",
            primary_eval=mock_eval,
        )

    def test_persistent_forced_off_multi_gpu(self):
        """num_inference_gpus > 1 should force use_persistent_vllm to False."""
        callback = self._make_callback(num_gpus=2, persistent=True)
        assert callback.use_persistent_vllm is False
        assert callback.num_inference_gpus == 2

    def test_persistent_unchanged_single_gpu(self):
        """num_inference_gpus=1 should not override use_persistent_vllm."""
        callback = self._make_callback(num_gpus=1, persistent=True)
        assert callback.use_persistent_vllm is True


# --- partition_prompts tests ---

class TestPartitionPrompts:
    def test_even_split(self):
        """100 prompts / 4 GPUs → 4 chunks of 25."""
        messages = list(range(100))
        chunks = partition_prompts(messages, 4)
        assert len(chunks) == 4
        assert all(len(c) == 25 for c in chunks)
        # All items present
        assert sorted(item for c in chunks for item in c) == messages

    def test_uneven_split(self):
        """541 prompts / 4 GPUs → chunks of [136, 135, 135, 135]."""
        messages = list(range(541))
        chunks = partition_prompts(messages, 4)
        assert len(chunks) == 4
        assert [len(c) for c in chunks] == [136, 135, 135, 135]
        assert sorted(item for c in chunks for item in c) == messages

    def test_single_gpu(self):
        """541 prompts / 1 GPU → 1 chunk of 541."""
        messages = list(range(541))
        chunks = partition_prompts(messages, 1)
        assert len(chunks) == 1
        assert len(chunks[0]) == 541

    def test_more_gpus_than_prompts(self):
        """3 prompts / 8 GPUs → 3 chunks of 1."""
        messages = list(range(3))
        chunks = partition_prompts(messages, 8)
        assert len(chunks) == 3
        assert all(len(c) == 1 for c in chunks)

    def test_preserves_order(self):
        """Chunks should preserve the original order."""
        messages = list(range(10))
        chunks = partition_prompts(messages, 3)
        flattened = [item for c in chunks for item in c]
        assert flattened == messages

    def test_two_items_two_gpus(self):
        """Edge case: exactly 2 items on 2 GPUs."""
        messages = ["a", "b"]
        chunks = partition_prompts(messages, 2)
        assert chunks == [["a"], ["b"]]

    def test_one_item_one_gpu(self):
        """Edge case: single item."""
        messages = ["only"]
        chunks = partition_prompts(messages, 1)
        assert chunks == [["only"]]


# --- Merge results order test ---

class TestMergeResultsOrder:
    def test_merge_preserves_prompt_order(self):
        """Results from N workers should merge in original prompt order."""
        # Simulate what _run_data_parallel_inference does internally:
        # GPU 0 gets prompts [A, B], GPU 1 gets [C, D]
        prompts = ["A", "B", "C", "D"]
        prompt_chunks = partition_prompts(prompts, 2)

        # Simulated results_by_gpu (text outputs per prompt)
        results_by_gpu = {
            0: [["resp_A"], ["resp_B"]],
            1: [["resp_C"], ["resp_D"]],
        }

        merged = []
        for gpu_id in range(2):
            chunk_texts = results_by_gpu[gpu_id]
            chunk_prompts = prompt_chunks[gpu_id]
            for prompt, response_texts in zip(chunk_prompts, chunk_texts):
                merged.append({"prompt": prompt, "responses": response_texts})

        assert [m["prompt"] for m in merged] == ["A", "B", "C", "D"]
        assert merged[0]["responses"] == ["resp_A"]
        assert merged[3]["responses"] == ["resp_D"]
