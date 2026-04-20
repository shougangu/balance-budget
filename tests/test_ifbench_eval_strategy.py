# ABOUTME: Tests for IFBenchStrategy eval implementation.
# ABOUTME: Validates interface conformance, scoring, and W&B metrics.

from unittest.mock import patch, MagicMock
from datasets import Dataset


def test_ifbench_strategy_implements_interface():
    from tuning.training.eval_strategy import IFBenchStrategy, EvalStrategy
    assert issubclass(IFBenchStrategy, EvalStrategy)

    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert hasattr(strategy, "get_test_messages")
        assert hasattr(strategy, "score_responses")
        assert hasattr(strategy, "stopping_metric")
        assert hasattr(strategy, "wandb_metrics")
        assert hasattr(strategy, "label_prefix")


def test_ifbench_strategy_id():
    from tuning.training.eval_strategy import IFBenchStrategy
    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert strategy.id == "ifbench"


def test_ifbench_stopping_metric():
    from tuning.training.eval_strategy import IFBenchStrategy
    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1, 5], n_samples=5, num_prompts=1)
        assert strategy.stopping_metric() == "pass_at_1"


def test_ifbench_label_prefix():
    from tuning.training.eval_strategy import IFBenchStrategy
    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert strategy.label_prefix == "ifbench-p@1"


def test_ifbench_wandb_metrics():
    from tuning.training.eval_strategy import IFBenchStrategy
    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1], n_samples=1, num_prompts=1)
        scores = {"pass_at_1": 0.42, "pass_at_1_prompt": 0.35, "avg_response_length_tokens": 100.0, "num_prompts_evaluated": 10}
        wandb_dict = strategy.wandb_metrics(scores)
        assert "eval/ifbench_pass_at_1" in wandb_dict
        assert "eval/ifbench_pass_at_1_prompt" in wandb_dict
        assert "eval/ifbench_avg_response_length_tokens" in wandb_dict
