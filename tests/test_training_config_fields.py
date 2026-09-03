# ABOUTME: Tests field typing on the training-argument config models.
# ABOUTME: A warmup ratio is a fraction of the run and must survive construction as a float.

from tuning.training.config_training import TrainingArgumentsConfig


def test_fractional_warmup_ratio_is_accepted_at_construction():
    cfg = TrainingArgumentsConfig(warmup_ratio=0.005)
    assert cfg.warmup_ratio == 0.005
    assert cfg.to_hf_args(output_dir="/tmp/x")["warmup_ratio"] == 0.005
