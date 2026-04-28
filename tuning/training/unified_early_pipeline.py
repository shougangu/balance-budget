# ABOUTME: CLI-driven unified SFT+post-training pipeline with optional pass@k and perplexity callbacks.
# ABOUTME: Supports SFT→{DPO,GRPO,KTO} runs from a single command.

import json
import sys
import subprocess
from pathlib import Path

import tuning.config

from tuning.training.pipeline.cli import (
    init_cuda_env, is_worker_mode,
    SBATCH_WORKER_SCRIPT_DEFAULT as SBATCH_WORKER_SCRIPT,
    MODEL_TO_GPU_1, MODEL_TO_GPU_2, MODEL_TO_GPU_3, MODEL_TO_SIMPLERL_TIER,
    parse_early_tuple, effective_eval_seed, _resolve_simplerl_dataset,
    _init_seeds, _parse_args,
)
from tuning.training.pipeline.checkpoint_metadata import (
    load_checkpoints, next_checkpoint, claim_next_checkpoint, mark_completed,
    print_metadata_paths, parse_metadata_from_output,
)
from tuning.training.pipeline.eval_components import (
    _build_eval_components, _sft_ppl_config, _dpo_ppl_config,
    _sft_tags, post_training_tags,
)


if is_worker_mode():
    init_cuda_env()
    if "--run-grpo" not in sys.argv:
        import unsloth  # noqa: F401 - must be imported before trl/transformers/peft


from tuning.training.pipeline.orchestrator import (
    _build_base_cmd, _submit_sbatch_worker, _dispatch_parallel_workers, main,
)


if __name__ == "__main__":
    main()
