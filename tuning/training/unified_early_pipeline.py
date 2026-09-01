# ABOUTME: Entry point for the unified SFT+post-training pipeline.
# ABOUTME: Slurm scripts call this by file path; all logic lives in tuning.training.pipeline.

import sys

from tuning.training.pipeline.cli import init_cuda_env, is_worker_mode, sft_skips_unsloth

# Worker-mode early init must happen before any transformers/peft import.
# unsloth must be loaded before trl/transformers/peft for non-grpo training.
if is_worker_mode():
    init_cuda_env()
    # unsloth's import monkeypatches transformers globally; only the LoRA SFT/DPO
    # paths use it. GRPO and plain-HF FSDP2 SFT must stay clean.
    if "--run-grpo" not in sys.argv and not sft_skips_unsloth():
        import unsloth  # noqa: F401

from tuning.training.pipeline.orchestrator import main


if __name__ == "__main__":
    main()
