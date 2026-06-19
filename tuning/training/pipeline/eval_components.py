# ABOUTME: Builders that turn parsed CLI args into PassAtKConfig / EvalStrategy /
# ABOUTME: PerplexityConfig / W&B-tag lists. Pure data-shaping; no training imports.


def _sft_ppl_config(args):
    """Return PerplexityConfig for SFT, or None if disabled."""
    if not args.sft_enable_ppl:
        return None
    from tuning.training.config_training import PerplexityConfig
    return PerplexityConfig(
        perplexity_thresholds=args.sft_ppl_thresholds,
        num_samples=args.sft_ppl_num_samples,
        early_tuples=args.sft_ppl_early or None,
        enabled=True,
    )


def _dpo_ppl_config(args):
    """Return PerplexityConfig for DPO, or None if disabled."""
    if not args.dpo_enable_ppl:
        return None
    from tuning.training.config_training import PerplexityConfig
    return PerplexityConfig(
        perplexity_thresholds=args.dpo_ppl_thresholds,
        num_samples=args.dpo_ppl_num_samples,
        early_tuples=args.dpo_ppl_early or None,
        enabled=True,
    )


def _build_eval_components(args, stage, gpu_util):
    """Build PassAtKConfig + eval strategies for the given task and stage.

    Returns (passk_config, primary_eval, monitor_evals).
    All three are None/[] if pass@k is disabled for this stage.
    """
    prefix = stage  # "sft", "dpo", or "grpo"
    if not getattr(args, f"{prefix}_enable_passk", False):
        return None, None, []

    from tuning.training.config_training import JudgeConfig, PassAtKConfig
    judge_config = None
    if getattr(args, f"{prefix}_enable_judge", False):
        judge_config = JudgeConfig(
            enabled=True,
            model=getattr(args, f"{prefix}_judge_model", "deepseek-v4-flash"),
            base_url=getattr(args, f"{prefix}_judge_base_url", "https://api.deepseek.com"),
            api_key_env=getattr(args, f"{prefix}_judge_api_key_env", "DEEPSEEK_API_KEY"),
            samples_per_prompt=getattr(args, f"{prefix}_judge_samples_per_prompt", 1),
            concurrency=getattr(args, f"{prefix}_judge_concurrency", 16),
            timeout=getattr(args, f"{prefix}_judge_timeout", 60.0),
            max_retries=getattr(args, f"{prefix}_judge_max_retries", 3),
            max_tokens=getattr(args, f"{prefix}_judge_max_tokens", 64),
        )

    passk_config = PassAtKConfig(
        target_pass_at_k=getattr(args, f"{prefix}_passk_targets"),
        early_tuples=getattr(args, f"{prefix}_passk_early") or None,
        temperature=getattr(args, f"{prefix}_passk_temperature"),
        enabled=True,
        num_inference_gpus=getattr(args, f"{prefix}_passk_num_inference_gpus"),
        use_persistent_vllm=getattr(args, f"{prefix}_passk_persistent_vllm"),
        vllm_gpu_memory_utilization=gpu_util,
        max_checkpoint_gap=getattr(args, f"{prefix}_passk_max_checkpoint_gap", None),
        target_data_points=getattr(args, f"{prefix}_passk_target_data_points", None),
        target_total_minutes=getattr(args, f"{prefix}_passk_target_total_minutes", None),
        judge=judge_config,
    )

    k_values = getattr(args, f"{prefix}_passk_k_values", [1])
    n_samples = getattr(args, f"{prefix}_passk_n_samples", 1)
    num_prompts = getattr(args, f"{prefix}_passk_num_prompts", None)

    if args.task_name == "ifeval":
        from tuning.training.eval_strategy import IFEvalStrategy
        strict = getattr(args, f"{prefix}_passk_strict", True)
        primary_eval = IFEvalStrategy(
            k_values=k_values, n_samples=n_samples,
            num_prompts=num_prompts or 541, strict=strict,
        )
    elif args.task_name == "gsm8k":
        from tuning.training.eval_strategy import GSM8KEvalStrategy
        primary_eval = GSM8KEvalStrategy(
            k_values=k_values, n_samples=n_samples, num_prompts=num_prompts,
        )
    elif args.task_name == "math500":
        from tuning.training.eval_strategy import MATH500EvalStrategy
        primary_eval = MATH500EvalStrategy(
            k_values=k_values, n_samples=n_samples, num_prompts=num_prompts,
        )
    elif args.task_name == "amc":
        from tuning.training.eval_strategy import AMCEvalStrategy
        primary_eval = AMCEvalStrategy(
            k_values=k_values, n_samples=n_samples, num_prompts=num_prompts,
        )
    elif args.task_name == "ifbench":
        from tuning.training.eval_strategy import IFBenchStrategy
        strict = getattr(args, f"{prefix}_passk_strict", True)
        primary_eval = IFBenchStrategy(
            k_values=k_values, n_samples=n_samples,
            num_prompts=num_prompts, strict=strict,
        )
    else:
        raise ValueError(f"Unknown task name: {args.task_name}")

    monitor_evals = _build_monitor_evals(args, k_values, n_samples)
    return passk_config, primary_eval, monitor_evals


def _build_monitor_evals(args, k_values, n_samples):
    """Build monitor eval strategies from --monitor-evals arg."""
    monitor_evals = []
    for name in getattr(args, "monitor_evals", []):
        if name == args.task_name:
            continue
        if name == "math500":
            from tuning.training.eval_strategy import MATH500EvalStrategy
            monitor_evals.append(MATH500EvalStrategy(k_values=k_values, n_samples=n_samples))
        elif name == "amc":
            from tuning.training.eval_strategy import AMCEvalStrategy
            monitor_evals.append(AMCEvalStrategy(k_values=k_values, n_samples=n_samples))
        elif name == "gsm8k":
            from tuning.training.eval_strategy import GSM8KEvalStrategy
            monitor_evals.append(GSM8KEvalStrategy(k_values=k_values, n_samples=n_samples))
        elif name == "ifeval":
            from tuning.training.eval_strategy import IFEvalStrategy
            monitor_evals.append(IFEvalStrategy(k_values=k_values, n_samples=n_samples))
        elif name == "ifbench":
            from tuning.training.eval_strategy import IFBenchStrategy
            monitor_evals.append(IFBenchStrategy(k_values=k_values, n_samples=n_samples))
    return monitor_evals


def _sft_tags(passk_config, ppl_config, primary_eval=None):
    """Build W&B tags for an SFT run."""
    from tuning.training.wandb_utils import (
        get_early_pairs, early_pair_tag, get_early_abs, early_abs_tag,
    )
    tags = ["sft"]
    if primary_eval is not None:
        tags.append(primary_eval.id)
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        # tags.append(f"p{k_val}")
        # tags.append(early_pair_tag(get_early_pairs(passk_config)))
    if ppl_config is not None:
        tags.append("ppl")
        tags.append(early_pair_tag(get_early_pairs(ppl_config)))
        tags.append(early_abs_tag(get_early_abs(ppl_config)))
    if passk_config is None and ppl_config is None:
        tags.append("no_callbacks")
    return tags


def post_training_tags(method, checkpoint, primary_eval, passk_config, ppl_config=None):
    """Build W&B tags for a DPO/GRPO/KTO run forked from an SFT checkpoint."""
    tags = [method, str(checkpoint["threshold_value"]), str(checkpoint["data_points_seen"])]
    if primary_eval is not None:
        tags.append(primary_eval.id)
    if passk_config is not None:
        k_val = primary_eval.stopping_k if primary_eval else 1
        tags.append(f"p{k_val}")
    if ppl_config is not None:
        tags.append("ppl")
    return tags
