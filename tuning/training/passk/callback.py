# ABOUTME: PassAtKStoppingCallback — slim TrainerCallback that orchestrates eval and
# ABOUTME: checkpoint saving via VLLMRunner and CheckpointDecisionEngine.

import tempfile
import os
import datetime
from typing import List, Dict
import torch.distributed as dist
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from tuning.config import MODELS_METADATA_DIR
from tuning.training.callback_utils import (
    get_total_minutes_from_state,
    save_sweetspot_checkpoint,
)
from tuning.training.eval_strategy import EvalStrategy
from .decisions import CheckpointDecision, CheckpointDecisionEngine
from .logging import log_eval_metrics, log_judge_quality
from .judge import DeepSeekJudge, aggregate_quality
from .runners import (
    RunnerConfig,
    VLLMRunner,
    ExternalVLLMRunner,
    ServerVLLMRunner,
    PersistentVLLMRunner,
    EphemeralVLLMRunner,
    DataParallelVLLMRunner,
    trainer_vllm_awake_for_passk,
)


class PassAtKStoppingCallback(TrainerCallback):
    """
    Save checkpoints at eval metric sweetspots for downstream runs.

    Implements the "Fork Strategy": training continues through all thresholds,
    saving checkpoints at each sweetspot without stopping. The final threshold
    in the list will stop training.

    Supports two vLLM modes for inference during training:
    - Persistent mode (default): Keeps vLLM engine alive with base model loaded,
      swaps LoRA adapters each eval. Eliminates cold-start overhead.
    - Non-persistent mode: Creates/destroys vLLM each eval, but still uses
      adapter-only saves instead of full merged model saves.
    """

    def __init__(
        self,
        config,  # PassAtKConfig
        tokenizer,
        model_name: str,
        base_model_hf: str,
        primary_eval: EvalStrategy,
        monitor_evals: list[EvalStrategy] = None,
    ):
        self._decision_engine = CheckpointDecisionEngine(
            target_thresholds=config.target_pass_at_k,
            early_tuples=config.early_tuples,
            max_checkpoint_gap=getattr(config, "max_checkpoint_gap", None),
            target_data_points=getattr(config, "target_data_points", None),
            target_total_minutes=getattr(config, "target_total_minutes", None),
            eval_only_minutes=getattr(config, "eval_only_minutes", None),
        )
        self._step_offset = int(getattr(config, "initial_global_step", 0) or 0)
        self.tokenizer = tokenizer
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.model_name = model_name
        self.metadata_path = None
        self._primary_metric_history = []
        self._last_eval_step = -1
        self._last_checkpoint_data_points = 0
        self._accelerator = None
        # Set by the GRPO setup when the callback reuses the trainer's vLLM engine
        # (colocate/server). Used to refresh the engine before the baseline eval.
        self._trainer_vllm_generation = None

        self._judge = None
        self._judge_samples_per_prompt = 1
        self._judge_conditioned_metrics = True
        judge_config = getattr(config, "judge", None)
        if judge_config is not None and judge_config.enabled and self._is_rank_zero():
            self._judge_samples_per_prompt = judge_config.samples_per_prompt
            self._judge_conditioned_metrics = judge_config.conditioned_metrics
            self._judge = DeepSeekJudge(
                model=judge_config.model,
                base_url=judge_config.base_url,
                api_key_env=judge_config.api_key_env,
                concurrency=judge_config.concurrency,
                timeout=judge_config.timeout,
                max_retries=judge_config.max_retries,
                temperature=judge_config.temperature,
                max_tokens=judge_config.max_tokens,
            )
            print(f"[PassAtKCallback] LLM judge enabled: {self._judge.model}")

        # Eval strategies
        self.primary_eval = primary_eval
        self.monitor_evals = monitor_evals or []

        # LoRA adapter / persistent vLLM settings
        self.num_inference_gpus = config.num_inference_gpus
        # Capture the full set of CUDA devices available for inference workers.
        # The pipeline script saves the original SLURM allocation to CUDA_VISIBLE_DEVICES_ALL
        # before restricting CUDA_VISIBLE_DEVICES to GPU 0 for training.
        cuda_all = os.environ.get("CUDA_VISIBLE_DEVICES_ALL", "")
        cuda_env = cuda_all or os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_env:
            self._available_gpus = [g.strip() for g in cuda_env.split(",") if g.strip()]
        else:
            # No env var set — assume GPUs 0..N-1 are available
            self._available_gpus = [str(i) for i in range(max(self.num_inference_gpus, 1))]
        self.use_persistent_vllm = config.use_persistent_vllm
        if self.num_inference_gpus > 1 and self.use_persistent_vllm:
            print(f"[PassAtKCallback] WARNING: num_inference_gpus={self.num_inference_gpus} requires ephemeral mode. "
                  f"Overriding use_persistent_vllm=True → False.")
            self.use_persistent_vllm = False

        self._runner_config = RunnerConfig(
            base_model_hf=base_model_hf,
            vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            lora_max_rank=getattr(config, "lora_max_rank", 32),
            chat_template=tokenizer.chat_template,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            available_gpus=self._available_gpus,
            max_model_len=config.vllm_max_model_len,
            num_inference_gpus=self.num_inference_gpus,
        )
        self._runner = self._build_runner(config)

        # n_samples from primary eval for vLLM sampling
        self.n_samples = primary_eval.n_samples

        mode_str = "persistent" if self.use_persistent_vllm else "non-persistent"
        if not self._decision_engine.early_tuples:
            print(f"[PassAtKCallback] Initialized with {primary_eval.label_prefix} "
                  f"thresholds={self._decision_engine.target_thresholds}")
            print(f"[PassAtKCallback] Training will stop when hardest threshold is "
                  f"reached: {self._decision_engine.target_thresholds[0]}")
        else:
            print(f"[PassAtKCallback] Initialized with "
                  f"early_tuples={self._decision_engine.early_tuples}")
            print(f"[PassAtKCallback] Training will stop when all early_tuples have "
                  f"triggered")

        print(f"[PassAtKCallback] primary_eval={primary_eval.__class__.__name__}, "
              f"monitor_evals={[e.__class__.__name__ for e in self.monitor_evals]}")
        print(f"[PassAtKCallback] n_samples={self.n_samples}, temperature={self.temperature}")
        parallelism_str = f", data-parallel over {self.num_inference_gpus} GPUs" if self.num_inference_gpus > 1 else ""
        print(f"[PassAtKCallback] vLLM mode: {mode_str}{parallelism_str}, base_model_hf={base_model_hf}, gpu_mem={config.vllm_gpu_memory_utilization}")
        print(f"[PassAtKCallback] Chat template: {tokenizer.chat_template}")
        print(f"[PassAtKCallback] Config: {config}")

        # Log a sample formatted prompt to verify template
        sample_messages = primary_eval.get_test_messages()[0]
        sample_formatted = self.tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        print(f"\n{'='*60}")
        print(f"[DEBUG] chat_template used for inference: {tokenizer.chat_template[:200]}...")
        print(f"[DEBUG] Sample prompt (index 0):")
        print(sample_formatted)
        print(f"{'='*60}\n")

    def _is_rank_zero(self) -> bool:
        """True when not under DDP, or when this is rank 0 under DDP."""
        return not dist.is_initialized() or dist.get_rank() == 0

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.model_name:
            self.model_name = kwargs.get("model")
        print(f"[PassAtKCallback] on_train_begin: model_name={self.model_name}")
        now = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.metadata_path = os.path.join(MODELS_METADATA_DIR, f"{self.model_name}_{self.primary_eval.id}_{self.primary_eval.stopping_metric()}-{now}.json")

        # When reusing the trainer's vLLM engine, refresh its weights from the trainer
        # model before the baseline eval. HF applies resume_from_checkpoint inside
        # train(), after the engine was built, so it still holds pre-resume weights here.
        if self._trainer_vllm_generation is not None:
            print("[PassAtKCallback] syncing trainer weights to vLLM before baseline eval")
            self._trainer_vllm_generation.sync_weights()

        # Baseline evaluation before training starts
        self.on_evaluate(args, state, control, **kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        total_minutes = get_total_minutes_from_state(state)
        crossed = self._decision_engine.consume_crossed_total_minute_targets(total_minutes)
        if crossed:
            control.should_evaluate = True
            print(
                "[PassAtKCallback] total_minutes target crossed; "
                f"requesting eval for {max(crossed):g}m "
                f"(actual {total_minutes:.2f}m)"
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Cleanup persistent vLLM engine when training ends."""
        if self._last_eval_step != state.global_step:
            model = kwargs.pop("model", None)
            if model is not None:
                print(f"[PassAtKCallback] Running final evaluation at end of training (step {state.global_step})...")
                self.on_evaluate(args, state, control, model=model, **kwargs)
            else:
                print("[PassAtKCallback] Warning: model is None at on_train_end, skipping final evaluation")

        self._runner.cleanup()

    def _build_runner(self, config) -> VLLMRunner:
        if config.num_inference_gpus > 1:
            return DataParallelVLLMRunner(self._runner_config)
        if config.use_persistent_vllm:
            return PersistentVLLMRunner(self._runner_config)
        return EphemeralVLLMRunner(self._runner_config)

    def set_trainer_vllm(self, llm, vllm_generation=None):
        self._trainer_vllm_generation = vllm_generation
        self._runner = ExternalVLLMRunner(
            self._runner_config, llm=llm, vllm_generation=vllm_generation,
        )

    def set_trainer_vllm_client(self, client):
        """Route eval through the GRPO trainer's vLLM server (server mode).

        `client` is `trainer.vllm_generation.vllm_client` on rank 0 and None on
        other ranks. The DDP eval path runs `run()` only on rank 0 and
        broadcasts results, so non-rank-0 runners holding None never execute.
        """
        self._runner = ServerVLLMRunner(
            self._runner_config, client=client, tokenizer=self.tokenizer,
        )

    def _save_lora_adapter(self, model, adapter_dir: str):
        """Save only the LoRA adapter weights (~50MB instead of ~2GB merged)."""
        print(f"[PassAtKCallback] Saving LoRA adapter to {adapter_dir}...")

        # Use standard PEFT save to ensure adapter_config.json is created for vLLM
        if hasattr(model, 'save_pretrained'):
            print(f"[PassAtKCallback] PEFT saving adaptor only")
            # PEFT model - save adapter only
            model.save_pretrained(adapter_dir)
        else:
            # Fallback: use unsloth's method
            print(f"[PassAtKCallback] Model does not have save_pretrained, using merged method with lora save")
            model.save_pretrained_merged(adapter_dir, self.tokenizer, save_method="lora")
        # Save tokenizer so vLLM doesn't warn about missing tokenizer in adapter dir
        self.tokenizer.save_pretrained(adapter_dir)
        print(f"[PassAtKCallback] LoRA adapter saved")

    def _save_decision_checkpoint(
        self,
        model,
        decision: CheckpointDecision,
        state: TrainerState,
        args: TrainingArguments,
    ):
        """Save the checkpoint requested by the decision engine."""
        threshold_type = decision.metadata_type or self.primary_eval.stopping_metric()
        threshold_value = (
            decision.metadata_value
            if decision.metadata_value is not None
            else decision.label
        )
        extra_metadata = {
            "threshold_type": threshold_type,
            "threshold_value": threshold_value,
        }
        if decision.eval_only:
            # Pre-claim the row so claim_next_checkpoint skips it: the checkpoint
            # is saved and evaluated but never seeds post-training.
            extra_metadata["eval_only"] = True
            extra_metadata["claimed"] = True
        return save_sweetspot_checkpoint(
            model=model,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            threshold_label=f"{self.primary_eval.label_prefix}-{decision.label}",
            state=state,
            args=args,
            metadata_path=self.metadata_path,
            extra_metadata=extra_metadata,
            accelerator=self._accelerator,
        )

    def _save_adapter_if_needed(self, model, adapter_dir: str):
        if isinstance(self._runner, (ExternalVLLMRunner, ServerVLLMRunner)):
            return None
        self._save_lora_adapter(model, adapter_dir)
        return adapter_dir

    def _run_eval_with_results(self, model, eval_strategy: EvalStrategy) -> tuple[Dict[str, float], List[Dict]]:
        """Run vLLM inference and score responses using the given eval strategy."""
        if dist.is_initialized() and dist.get_world_size() > 1:
            return self._run_eval_with_results_ddp(model, eval_strategy)
        with tempfile.TemporaryDirectory() as adapter_dir:
            adapter_path = self._save_adapter_if_needed(model, adapter_dir)
            try:
                model_results = self._runner.run(model, eval_strategy, adapter_path)
            except Exception as exc:
                if isinstance(self._runner, PersistentVLLMRunner):
                    print(f"[PassAtKCallback] Persistent vLLM failed: {exc}, "
                          f"swapping to ephemeral runner and retrying")
                    self._runner.cleanup()
                    self._runner = EphemeralVLLMRunner(self._runner_config)
                    model_results = self._runner.run(model, eval_strategy, adapter_path)
                else:
                    raise

        print(f"[PassAtKCallback] Scoring responses with "
              f"{eval_strategy.__class__.__name__}...")
        scores = eval_strategy.score_responses(model_results, self.tokenizer)
        return scores, model_results

    def _run_eval_with_results_ddp(self, model, eval_strategy: EvalStrategy) -> tuple[Dict[str, float], List[Dict]]:
        """DDP eval path. Two flavours:
        - ExternalVLLMRunner (colocate): each rank generates its slice via its own
          colocated vLLM, all ranks gather responses, all ranks score.
        - ServerVLLMRunner (server mode): rank 0 issues the HTTP call to the
          shared vLLM server and broadcasts results to the other ranks.

        Returns (scores, model_results) on every rank.
        """
        from collections import defaultdict

        if isinstance(self._runner, ServerVLLMRunner):
            rank = dist.get_rank()
            if rank == 0:
                model_results = self._runner.run(model, eval_strategy, adapter_path=None)
            else:
                model_results = None
            payload = [model_results]
            dist.broadcast_object_list(payload, src=0)
            model_results = payload[0]
            scores = eval_strategy.score_responses(model_results, self.tokenizer)
            return scores, model_results

        from vllm import SamplingParams
        from tuning.inference.config_inference import VLLMSamplingParamsConfig

        if not isinstance(self._runner, ExternalVLLMRunner):
            raise RuntimeError(
                "DDP eval path requires the colocated trainer vLLM "
                "(set via set_trainer_vllm) or the server client (set via "
                "set_trainer_vllm_client)."
            )
        llm = self._runner._llm

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        all_messages = eval_strategy.get_test_messages()
        local_indices = list(range(rank, len(all_messages), world_size))
        local_messages = [all_messages[i] for i in local_indices]

        sampling_params = SamplingParams(**VLLMSamplingParamsConfig(
            n=eval_strategy.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).model_dump())

        with trainer_vllm_awake_for_passk(llm, self._trainer_vllm_generation):
            if local_messages:
                outputs = llm.chat(
                    local_messages,
                    sampling_params,
                    chat_template=self._runner.config.chat_template,
                )
                local_pairs = [
                    (idx, [r.text for r in out.outputs])
                    for idx, out in zip(local_indices, outputs)
                ]
            else:
                local_pairs = []

        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_pairs)

        flat = sorted(
            ((idx, texts) for chunk in gathered for idx, texts in chunk),
            key=lambda t: t[0],
        )
        responses_per_index = [texts for _, texts in flat]
        test_prompts = eval_strategy.get_test_prompts()
        grouped = defaultdict(list)
        for prompt, resp in zip(test_prompts, responses_per_index):
            grouped[prompt].extend(resp)
        model_results = [{"prompt": p, "responses": r} for p, r in grouped.items()]

        print(f"[PassAtKCallback] DDP eval rank={rank}/{world_size}: "
              f"local={len(local_messages)} prompts, gathered={len(model_results)} unique")
        scores = eval_strategy.score_responses(model_results, self.tokenizer)
        return scores, model_results

    def _run_eval(self, model, eval_strategy: EvalStrategy) -> Dict[str, float]:
        """Run vLLM inference and score responses using the given eval strategy."""
        scores, _ = self._run_eval_with_results(model, eval_strategy)
        return scores

    def _compute_data_points_seen(self, args, state) -> int:
        train_batch_size = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        world_size = getattr(args, "world_size", 1)
        return state.global_step * train_batch_size * grad_accum * world_size


    def _select_judge_examples(self, raw_results):
        pairs = []
        correctness = []
        samples_per_prompt = self._judge_samples_per_prompt
        for item in raw_results:
            prompt = item.get("prompt", "")
            responses = item.get("responses", [])
            if not isinstance(responses, list):
                responses = [responses]
            per_response_correct = item.get("per_response_correct", [])
            if not isinstance(per_response_correct, list):
                per_response_correct = [per_response_correct]

            limit = len(responses) if samples_per_prompt <= 0 else min(
                samples_per_prompt, len(responses),
            )
            for idx in range(limit):
                response = responses[idx]
                if response is None:
                    continue
                pairs.append((str(prompt), str(response)))
                if idx < len(per_response_correct):
                    correct = per_response_correct[idx]
                    correctness.append(None if correct is None else bool(correct))
                else:
                    correctness.append(None)
        return pairs, correctness

    def _maybe_run_judge(
        self, eval_strategy, raw_results, *, global_step: int, total_minutes: float,
    ) -> None:
        if self._judge is None:
            return

        pairs, correctness = self._select_judge_examples(raw_results)
        if not pairs:
            print("[PassAtKCallback] Judge enabled but no responses were available")
            return

        self._run_judge_job(
            eval_strategy.id, pairs, correctness, global_step, total_minutes,
        )

    def _run_judge_job(
        self, eval_id: str, pairs, correctness, global_step: int, total_minutes: float,
    ) -> None:
        try:
            quality = self._judge.score_pairs(pairs)
            metrics = aggregate_quality(
                quality, correctness, conditioned=self._judge_conditioned_metrics,
            )
            log_judge_quality(
                eval_id=eval_id,
                metrics=metrics,
                global_step=global_step,
                step_offset=self._step_offset,
                total_minutes=total_minutes,
            )
        except Exception as exc:
            print(f"[PassAtKCallback] Warning: judge job failed: {exc}")

    def _eval_and_log(self, model, eval_strategy, state, *, is_primary: bool):
        scores, raw_results = self._run_eval_with_results(model, eval_strategy)
        total_minutes = get_total_minutes_from_state(state)
        if self._is_rank_zero():
            log_eval_metrics(
                eval_strategy=eval_strategy,
                scores=scores,
                raw_results=raw_results,
                global_step=state.global_step,
                step_offset=self._step_offset,
                total_minutes=total_minutes,
                thresholds_remaining=self._decision_engine.target_thresholds,
                is_primary=is_primary,
            )
            if is_primary:
                self._maybe_run_judge(
                    eval_strategy, raw_results,
                    global_step=state.global_step,
                    total_minutes=total_minutes,
                )
        return scores

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        """Called after evaluation, run evals and stop if target reached."""
        if model is None:
            model = kwargs.get("model")
        if model is None:
            print("[PassAtKCallback] Warning: model is None, skipping eval")
            return control

        data_points_seen = self._compute_data_points_seen(args, state)
        total_minutes = get_total_minutes_from_state(state)

        primary_scores = self._eval_and_log(model, self.primary_eval, state,
                                            is_primary=True)
        primary_metric = primary_scores[self.primary_eval.stopping_metric()]
        self._primary_metric_history.append(primary_metric)

        for monitor_eval in self.monitor_evals:
            self._eval_and_log(model, monitor_eval, state, is_primary=False)

        decisions = self._decision_engine.decide(
            primary_metric=primary_metric,
            history=self._primary_metric_history,
            data_points_seen=data_points_seen,
            last_checkpoint_data_points=self._last_checkpoint_data_points,
            total_minutes=total_minutes,
        )
        for decision in decisions:
            if self._is_rank_zero():
                self._save_decision_checkpoint(model, decision, state, args)
            if decision.advances_state:
                self._last_checkpoint_data_points = data_points_seen
            if self._is_rank_zero():
                print(f"[PassAtKCallback] Saved checkpoint: {decision.label}")

        self._last_eval_step = state.global_step
        return control
