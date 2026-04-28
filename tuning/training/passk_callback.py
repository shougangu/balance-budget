# ABOUTME: HuggingFace TrainerCallback that runs vLLM inference during training.
# ABOUTME: Saves checkpoints at metric sweetspots using pluggable EvalStrategy objects.

import wandb
import tempfile
import os
import datetime
import json
from typing import List, Dict
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from tuning.config import MODELS_METADATA_DIR
from tuning.training.callback_utils import save_sweetspot_checkpoint
from tuning.training.eval_strategy import EvalStrategy
from tuning.training.passk.decisions import CheckpointDecisionEngine
from tuning.training.passk.runners import (
    RunnerConfig,
    VLLMRunner,
    ExternalVLLMRunner,
    PersistentVLLMRunner,
    EphemeralVLLMRunner,
    DataParallelVLLMRunner,
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
        )
        self.target_pass_at_k_thresholds = self._decision_engine.target_thresholds
        self.early_tuples = self._decision_engine.early_tuples
        self._step_offset = int(getattr(config, "initial_global_step", 0) or 0)
        self.tokenizer = tokenizer
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.model_name = model_name
        self.metadata_path = None
        self.prevResults = []
        self._last_eval_step = -1
        self._last_checkpoint_data_points = 0

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
            num_inference_gpus=self.num_inference_gpus,
        )
        self._runner = self._build_runner(config)

        # n_samples from primary eval for vLLM sampling
        self.n_samples = primary_eval.n_samples

        mode_str = "persistent" if self.use_persistent_vllm else "non-persistent"
        if not self.early_tuples:
            print(f"[PassAtKCallback] Initialized with {primary_eval.label_prefix} thresholds={self.target_pass_at_k_thresholds}")
            print(f"[PassAtKCallback] Training will stop when hardest threshold is reached: {self.target_pass_at_k_thresholds[0]}")
        else:
            print(f"[PassAtKCallback] Initialized with early_tuples={self.early_tuples}")
            print(f"[PassAtKCallback] Training will stop when all early_tuples have triggered")

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
        print(f"[DEBUG] chat_template used for inference: {tokenizer.chat_template[:80]}...")
        print(f"[DEBUG] Sample prompt (index 0):")
        print(sample_formatted)
        print(f"{'='*60}\n")

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.model_name:
            self.model_name = kwargs.get("model")
        print(f"[PassAtKCallback] on_train_begin: model_name={self.model_name}")
        now = datetime.datetime.now().strftime("%m%d_%H%M%S")
        self.metadata_path = os.path.join(MODELS_METADATA_DIR, f"{self.model_name}_{self.primary_eval.id}_{self.primary_eval.stopping_metric()}-{now}.json")

        # Baseline evaluation before training starts
        self.on_evaluate(args, state, control, **kwargs)

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

    def set_trainer_vllm(self, llm):
        self._runner = ExternalVLLMRunner(self._runner_config, llm=llm)

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

    def _save_sweetspot_checkpoint(self, model, threshold, state: TrainerState, args: TrainingArguments):
        """Save a checkpoint when a sweetspot threshold is reached."""
        return save_sweetspot_checkpoint(
            model=model,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
            threshold_label=f"{self.primary_eval.label_prefix}-{threshold}",
            state=state,
            args=args,
            metadata_path=self.metadata_path,
            extra_metadata={
                "threshold_type": self.primary_eval.stopping_metric(),
                "threshold_value": threshold,
            },
        )

    def _log_raw_generation_table(
        self,
        eval_strategy: EvalStrategy,
        model_results: List[Dict],
        global_step: int,
        stopping_metric_name: str,
        stopping_metric_value: float | None,
    ) -> None:
        """Best-effort logging of raw generations as a per-step W&B table."""
        eval_slug = eval_strategy.id
        table_key = f"raw_generations/{eval_slug}/step_{global_step}"

        try:
            table = wandb.Table(columns=[
                "global_step",
                "eval_name",
                "prompt_index",
                "prompt",
                "responses",
                "num_responses",
                "per_response_correct",
                "per_response_instructions",
                "prompt_accuracy",
                "stopping_metric_name",
                "stopping_metric_value",
                "thresholds_remaining",
                "timestamp_utc",
            ])

            timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
            thresholds_remaining = json.dumps(self.target_pass_at_k_thresholds)

            for prompt_index, item in enumerate(model_results):
                prompt = item.get("prompt", "")
                responses = item.get("responses", [])
                if not isinstance(responses, list):
                    responses = [responses]

                try:
                    responses_json = json.dumps(responses)
                except TypeError:
                    responses_json = json.dumps([str(response) for response in responses])

                correctness = item.get("per_response_correct", [])
                prompt_accuracy = sum(correctness) / len(correctness) if correctness else None
                instructions = item.get("per_response_instructions", [])

                table.add_data(
                    global_step,
                    eval_slug,
                    prompt_index,
                    prompt,
                    responses_json,
                    len(responses),
                    json.dumps(correctness) if correctness else None,
                    json.dumps(instructions) if instructions else None,
                    prompt_accuracy,
                    stopping_metric_name,
                    stopping_metric_value,
                    thresholds_remaining,
                    timestamp_utc,
                )

            wandb.log({
                "train/global_step": global_step,
                "train/total_global_step": global_step + self._step_offset,
                table_key: table,
            })
        except Exception as exc:
            print(f"[PassAtKCallback] Warning: failed to log raw generation table ({table_key}): {exc}")

    def _save_adapter_if_needed(self, model, adapter_dir: str):
        if isinstance(self._runner, ExternalVLLMRunner):
            return None
        self._save_lora_adapter(model, adapter_dir)
        return adapter_dir

    def _run_eval_with_results(self, model, eval_strategy: EvalStrategy) -> tuple[Dict[str, float], List[Dict]]:
        """Run vLLM inference and score responses using the given eval strategy."""
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

    def _run_eval(self, model, eval_strategy: EvalStrategy) -> Dict[str, float]:
        """Run vLLM inference and score responses using the given eval strategy."""
        scores, _ = self._run_eval_with_results(model, eval_strategy)
        return scores

    def on_evaluate(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, model=None, **kwargs):
        """Called after evaluation, run evals and stop if target reached."""
        train_batch_size = args.per_device_train_batch_size
        grad_accum = args.gradient_accumulation_steps
        world_size = getattr(args, "world_size", 1)
        data_points_seen = state.global_step * train_batch_size * grad_accum * world_size

        if model is None:
            model = kwargs.get("model")
        if model is None:
            print("[PassAtKCallback] Warning: model is None, skipping eval")
            return control

        # Run primary eval
        scores, raw_results = self._run_eval_with_results(model, self.primary_eval)

        # Log primary eval metrics to wandb
        log_dict = {"train/global_step": state.global_step, "train/total_global_step": state.global_step + self._step_offset}
        log_dict.update(self.primary_eval.wandb_metrics(scores))
        wandb.log(log_dict)

        stopping_key = self.primary_eval.stopping_metric()
        stopping_value = scores[stopping_key]
        self.prevResults.append(stopping_value)
        self._log_raw_generation_table(
            eval_strategy=self.primary_eval,
            model_results=raw_results,
            global_step=state.global_step,
            stopping_metric_name=stopping_key,
            stopping_metric_value=stopping_value,
        )

        scores_str = ", ".join([f"{k}={v:.4f}" for k, v in scores.items() if isinstance(v, float)])
        print(f"\n[PassAtKCallback] Step {state.global_step}, Data Points {data_points_seen}: "
              f"{scores_str} ({scores.get('num_prompts_evaluated', '?')} prompts)")

        # Run monitor evals (wandb logging only, no stopping)
        for monitor_eval in self.monitor_evals:
            monitor_scores, monitor_raw_results = self._run_eval_with_results(model, monitor_eval)
            monitor_log = {"train/global_step": state.global_step, "train/total_global_step": state.global_step + self._step_offset}
            monitor_log.update(monitor_eval.wandb_metrics(monitor_scores))
            wandb.log(monitor_log)
            monitor_stopping_key = monitor_eval.stopping_metric()
            self._log_raw_generation_table(
                eval_strategy=monitor_eval,
                model_results=monitor_raw_results,
                global_step=state.global_step,
                stopping_metric_name=monitor_stopping_key,
                stopping_metric_value=monitor_scores.get(monitor_stopping_key),
            )
            monitor_str = ", ".join([f"{k}={v:.4f}" for k, v in monitor_scores.items() if isinstance(v, float)])
            print(f"[PassAtKCallback] Monitor ({monitor_eval.__class__.__name__}): {monitor_str}")

        decisions = self._decision_engine.decide(
            primary_metric=stopping_value,
            history=self.prevResults,
            data_points_seen=data_points_seen,
            last_checkpoint_data_points=self._last_checkpoint_data_points,
        )
        for decision in decisions:
            self._save_sweetspot_checkpoint(model, decision.label, state, args)
            if decision.advances_state:
                self._last_checkpoint_data_points = data_points_seen
            print(f"[PassAtKCallback] Saved checkpoint: {decision.label}")

        self._last_eval_step = state.global_step
        return control
