# ABOUTME: GRPO (RLVR) training using TRL's GRPOTrainer with standard HF/PEFT.
# ABOUTME: Mirrors dpo_training.py pattern with verifiable reward functions.

import time

import torch
import wandb

from tuning.config import MODELS_DIR
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import PTRunConfig, LoraConfig, ModelLoadConfig, GRPOTrainingConfig
from tuning.training.callback_utils import (
    CompletionsIntervalCallback,
    OffsetAwareWandbCallback,
    load_total_seconds_from_checkpoint,
    remove_default_wandb_callback,
)
from tuning.training.grpo_timing import GRPOStepTiming
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.training.model_utils import (
    load_model_with_lora,
    save_trained_model,
    top_layer_indices,
    upcast_lm_head_to_fp32,
    upcast_vllm_lm_head_to_fp32,
)
from tuning.training.server_rollouts import install_client_rendered_chat

from tuning.utils.utils import chat_template_func
from trl import GRPOTrainer, GRPOConfig
from typing import Callable, List
from tuning.config import HF_MODEL_MAP
import subprocess


def _enable_vllm_engine_stats():
    # TRL builds vllm.LLM() without passing disable_log_stats; vllm's entrypoint
    # then forces it to True, which propagates log_stats=False all the way into
    # the Scheduler so make_stats() returns None and no periodic engine stats
    # (throughput, KV-cache usage, queue depths) are ever logged. Override the
    # default so the Scheduler emits stats and LoggingStatLogger has data to log.
    from vllm import LLM
    orig_init = LLM.__init__
    def init(self, *args, **kwargs):
        kwargs.setdefault("disable_log_stats", False)
        return orig_init(self, *args, **kwargs)
    LLM.__init__ = init


_enable_vllm_engine_stats()


class _GRPOTrainer(GRPOTrainer):
    """GRPOTrainer with an extra metric: fraction of sequences masked by the vLLM IS ratio cap."""

    TIMING_ENABLED = True  # Change to False to disable all GRPO timing metrics.

    def __init__(self, *args, **kwargs):
        self._step_timing = GRPOStepTiming() if self.TIMING_ENABLED else None
        self._timing_in_rollout = False
        super().__init__(*args, **kwargs)
        if self.TIMING_ENABLED:
            self._install_timing_wrappers()

    def _install_timing_wrappers(self):
        """Time vLLM and backward calls without copying TRL's training loop."""
        if hasattr(self, "vllm_generation"):
            for method_name, timing_name in (
                ("sync_weights", "weight_sync"),
                ("generate", "vllm_generate"),
            ):
                original = getattr(self.vllm_generation, method_name, None)
                if original is None:
                    continue

                def timed_vllm_call(
                    *args,
                    _original=original,
                    _timing_name=timing_name,
                    **kwargs,
                ):
                    start = time.perf_counter()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        if self._timing_in_rollout and self.model.training:
                            self._step_timing.add(
                                _timing_name,
                                time.perf_counter() - start,
                            )

                setattr(self.vllm_generation, method_name, timed_vllm_call)

        original_backward = self.accelerator.backward

        def timed_backward(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original_backward(*args, **kwargs)
            finally:
                if self.model.training:
                    self._step_timing.add(
                        "backward",
                        time.perf_counter() - start,
                    )

        self.accelerator.backward = timed_backward

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        if not self.TIMING_ENABLED:
            return super()._get_per_token_logps_and_entropies(*args, **kwargs)
        start = time.perf_counter()
        try:
            return super()._get_per_token_logps_and_entropies(*args, **kwargs)
        finally:
            if self._timing_in_rollout and self.model.training:
                self._step_timing.add(
                    "rollout_logps",
                    time.perf_counter() - start,
                )

    def _calculate_rewards(self, *args, **kwargs):
        if not self.TIMING_ENABLED:
            return super()._calculate_rewards(*args, **kwargs)
        start = time.perf_counter()
        try:
            return super()._calculate_rewards(*args, **kwargs)
        finally:
            if self._timing_in_rollout and self.model.training:
                self._step_timing.add(
                    "reward",
                    time.perf_counter() - start,
                )

    def _generate_and_score_completions(self, *args, **kwargs):
        if not self.TIMING_ENABLED:
            return super()._generate_and_score_completions(*args, **kwargs)
        is_training = self.model.training
        previous_in_rollout = self._timing_in_rollout
        if is_training:
            self._timing_in_rollout = True
        start = time.perf_counter()
        output = None
        try:
            output = super()._generate_and_score_completions(*args, **kwargs)
            return output
        finally:
            if is_training:
                self._step_timing.add(
                    "rollout",
                    time.perf_counter() - start,
                )
                if output is not None:
                    generated_tokens = output.get("num_items_in_batch", 0.0)
                    if torch.is_tensor(generated_tokens):
                        generated_tokens = generated_tokens.detach().item()
                    self._step_timing.add("generated_tokens", generated_tokens)
            self._timing_in_rollout = previous_in_rollout

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        start = time.perf_counter() if self.TIMING_ENABLED else None
        try:
            if (
                self.use_vllm
                and self.vllm_importance_sampling_correction
                and self.vllm_importance_sampling_mode in ["sequence_mask", "token_mask"]
                and "importance_sampling_ratio" in inputs
            ):
                mode = "train" if self.model.training else "eval"
                is_ratio = inputs["importance_sampling_ratio"]
                frac_masked = (is_ratio == 0.0).float().mean()
                gathered = self.accelerator.gather(frac_masked)
                self._metrics[mode]["sampling/importance_sampling_ratio/frac_masked"].append(
                    gathered.nanmean().item()
                )
            return super().compute_loss(
                model, inputs, return_outputs, num_items_in_batch,
            )
        finally:
            if self.TIMING_ENABLED and self.model.training:
                self._step_timing.add(
                    "policy_forward_loss",
                    time.perf_counter() - start,
                )

    def training_step(self, model, inputs, num_items_in_batch):
        if not self.TIMING_ENABLED:
            return super().training_step(model, inputs, num_items_in_batch)
        start = time.perf_counter()
        output = super().training_step(model, inputs, num_items_in_batch)
        self._step_timing.add("step", time.perf_counter() - start)
        if self._step % self.current_gradient_accumulation_steps == 0:
            for name, value in self._step_timing.finish().items():
                self._metrics["train"][name].append(value)
        return output


def train_model_grpo(
    run_config: PTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: GRPOTrainingConfig = None,
    reward_funcs: List[Callable] = None,
    passk_config = None,
    primary_eval = None,
    monitor_evals = None,
    initial_global_step = None,
    lora_layers_fraction = 1.0,
):
    # Resolve model path: SFT checkpoint or base HF model
    if run_config.sft_run_config:
        if run_config.sft_run_config.dataset_config.dynamic_path:
            model_path = f"{MODELS_DIR}/{run_config.sft_run_config.dataset_config.dynamic_path}"
        else:
            model_path = f"{MODELS_DIR}/{run_config.sft_run_config.run_name}"
    else:
        model_path = run_config.model_name_hf

    initial_total_seconds = 0.0
    if run_config.sft_run_config and not training_args.resume_from_checkpoint:
        initial_total_seconds = load_total_seconds_from_checkpoint(model_path)

    raw_dataset = get_train_dataset(run_config)

    layers = None
    if lora_layers_fraction < 1.0:
        layers = top_layer_indices(run_config.model_name_hf, lora_layers_fraction)

    print(f"Loading model from {model_path}")
    model, tokenizer = load_model_with_lora(
        model_path=model_path,
        model_name=run_config.model_name,
        model_load_config=model_load_config,
        lora_config=lora_config,
        use_unsloth=False,
        layers_to_transform=layers,
    )
    tokenizer = chat_template_func(tokenizer)

    if training_args.upcast_lm_head_fp32:
        upcast_lm_head_to_fp32(model)
        print("[GRPO] upcast lm_head to fp32 on trainer")

    callbacks = [
        OffsetAwareWandbCallback(
            initial_global_step=initial_global_step or 0,
            initial_total_seconds=initial_total_seconds,
        )
    ]

    if passk_config is not None and passk_config.enabled:
        passk_callback = PassAtKStoppingCallback(
            config=passk_config,
            tokenizer=tokenizer,
            model_name=run_config.model_name,
            base_model_hf=model_path,
            primary_eval=primary_eval,
            monitor_evals=monitor_evals or [],
        )
        callbacks.append(passk_callback)

    trainer = _GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        train_dataset=raw_dataset["train"],
        eval_dataset=raw_dataset["test"],
        callbacks=callbacks if callbacks else None,
        args=GRPOConfig(
            **training_args.to_hf_args(output_dir=run_config.output_dir),
        ),
    )

    if trainer.log_completions:
        trainer.add_callback(CompletionsIntervalCallback(trainer, interval=64))

    vllm_mode = getattr(trainer, "vllm_mode", None)
    is_colocate = vllm_mode == "colocate" and hasattr(trainer, "vllm_generation")
    is_server = vllm_mode == "server" and hasattr(trainer, "vllm_generation")
    server_client = None
    if is_server:
        server_client = getattr(trainer.vllm_generation, "vllm_client", None)
        if trainer.accelerator.is_main_process and server_client is None:
            raise RuntimeError(
                "GRPO server mode initialized without a rank-0 VLLMClient. "
                "Start trl vllm-serve and pass --grpo-vllm-server-host/port."
            )
        if server_client is not None:
            install_client_rendered_chat(server_client, tokenizer)
            print("[GRPO] Server rollouts will render prompts with the trainer tokenizer "
                  "and use the vLLM /generate/ endpoint")

    for cb in callbacks or []:
        if isinstance(cb, PassAtKStoppingCallback):
            if is_colocate:
                cb.set_trainer_vllm(
                    trainer.vllm_generation.llm,
                    vllm_generation=trainer.vllm_generation,
                )
                print(f"[GRPO] PassAtK callback will reuse GRPOTrainer's colocate vLLM engine")
            elif is_server:
                # rank 0 owns vllm_client; other ranks pass None and receive results via broadcast
                cb.set_trainer_vllm_client(server_client)
                print(f"[GRPO] PassAtK callback will reuse GRPOTrainer's vLLM server "
                      f"(rank-0 client {'attached' if server_client is not None else 'absent'})")
            if is_colocate or is_server:
                cb._trainer_vllm_generation = trainer.vllm_generation
            cb._accelerator = trainer.accelerator

    if is_colocate and training_args.upcast_lm_head_fp32:
        llm = trainer.vllm_generation.llm
        if getattr(trainer.vllm_generation, "enable_sleep_mode", False):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            llm.wake_up(tags=["weights"])
            try:
                upcast_vllm_lm_head_to_fp32(llm)
            finally:
                llm.sleep(level=2)
        else:
            upcast_vllm_lm_head_to_fp32(llm)
        print("[GRPO] upcast lm_head to fp32 on vLLM engine")
    elif is_server and training_args.upcast_lm_head_fp32:
        print("[GRPO] WARNING: trainer lm_head is fp32; server-side lm_head dtype "
              "is controlled by vLLM storage during weight sync")

    remove_default_wandb_callback(trainer)

    try:
        trainer_stats = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint,
        )
    except KeyboardInterrupt:
        if wandb.run:
            wandb.run.tags = list(wandb.run.tags) + ["interrupted"]
        raise
    except torch.cuda.OutOfMemoryError:
        print(subprocess.check_output("nvidia-smi").decode())
        if wandb.run:
            wandb.run.tags = list(wandb.run.tags) + ["oom"]
        raise

    if trainer.accelerator.is_main_process:
        unwrapped = trainer.accelerator.unwrap_model(model)
        save_trained_model(unwrapped, tokenizer, trainer, run_config.output_dir)
    trainer.accelerator.wait_for_everyone()

    return model, tokenizer, trainer, callbacks
