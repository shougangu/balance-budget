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
from tuning.training.grpo_effective_batch import (
    nonzero_variance_row_mask,
    select_row_aligned,
)
from tuning.training.passk_callback import PassAtKStoppingCallback
from tuning.training.model_utils import (
    load_model_with_lora,
    save_trained_model,
    top_layer_indices,
    install_trl_vllm_dtype_patch,
    install_vllm_fp32_logits_patch,
    load_trainable_adapter,
    sync_colocated_vllm_chat_template,
    upcast_lm_head_to_fp32,
)
from tuning.training.server_rollouts import install_client_rendered_chat

from tuning.utils.utils import chat_template_func
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.utils import RepeatSampler
from transformers.trainer_utils import get_last_checkpoint
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


class _EpochSeededRepeatSampler(RepeatSampler):
    """RepeatSampler whose historical per-epoch order can be reconstructed.

    RepeatSampler seeds its generator once and advances it once per epoch. A
    fresh sampler therefore cannot jump directly to a resumed epoch. Replaying
    the preceding permutation draws reconstructs that epoch without changing
    the order of uninterrupted or pre-existing runs.

    Epochs are monotonic for this training sampler because Accelerate's
    mid-epoch skip wrapper starts with iteration zero and otherwise overwrites
    the epoch that Trainer just restored.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = max(self.epoch, epoch)

    def __iter__(self):
        if self.shuffle and self.seed is not None:
            self.generator.manual_seed(self.seed)
            for _ in range(self.epoch):
                torch.randperm(self.num_samples, generator=self.generator)
        yield from super().__iter__()
        self.epoch += 1


class _GRPOTrainer(GRPOTrainer):
    """GRPOTrainer with an extra metric: fraction of sequences masked by the vLLM IS ratio cap."""

    TIMING_ENABLED = True  # Change to False to disable all GRPO timing metrics.

    def __init__(self, *args, **kwargs):
        self.zero_variance_filter = kwargs.pop("zero_variance_filter", True)
        self.zero_variance_filter_epsilon = kwargs.pop("zero_variance_filter_epsilon", 0.0)
        self._step_timing = GRPOStepTiming() if self.TIMING_ENABLED else None
        self._timing_in_rollout = False
        super().__init__(*args, **kwargs)
        if self.TIMING_ENABLED:
            self._install_timing_wrappers()

    def _get_train_sampler(self, dataset=None):
        """Reconstruct each epoch's historical prompt order so resumes can skip exactly.

        Rebuilt from TRL's sampler so its batch-size arithmetic stays in one place.
        """
        sampler = super()._get_train_sampler(dataset)
        return _EpochSeededRepeatSampler(
            data_source=sampler.data_source,
            mini_repeat_count=sampler.mini_repeat_count,
            batch_size=sampler.batch_size,
            repeat_count=sampler.repeat_count,
            shuffle=sampler.shuffle,
            seed=sampler.seed,
        )

    def _compute_zero_variance_keep_mask(self, output, mode):
        advantages = output.get("advantages")
        if not torch.is_tensor(advantages) or advantages.ndim == 0:
            return None, None

        num_generations = self.num_generations if mode == "train" else self.num_generations_eval
        gathered_advantages = self.accelerator.gather(advantages.detach())
        try:
            global_keep = nonzero_variance_row_mask(
                gathered_advantages,
                num_generations=num_generations,
                epsilon=self.zero_variance_filter_epsilon,
            )
        except ValueError:
            return None, None

        local_rows = advantages.shape[0]
        process_slice = slice(
            self.accelerator.process_index * local_rows,
            (self.accelerator.process_index + 1) * local_rows,
        )
        return global_keep[process_slice].to(device=advantages.device), global_keep

    def _annotate_zero_variance_filter(self, output, mode):
        if not self.zero_variance_filter or mode != "train":
            return output

        local_keep, global_keep = self._compute_zero_variance_keep_mask(output, mode)
        if local_keep is None:
            return output

        output["zero_variance_filter_mask"] = local_keep

        loss_mask = output["completion_mask"]
        if "tool_mask" in output:
            loss_mask = loss_mask * output["tool_mask"]
        local_effective_tokens = loss_mask[local_keep].sum()
        effective_tokens = self.accelerator.gather(local_effective_tokens).sum()
        output["num_items_in_batch"] = effective_tokens.clamp(min=1)

        group_keep = global_keep.view(-1, self.num_generations).any(dim=1)
        self._metrics[mode]["effective_batch/nonzero_variance_prompt_frac"].append(
            group_keep.float().mean().item()
        )
        self._metrics[mode]["effective_batch/dropped_prompt_count"].append(
            (~group_keep).sum().item()
        )
        self._metrics[mode]["effective_batch/effective_completion_tokens"].append(
            effective_tokens.item()
        )
        return output

    def _zero_dummy_inputs(self, inputs, row_count):
        dummy = select_row_aligned(inputs, slice(0, 1), row_count)
        dummy["advantages"] = torch.zeros_like(dummy["advantages"])
        dummy["completion_mask"] = torch.zeros_like(dummy["completion_mask"])
        if "tool_mask" in dummy:
            dummy["tool_mask"] = torch.zeros_like(dummy["tool_mask"])
        if "zero_variance_filter_mask" in dummy:
            dummy["zero_variance_filter_mask"] = torch.zeros_like(
                dummy["zero_variance_filter_mask"],
                dtype=torch.bool,
            )
        return dummy

    def _filter_zero_variance_inputs(self, inputs):
        keep_mask = inputs.get("zero_variance_filter_mask")
        if not torch.is_tensor(keep_mask):
            return inputs, False

        keep_mask = keep_mask.bool()
        row_count = keep_mask.shape[0]
        local_kept = keep_mask.sum()
        global_kept = self.accelerator.gather(local_kept).sum()
        mode = "train" if self.model.training else "eval"

        if global_kept.item() == 0:
            self._metrics[mode]["effective_batch/empty_microbatch"].append(1.0)
            return inputs, True

        self._metrics[mode]["effective_batch/empty_microbatch"].append(0.0)
        if local_kept.item() == 0:
            return self._zero_dummy_inputs(inputs, row_count), False
        if local_kept.item() == row_count:
            return inputs, False
        return select_row_aligned(inputs, keep_mask, row_count), False

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
            output = super()._generate_and_score_completions(*args, **kwargs)
            mode = "train" if self.model.training else "eval"
            return self._annotate_zero_variance_filter(output, mode)
        is_training = self.model.training
        previous_in_rollout = self._timing_in_rollout
        if is_training:
            self._timing_in_rollout = True
        start = time.perf_counter()
        output = None
        try:
            output = super()._generate_and_score_completions(*args, **kwargs)
            mode = "train" if self.model.training else "eval"
            return self._annotate_zero_variance_filter(output, mode)
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
            if return_outputs:
                return super().compute_loss(
                    model, inputs, return_outputs, num_items_in_batch,
                )
            inputs, empty_effective_batch = self._filter_zero_variance_inputs(inputs)
            if empty_effective_batch:
                return torch.zeros((), device=self.accelerator.device, requires_grad=True)
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

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Colocate sleep mode discards vLLM weights after every generation, and TRL only re-syncs
        # them when global_step advances. The eval split runs at the global_step the preceding
        # training rollout already synced, so the sync is skipped and each eval batch generates with
        # stale (non-trained) weights, collapsing eval reward. Reset the sync marker so every eval
        # batch reloads the current policy weights into vLLM, mirroring each training rollout.
        vllm_generation = getattr(self, "vllm_generation", None)
        if vllm_generation is not None and getattr(vllm_generation, "enable_sleep_mode", False):
            self._last_loaded_step = -1
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

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
            time_multiplier=float(training_args.num_compute_gpus),
        )
    ]


    precision_dtype_names = {"fp16": "float16", "bf16": "bfloat16"}
    vllm_dtype_name = precision_dtype_names.get(training_args.precision)
    if training_args.use_vllm and training_args.vllm_mode == "colocate":
        if training_args.upcast_lm_head_fp32:
            installed = install_vllm_fp32_logits_patch()
            action = "installed" if installed else "already installed"
            print(f"[GRPO] {action} vLLM fp32 logits patch")
        if vllm_dtype_name is not None:
            install_trl_vllm_dtype_patch(vllm_dtype_name)
            print(f"[GRPO] forcing colocated vLLM dtype={vllm_dtype_name}")
    elif training_args.use_vllm and training_args.vllm_mode == "server":
        if training_args.upcast_lm_head_fp32:
            print("[GRPO] WARNING: server-mode fp32 logits patching must be "
                  "configured in the vLLM server process")
        if vllm_dtype_name is not None:
            print(f"[GRPO] WARNING: server-mode vLLM dtype must be "
                  f"configured in the vLLM server process (expected {vllm_dtype_name})")

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
        zero_variance_filter=training_args.zero_variance_filter,
        zero_variance_filter_epsilon=training_args.zero_variance_filter_epsilon,
    )

    resume_checkpoint = training_args.resume_from_checkpoint
    if resume_checkpoint is True:
        resume_checkpoint = get_last_checkpoint(run_config.output_dir)
        if resume_checkpoint is None:
            raise ValueError(
                f"No valid checkpoint found in {run_config.output_dir}"
            )
    if resume_checkpoint:
        # GRPO has now created ref from the original policy; restore default afterward.
        load_trainable_adapter(trainer.model, resume_checkpoint)
        print(f"[GRPO resume] loaded trainable policy from {resume_checkpoint}")

    if trainer.log_completions:
        trainer.add_callback(CompletionsIntervalCallback(trainer, interval=64))

    if sync_colocated_vllm_chat_template(trainer, tokenizer):
        print("[GRPO] Colocated vLLM will use trainer tokenizer chat_template")

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
                    trainer=trainer,
                )
                print(f"[GRPO] PassAtK callback will reuse GRPOTrainer's colocate vLLM engine")
            elif is_server:
                # rank 0 owns vllm_client; other ranks pass None and receive results via broadcast
                cb.set_trainer_vllm_client(
                    server_client,
                    vllm_generation=trainer.vllm_generation,
                    trainer=trainer,
                )
                print(f"[GRPO] PassAtK callback will reuse GRPOTrainer's vLLM server "
                      f"(rank-0 client {'attached' if server_client is not None else 'absent'})")
            if is_colocate or is_server:
                cb._trainer_vllm_generation = trainer.vllm_generation
            cb._accelerator = trainer.accelerator


    remove_default_wandb_callback(trainer)

    try:
        trainer_stats = trainer.train(
            resume_from_checkpoint=resume_checkpoint,
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
