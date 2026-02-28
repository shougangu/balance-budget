# ABOUTME: HuggingFace TrainerCallback that runs vLLM inference during training.
# ABOUTME: Saves checkpoints at metric sweetspots using pluggable EvalStrategy objects.

import torch
import wandb
import tempfile
import shutil
import os
import datetime
import multiprocessing as mp
from typing import List, Dict
from pathlib import Path
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from collections import defaultdict
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from tuning.config import MODELS_METADATA_DIR
from tuning.inference.config_inference import VLLMSamplingParamsConfig
from tuning.utils.gpu import cleanup_gpu
from tuning.training.callback_utils import save_sweetspot_checkpoint
from tuning.training.eval_strategy import EvalStrategy


def partition_prompts(messages: List, num_chunks: int) -> List[List]:
    """Split a list of messages into num_chunks roughly-equal chunks.

    If num_chunks > len(messages), only len(messages) chunks are returned (1 item each).
    """
    n = len(messages)
    num_chunks = min(num_chunks, n)
    chunks = []
    base_size = n // num_chunks
    remainder = n % num_chunks
    start = 0
    for i in range(num_chunks):
        size = base_size + (1 if i < remainder else 0)
        chunks.append(messages[start:start + size])
        start += size
    return chunks


def _data_parallel_worker(worker_id, cuda_device, messages_chunk, base_model_hf, adapter_path,
                          n_samples, temperature, max_tokens, chat_template,
                          lora_max_rank, gpu_memory_utilization, result_queue,
                          stop_tokens=None):
    """Worker function for data-parallel vLLM inference. Runs in a subprocess.

    Each worker pins itself to a single GPU, creates an ephemeral vLLM engine,
    runs inference on its chunk of prompts, and returns serialized outputs.

    Args:
        worker_id: Logical worker index (0, 1, 2...) used for result ordering.
        cuda_device: The actual CUDA device string (e.g. "3") from SLURM allocation.
    """
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        llm = LLM(
            model=base_model_hf,
            enable_lora=True,
            max_lora_rank=lora_max_rank,
            max_loras=1,
            gpu_memory_utilization=0.75,
            trust_remote_code=True,
            enforce_eager=True,
        )

        sp_kwargs = dict(
            n=n_samples,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if stop_tokens:
            sp_kwargs["stop"] = stop_tokens
        sampling_params = SamplingParams(**sp_kwargs)

        lora_request = LoRARequest(
            lora_name=f"adapter_worker{worker_id}",
            lora_int_id=1,
            lora_path=adapter_path,
        )

        outputs = llm.chat(
            messages_chunk,
            sampling_params,
            chat_template=chat_template,
            lora_request=lora_request,
        )

        # Serialize outputs: extract text from each output
        serialized = []
        for output in outputs:
            texts = [resp.text for resp in output.outputs]
            serialized.append(texts)

        # Cleanup
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        result_queue.put((worker_id, serialized, None))
    except Exception as e:
        import traceback
        result_queue.put((worker_id, None, traceback.format_exc()))


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
        # Sort thresholds in descending order (hardest to easiest: 0.7, 0.5, 0.3)
        # Higher pass@k = harder to reach, so we process from largest to smallest
        self.target_pass_at_k_thresholds = sorted(config.target_pass_at_k, reverse=True)
        self.early_tuples = list(config.early_tuples) if config.early_tuples else None
        self.tokenizer = tokenizer
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.model_name = model_name
        self.metadata_path = None
        self.prevResults = []
        self._last_eval_step = -1

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
        self.base_model_hf = base_model_hf
        self.vllm_gpu_memory_utilization = config.vllm_gpu_memory_utilization
        self.lora_max_rank = getattr(config, 'lora_max_rank', 32)
        self._vllm_engine = None
        self._lora_request_id = 0
        self._chat_template = self.tokenizer.chat_template

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
        print(f"[PassAtKCallback] vLLM mode: {mode_str}{parallelism_str}, base_model_hf={base_model_hf}, gpu_mem={self.vllm_gpu_memory_utilization}")
        print(f"[PassAtKCallback] Chat template: {self._chat_template}")

        # Log a sample formatted prompt to verify template
        sample_messages = primary_eval.get_test_messages()[0]
        sample_formatted = self.tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=True
        )
        print(f"\n{'='*60}")
        print(f"[DEBUG] chat_template used for inference: {self._chat_template[:80]}...")
        print(f"[DEBUG] Sample prompt (index 0):")
        print(sample_formatted)
        print(f"{'='*60}\n")

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.model_name:
            self.model_name = kwargs.get("model")
        print(f"[PassAtKCallback] on_train_begin: model_name={self.model_name}")
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        self.metadata_path = os.path.join(MODELS_METADATA_DIR, f"{self.model_name}_passatk-{now}.json")

    def on_train_end(self, args, state, control, **kwargs):
        """Cleanup persistent vLLM engine when training ends."""
        if self._last_eval_step != state.global_step:
            model = kwargs.pop("model", None)
            if model is not None:
                print(f"[PassAtKCallback] Running final evaluation at end of training (step {state.global_step})...")
                self.on_evaluate(args, state, control, model=model, **kwargs)
            else:
                print("[PassAtKCallback] Warning: model is None at on_train_end, skipping final evaluation")

        self._cleanup_vllm()

    def _init_persistent_vllm(self):
        """Lazily initialize the persistent vLLM engine with LoRA support."""
        if self._vllm_engine is not None:
            return

        print(f"[PassAtKCallback] Initializing persistent vLLM engine with base model: {self.base_model_hf}")
        print(f"[PassAtKCallback] gpu_memory_utilization={self.vllm_gpu_memory_utilization}, max_lora_rank={self.lora_max_rank}")

        # enforce_eager=True is required for LoRA — CUDA graph capture is incompatible with dynamic adapter swapping
        self._vllm_engine = LLM(
            model=self.base_model_hf,
            enable_lora=True,
            max_lora_rank=self.lora_max_rank,
            max_loras=1,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
            # max_model_len=2048,
        )

        print(f"[PassAtKCallback] Persistent vLLM engine initialized successfully")

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

    def _run_vllm_inference(self, llm, eval_strategy: EvalStrategy, adapter_path: str = None) -> List[Dict]:
        """Run inference on a vLLM engine, optionally with a LoRA adapter."""
        inference_config = VLLMSamplingParamsConfig(
            n=eval_strategy.n_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        sampling_params = SamplingParams(**inference_config.model_dump())

        lora_request = None
        if adapter_path:
            self._lora_request_id += 1
            lora_request = LoRARequest(
                lora_name=f"adapter_{self._lora_request_id}",
                lora_int_id=self._lora_request_id,
                lora_path=adapter_path,
            )

        test_messages = eval_strategy.get_test_messages()
        mode = "persistent" if self.use_persistent_vllm else "ephemeral"
        lora_info = f", lora_id={self._lora_request_id}" if lora_request else ""
        print(f"[PassAtKCallback] Generating {len(test_messages)} prompts x {eval_strategy.n_samples} samples ({mode}{lora_info})...")

        outputs = llm.chat(
            test_messages,
            sampling_params,
            chat_template=self._chat_template,
            lora_request=lora_request,
        )

        return self._format_outputs(outputs, eval_strategy)

    def _create_ephemeral_vllm(self):
        """Create an ephemeral vLLM engine with LoRA support."""
        print(f"[PassAtKCallback] Loading ephemeral vLLM with base model: {self.base_model_hf}")
        return LLM(
            model=self.base_model_hf,
            enable_lora=True,
            max_lora_rank=self.lora_max_rank,
            max_loras=1,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
        )

    def _cleanup_ephemeral_vllm(self, llm):
        """Destroy an ephemeral vLLM engine and free GPU memory."""
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
        del llm
        cleanup_gpu()

    def _cleanup_vllm(self):
        """Destroy the persistent vLLM engine and free GPU memory."""
        if self._vllm_engine is None:
            return

        print("[PassAtKCallback] Cleaning up persistent vLLM engine...")
        try:
            # Explicitly stop executor workers before dropping references.
            llm_engine = getattr(self._vllm_engine, "llm_engine", None)
            if llm_engine is not None:
                executor = getattr(llm_engine, "model_executor", None)
                if executor is not None:
                    executor.shutdown()
        finally:
            del self._vllm_engine
            self._vllm_engine = None

            # Full vLLM + distributed cleanup (not just model-parallel groups).
            from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
            cleanup_dist_env_and_memory(shutdown_ray=False)

            cleanup_gpu()
            print("[PassAtKCallback] vLLM engine cleaned up")

    def _run_data_parallel_inference(self, eval_strategy: EvalStrategy, adapter_path: str) -> List[Dict]:
        """Run data-parallel vLLM inference across multiple GPUs.

        Partitions prompts into chunks, spawns one vLLM worker process per GPU,
        and merges results in original prompt order.

        Uses self._available_gpus (SLURM-assigned device IDs) so workers pin
        to the correct physical GPUs regardless of SLURM allocation.
        """
        all_messages = eval_strategy.get_test_messages()
        all_prompts = eval_strategy.get_test_prompts()
        num_gpus = self.num_inference_gpus

        # Resolve actual CUDA device IDs from the SLURM/system allocation.
        # The parent process may have restricted CUDA_VISIBLE_DEVICES to GPU 0
        # for training, so we read the full list that was saved at init time.
        available_gpus = self._available_gpus
        if len(available_gpus) < num_gpus:
            print(f"[PassAtKCallback] WARNING: requested {num_gpus} inference GPUs but only "
                  f"{len(available_gpus)} available ({available_gpus}). Using {len(available_gpus)}.")
            num_gpus = len(available_gpus)

        message_chunks = partition_prompts(all_messages, num_gpus)
        prompt_chunks = partition_prompts(all_prompts, num_gpus)
        actual_num_workers = len(message_chunks)

        print(f"[PassAtKCallback] Data-parallel inference: {len(all_messages)} prompts across {actual_num_workers} GPUs")
        for i, chunk in enumerate(message_chunks):
            print(f"[PassAtKCallback]   Worker {i} → CUDA device {available_gpus[i]}: {len(chunk)} prompts")

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        # Compute stop tokens here since subprocess won't have the global set
        from tuning.utils.utils import get_stop_tokens
        stop_tokens = get_stop_tokens()

        for i in range(actual_num_workers):
            p = ctx.Process(
                target=_data_parallel_worker,
                args=(
                    i, available_gpus[i], message_chunks[i], self.base_model_hf,
                    adapter_path, eval_strategy.n_samples, self.temperature, self.max_tokens,
                    self._chat_template, self.lora_max_rank,
                    self.vllm_gpu_memory_utilization, result_queue,
                    stop_tokens, # picked up in subprocess since global may not be set due to spawn context
                ),
            )
            p.start()
            processes.append(p)

        # Collect results
        results_by_worker = {}
        for _ in range(actual_num_workers):
            worker_id, serialized, error = result_queue.get()
            if error is not None:
                # Terminate remaining workers
                for p in processes:
                    if p.is_alive():
                        p.terminate()
                raise RuntimeError(f"[PassAtKCallback] Worker {worker_id} failed:\n{error}")
            results_by_worker[worker_id] = serialized

        for p in processes:
            p.join(timeout=30)

        # Merge results in original prompt order
        merged = []
        for worker_id in range(actual_num_workers):
            chunk_texts = results_by_worker[worker_id]  # List of List[str]
            chunk_prompts = prompt_chunks[worker_id]
            for prompt, response_texts in zip(chunk_prompts, chunk_texts):
                merged.append({"prompt": prompt, "responses": response_texts})

        # Group by prompt (in case of duplicates)
        grouped = defaultdict(list)
        for item in merged:
            grouped[item["prompt"]].extend(item["responses"])

        print(f"[PassAtKCallback] Data-parallel inference complete: {len(grouped)} unique prompts")
        return [{"prompt": p, "responses": resps} for p, resps in grouped.items()]

    def _format_outputs(self, outputs, eval_strategy: EvalStrategy) -> List[Dict]:
        """Format vLLM outputs into grouped results for evaluation."""
        n_samples = eval_strategy.n_samples
        if n_samples == 1:
            responses = [output.outputs[0].text for output in outputs]
        else:
            responses = [[response.text for response in output.outputs] for output in outputs]

        test_prompts = eval_strategy.get_test_prompts()
        grouped = defaultdict(list)
        for prompt, resp in zip(test_prompts, responses):
            if isinstance(resp, list):
                grouped[prompt].extend(resp)
            else:
                grouped[prompt].append(resp)

        return [{"prompt": p, "responses": resps} for p, resps in grouped.items()]

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

    def _run_eval(self, model, eval_strategy: EvalStrategy) -> Dict[str, float]:
        """Run vLLM inference and score responses using the given eval strategy."""

        temp_dir = tempfile.mkdtemp()

        try:
            self._save_lora_adapter(model, temp_dir)

            if self.num_inference_gpus > 1:
                # Data-parallel mode: spawn N vLLM workers across GPUs
                original_device = next(model.parameters()).device
                model.cpu()
                torch.cuda.empty_cache()
                print(f"[PassAtKCallback] Training model offloaded to CPU for {self.num_inference_gpus}-GPU data-parallel inference")
                model_results = self._run_data_parallel_inference(eval_strategy, adapter_path=temp_dir)
                model.to(original_device)
                model.train()
            elif self.use_persistent_vllm:
                # Persistent mode: keep vLLM engine alive, swap LoRA adapters
                try:
                    self._init_persistent_vllm()
                    model_results = self._run_vllm_inference(self._vllm_engine, eval_strategy, adapter_path=temp_dir)
                except Exception as e:
                    print(f"[PassAtKCallback] Persistent vLLM failed: {e}, falling back to ephemeral mode")
                    self._cleanup_vllm()
                    self.use_persistent_vllm = False
                    # Fall through to ephemeral path
                    original_device = next(model.parameters()).device
                    model.cpu()
                    torch.cuda.empty_cache()
                    llm = self._create_ephemeral_vllm()
                    model_results = self._run_vllm_inference(llm, eval_strategy, adapter_path=temp_dir)
                    self._cleanup_ephemeral_vllm(llm)
                    model.to(original_device)
                    model.train()
            else:
                # Ephemeral mode: create/destroy vLLM each eval
                original_device = next(model.parameters()).device
                model.cpu()
                torch.cuda.empty_cache()
                llm = self._create_ephemeral_vllm()
                model_results = self._run_vllm_inference(llm, eval_strategy, adapter_path=temp_dir)
                self._cleanup_ephemeral_vllm(llm)
                model.to(original_device)
                model.train()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Score responses using the eval strategy
        print(f"[PassAtKCallback] Scoring responses with {eval_strategy.__class__.__name__}...")
        scores = eval_strategy.score_responses(model_results, self.tokenizer)
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
        scores = self._run_eval(model, self.primary_eval)

        # Log primary eval metrics to wandb
        log_dict = {"train/global_step": state.global_step}
        log_dict.update(self.primary_eval.wandb_metrics(scores))
        wandb.log(log_dict)

        stopping_key = self.primary_eval.stopping_metric()
        stopping_value = scores[stopping_key]
        self.prevResults.append(stopping_value)

        scores_str = ", ".join([f"{k}={v:.4f}" for k, v in scores.items() if isinstance(v, float)])
        print(f"\n[PassAtKCallback] Step {state.global_step}, Data Points {data_points_seen}: "
              f"{scores_str} ({scores.get('num_prompts_evaluated', '?')} prompts)")

        # Run monitor evals (wandb logging only, no stopping)
        for monitor_eval in self.monitor_evals:
            monitor_scores = self._run_eval(model, monitor_eval)
            monitor_log = {"train/global_step": state.global_step}
            monitor_log.update(monitor_eval.wandb_metrics(monitor_scores))
            wandb.log(monitor_log)
            monitor_str = ", ".join([f"{k}={v:.4f}" for k, v in monitor_scores.items() if isinstance(v, float)])
            print(f"[PassAtKCallback] Monitor ({monitor_eval.__class__.__name__}): {monitor_str}")

        # Check each threshold and save checkpoint if crossed (Fork Strategy)
        # Thresholds are sorted descending (hardest to easiest: 0.7, 0.5, 0.3)
        # We iterate to find the hardest threshold that current metric has reached
        if not self.early_tuples:
            reached_threshold = None
            reached_index = None

            for i, threshold in enumerate(self.target_pass_at_k_thresholds):
                if stopping_value >= threshold:
                    reached_threshold = threshold
                    reached_index = i
                    break

            if reached_threshold is not None:
                print(f"[PassAtKCallback] Sweetspot threshold {reached_threshold} reached!")
                checkpoint_path = self._save_sweetspot_checkpoint(model, reached_threshold, state, args)

                # Trim thresholds to only include harder ones (before current index)
                self.target_pass_at_k_thresholds = self.target_pass_at_k_thresholds[:reached_index]
                print(f"[PassAtKCallback] Remaining thresholds: {self.target_pass_at_k_thresholds}")

                if len(self.target_pass_at_k_thresholds) == 0:
                    print(f"[PassAtKCallback] All thresholds reached! Stopping training.")
                    control.should_training_stop = True
                else:
                    print(f"[PassAtKCallback] Continuing training to next threshold: {self.target_pass_at_k_thresholds[0]}")

        if self.early_tuples is not None:
            triggered = []
            for idx, (patience, min_increase) in enumerate(self.early_tuples):
                if len(self.prevResults) > patience:
                    early_stopping = True
                    for old, new in zip(self.prevResults[-patience-1:], self.prevResults[-patience:]):
                        if new - old >= min_increase:
                            early_stopping = False
                            break
                    if early_stopping:
                        label = f"{patience}@{min_increase}"
                        checkpoint_path = self._save_sweetspot_checkpoint(model, label, state, args)
                        print(f"[PassAtKCallback] Early tuple ({patience}, {min_increase}) triggered. Checkpoint: {checkpoint_path}")
                        triggered.append(idx)
            for idx in reversed(triggered):
                self.early_tuples.pop(idx)
            if len(self.early_tuples) == 0:
                print(f"[PassAtKCallback] All early_tuples triggered! Stopping training.")
                control.should_training_stop = True

        self._last_eval_step = state.global_step
        return control
