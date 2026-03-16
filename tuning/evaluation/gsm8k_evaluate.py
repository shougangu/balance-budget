# ABOUTME: GSM8K evaluation via lm-eval harness using VLLMSamplingParamsConfig.
# ABOUTME: Configurable to match eval_strategy.py sampling params for equivalence testing.

from tuning.config import GSM8K_OUTPUTS_DIR, MODELS_DIR, RESPONSES_FILENAME
import subprocess
from tuning.inference.config_inference import VLLMSamplingParamsConfig
from tuning.data.config import SYSTEM_MESSAGE_GSM8K
import os

TASKS_DIR = os.path.join(os.path.dirname(__file__), "tasks")


def gsm8k_evaluate(model_name: str, temperature: float = 0.5, max_tokens: int = 1024,
                    gpu_memory_utilization: float = 0.9):

    print(f"Running GSM8K evaluation for {model_name}")

    model_path = f"{MODELS_DIR}/{model_name}"
    output_path = f"{GSM8K_OUTPUTS_DIR}/{model_name}/{RESPONSES_FILENAME}"

    sampling_params = VLLMSamplingParamsConfig(
        max_tokens=max_tokens,
        temperature=temperature,
    ).model_dump()
    print(model_path)

    try:
        result = subprocess.run(
            [
                "lm-eval",
                "--tasks", "gsm8k_matched",
                "--include_path", TASKS_DIR,
                "--model", "vllm",
                "--model_args", f"pretrained={model_path},max_gen_toks={sampling_params['max_tokens']},enforce_eager=True,gpu_memory_utilization={gpu_memory_utilization}",
                f"--output_path={output_path}",
                "--log_samples",
                "--batch_size", "auto:N",
                "--gen_kwargs", f"temperature={sampling_params['temperature']},top_p={sampling_params['top_p']},top_k={sampling_params['top_k']},skip_special_tokens=true",
                "--verbosity", "DEBUG",
                "--write_out",
                "--apply_chat_template", "true",
                "--system_instruction", SYSTEM_MESSAGE_GSM8K,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return

    print(model_name)
    print(output_path)
    return output_path

if __name__ == "__main__":
    print("Starting GSM8K evaluation")
    for model in ["llama3-3B", "llama3-8B", "qwen2-2B", "qwen2-3B", "qwen2-7B"]:
        gsm8k_evaluate(model)
