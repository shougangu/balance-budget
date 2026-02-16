import os
import warnings

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

print(ROOT_DIR)

DATA_DIR = "/data"

DATASETS_DIR = ROOT_DIR + DATA_DIR + "/datasets"
MODELS_DIR = ROOT_DIR + "/models" 
MODELS_METADATA_DIR = ROOT_DIR + "/models_metadata"
OUTPUTS_DIR = ROOT_DIR + "/outputs"

IFEVAL_OUTPUTS_DIR = OUTPUTS_DIR + "/ifeval"
GSM8K_OUTPUTS_DIR = OUTPUTS_DIR + "/gsm8k"

CHATML_TEMPLATE_PATH = ROOT_DIR + "/chatml.py"

PREF_DATASETS_DIR = DATASETS_DIR + "/preference_dataset_outputs"

RESPONSES_FILENAME = "responses.jsonl"
RESPONSES_ANNOTATED_FILENAME = "responses_annotated.jsonl"

HF_MODEL_MAP = {
    "llama3-8B": "unsloth/Meta-Llama-3.1-8B",
    "qwen2-7B": "unsloth/Qwen2.5-7B",
    "qwen2-3B": "unsloth/Qwen2.5-3B",
    "qwen2-2B": "unsloth/Qwen2.5-1.5B",
    "qwen2-14B": "unsloth/Qwen2.5-14B",
    "llama3-3B": "unsloth/Llama-3.2-3B",
    "llama3-1B": "unsloth/Llama-3.2-1B"
}

DEFAULT_CHAT_TEMPLATE = "chatml"

MODEL_CHAT_TEMPLATE_MAP = {
    "llama3-8B": "llama-3.1",
    "llama3-3B": "llama-3.1",
    "llama3-1B": "llama-3.1",
    "qwen2-14B": "chatml",
    "qwen2-7B": "chatml",
    "qwen2-3B": "chatml",
    "qwen2-2B": "chatml",
}


def resolve_chat_template(model_name: str, override: str = None) -> str:
    if override:
        return override
    if model_name in MODEL_CHAT_TEMPLATE_MAP:
        return MODEL_CHAT_TEMPLATE_MAP[model_name]

    # Support derived run names like "llama3-8B_sft-..._pt-..."
    base_model_name = model_name.split("_", 1)[0]
    if base_model_name in MODEL_CHAT_TEMPLATE_MAP:
        return MODEL_CHAT_TEMPLATE_MAP[base_model_name]

    warnings.warn(
        f"No chat template mapping found for model '{model_name}'. "
        f"Falling back to default template '{DEFAULT_CHAT_TEMPLATE}'.",
        UserWarning,
        stacklevel=2,
    )
    return DEFAULT_CHAT_TEMPLATE
