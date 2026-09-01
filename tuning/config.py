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
    "llama3-8B-instruct": "unsloth/Meta-Llama-3.1-8B-Instruct",
    "llama3-3B": "unsloth/Llama-3.2-3B",
    "llama3-3B-instruct": "unsloth/Llama-3.2-3B-Instruct",
    "llama3-1B": "unsloth/Llama-3.2-1B",
    "llama3-1B-instruct": "unsloth/Llama-3.2-1B-Instruct",
    "qwen2-32B": "unsloth/Qwen2.5-32B",
    "qwen2-32B-instruct": "unsloth/Qwen2.5-32B-Instruct",
    "qwen2-14B": "unsloth/Qwen2.5-14B",
    "qwen2-14B-instruct": "unsloth/Qwen2.5-14B-Instruct",
    "qwen2-7B": "unsloth/Qwen2.5-7B",
    "qwen2-7B-instruct": "unsloth/Qwen2.5-7B-Instruct",
    "qwen2-3B": "unsloth/Qwen2.5-3B",
    "qwen2-3B-instruct": "unsloth/Qwen2.5-3B-Instruct",
    "qwen2-2B": "unsloth/Qwen2.5-1.5B",
    "qwen2-2B-instruct": "unsloth/Qwen2.5-1.5B-Instruct",
    "qwen3-1.7B": "Qwen/Qwen3-1.7B-Base",
    "qwen3-4B": "Qwen/Qwen3-4B-Base",
    "qwen3-8B": "Qwen/Qwen3-8B-Base",
    "qwen3-14B": "Qwen/Qwen3-14B-Base",
    # Local repaired copy: upstream unsloth/gemma-3-1b-pt ships model.safetensors.index.json referencing shards that don't exist
    "gemma3-1B": "/project/6105902/shougan/persistent_cache/local_models/gemma-3-1b-pt",
    "gemma3-4B": "unsloth/gemma-3-4b-pt",
    "gemma3-12B": "unsloth/gemma-3-12b-pt",
}

DEFAULT_CHAT_TEMPLATE = "chatml"
_BASE_CHAT_TEMPLATE = None

MODEL_CHAT_TEMPLATE_MAP = {
    "llama3-8B": "llama-3.1",
    "llama3-8B-instruct": "llama-3.1",
    "llama3-3B": "llama-3.1",
    "llama3-3B-instruct": "llama-3.1",
    "llama3-1B": "llama-3.1",
    "llama3-1B-instruct": "llama-3.1",
    "qwen2-32B": "chatml",
    "qwen2-32B-instruct": "chatml",
    "qwen2-14B": "chatml",
    "qwen2-14B-instruct": "chatml",
    "qwen2-7B": "chatml",
    "qwen2-7B-instruct": "chatml",
    "qwen2-3B": "chatml",
    "qwen2-3B-instruct": "chatml",
    "qwen2-2B": "chatml",
    "qwen2-2B-instruct": "chatml",
    "qwen3-1.7B": "chatml",
    "qwen3-4B": "chatml",
    "qwen3-8B": "chatml",
    "qwen3-14B": "chatml",
    "gemma3-1B": "gemma-3",
    "gemma3-4B": "gemma-3",
    "gemma3-12B": "gemma-3",
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


def set_chat_template(model_name: str, simple: bool = False) -> str:
    global DEFAULT_CHAT_TEMPLATE, _BASE_CHAT_TEMPLATE
    resolved = resolve_chat_template(model_name)
    if simple:
        _BASE_CHAT_TEMPLATE = resolved
        DEFAULT_CHAT_TEMPLATE = "simple"
    else:
        _BASE_CHAT_TEMPLATE = None
        DEFAULT_CHAT_TEMPLATE = resolved
    return DEFAULT_CHAT_TEMPLATE


DEFAULT_SEED = 42
DEFAULT_EVAL_SEED = None  # When None, eval uses DEFAULT_SEED


def set_seed(seed: int):
    """Set the global training seed. Call once at pipeline start, like set_chat_template()."""
    global DEFAULT_SEED
    DEFAULT_SEED = seed


def set_eval_seed(seed: int):
    """Set the global eval seed (pass@k generation). Call once at pipeline start."""
    global DEFAULT_EVAL_SEED
    DEFAULT_EVAL_SEED = seed


def get_eval_seed() -> int:
    """Return the eval seed: DEFAULT_EVAL_SEED if set, else DEFAULT_SEED."""
    return DEFAULT_EVAL_SEED if DEFAULT_EVAL_SEED is not None else DEFAULT_SEED
