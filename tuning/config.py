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
    "llama3-1B": "unsloth/Llama-3.2-1B",
    "gemma3-1B": "unsloth/gemma-3-1b-pt",
    "gemma3-4B": "unsloth/gemma-3-4b-pt",
    "gemma3-12B": "unsloth/gemma-3-12b-pt",
}

DEFAULT_CHAT_TEMPLATE = "chatml"
_BASE_CHAT_TEMPLATE = None

MODEL_CHAT_TEMPLATE_MAP = {
    "llama3-8B": "llama-3.1",
    "llama3-3B": "llama-3.1",
    "llama3-1B": "llama-3.1",
    "qwen2-14B": "chatml",
    "qwen2-7B": "chatml",
    "qwen2-3B": "chatml",
    "qwen2-2B": "chatml",
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
