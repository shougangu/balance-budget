from tuning.inference.ifeval_inference import run_inference_ifeval
from tuning.training.config_training import SFTRunConfig, PTRunConfig
from tuning.config import resolve_chat_template
from typing import Union

def run_inference(run_config: Union[SFTRunConfig, PTRunConfig], chat_template: str = None):

    task_name = run_config.task_name
    model_name = run_config.run_name
    resolved_template = resolve_chat_template(
        run_config.model_name,
        chat_template if chat_template is not None else run_config.chat_template,
    )

    if task_name == "instruction":
        run_inference_ifeval(model_name, chat_template=resolved_template)
    elif task_name == "math":
        print(f"GSM8k inference is run during evaluation")
    else:
        raise ValueError(f"Task {task_name} not supported")
