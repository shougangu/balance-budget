from tuning.utils.utils import chat_template_func
from vllm import LLM, SamplingParams
from tuning.config import MODELS_DIR
from tuning.inference.config_inference import VLLMSamplingParamsConfig
import os

def load_vlm_model(model_name: str, n: int = None, temperature: float = None, max_logprobs = 20) -> LLM:
    model_path = f"{MODELS_DIR}/{model_name}"
    print(f"Loading model from {model_path}")

    gpu_util = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.80"))
    print(f"Using GPU memory utilization: {gpu_util}")

    llm = LLM(
        model=model_path,
        max_logprobs = max_logprobs,
        gpu_memory_utilization=gpu_util
    )
    
    config = VLLMSamplingParamsConfig()
    if n is not None:
        config.n = n
    if temperature is not None:
        config.temperature = temperature
    
    sampling_params = SamplingParams(
        **config.model_dump()
    )

    return llm, sampling_params

def make_vllm_call(llm: LLM, sampling_params: SamplingParams, prompts: list[str]) -> list[str]:
    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer)
    chat_template = tokenizer.chat_template

    outputs = llm.chat(prompts, sampling_params, chat_template=chat_template)
    if sampling_params.n == 1:
        responses = [output.outputs[0].text for output in outputs]
    else:
        responses = [[response.text for response in output.outputs] for output in outputs]

    print(f"Generated {len(responses)} responses using vllm")

    return responses

def tokenize_test_dataset(llm, messages):

    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer)
    tokenized_prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]

    return tokenized_prompts

def generate_responses_vllm(llm: LLM, sampling_params: SamplingParams, prompts: list[str], dataset):

    responses = make_vllm_call(llm, sampling_params, dataset)

    results = []
    for prompt, response_group in zip(prompts, responses):
        if sampling_params.n > 1 and isinstance(response_group, list):
            # This handles the case where n > 1 and make_vllm_call returns a list of responses for a prompt
            for single_response in response_group:
                results.append({"prompt": prompt, "response": single_response})
        else:
            # This handles the case where n == 1 and response_group is a single string
            results.append({"prompt": prompt, "response": response_group})
        
    return results



