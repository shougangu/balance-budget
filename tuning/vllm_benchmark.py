from tuning.utils.utils import chat_template_func
from vllm import LLM, SamplingParams
from tuning.config import MODELS_DIR
from tuning.inference.config_inference import VLLMSamplingParamsConfig
from tuning.data.test_dataset import get_ifeval_test_dataset
import os
import time


def build_prompts(tokenizer, messages):
    return [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]


def extract_texts(outputs):
    texts = []
    for output in outputs:
        if output.outputs:
            texts.append(output.outputs[0].text)
        else:
            texts.append("")
    return texts


def time_call(label, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label} took {elapsed:.3f}s")
    return result, elapsed


def average_length(texts):
    if not texts:
        return 0.0
    total = sum(len(text) for text in texts)
    return total / len(texts)


def average_token_length(outputs):
    """Calculate the average token length of model outputs."""
    if not outputs:
        return 0.0
    
    token_counts = []
    for output in outputs:
        if output.outputs and len(output.outputs) > 0:
            token_count = len(output.outputs[0].token_ids)
            token_counts.append(token_count)
    
    if not token_counts:
        return 0.0
    
    return sum(token_counts) / len(token_counts)


if __name__ == '__main__':    
    model_name = "llama3-8B_llama3-8B_ppl-5.00_sft-320_pt-tuluif-3776"
    model_path = os.path.join(MODELS_DIR, model_name)
    llm = LLM(model=model_path)
    config = VLLMSamplingParamsConfig()
    config.n = 8
    # config.temperature = 0
    sampling_params = SamplingParams(**config.model_dump())
    sampling_params.min_tokens = 2048
    sampling_params.max_tokens = 2048
    sampling_params.stop = []           # remove stop strings
    sampling_params.stop_token_ids = [] # remove stop token IDs
    sampling_params.ignore_eos = True   # ignore the EOS token entirely
    print(sampling_params)


    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer)
    chat_template = tokenizer.chat_template


    dataset = get_ifeval_test_dataset()
    messages = dataset["messages"]
    prompts = build_prompts(tokenizer, messages)

    # print("\n", "SELF MADE PROMPTS: ")
    # for i in range(len(messages)):
    #     print(f"Original messages[{i}]: {messages[i]}")
    #     print(f"Prompt[{i}]: {prompts[i]}")
    #     print(f"Tokenized prompt[{i}]: {tokenizer(prompts[i])}")
    #     print("-" * 50)

    generate_outputs, _ = time_call(
        "llm.generate",
        lambda: llm.generate(prompts, sampling_params),
    )
    generate_texts = extract_texts(generate_outputs)
    print(f"Avg generate_texts length: {average_length(generate_texts):.2f}")
    print(f"Avg chat_texts length (tokens): {average_token_length(generate_outputs):.2f}")
    print("First prompt input (llm.generate prompt string):")
    print(prompts[0])
    print("First prompt response (llm.generate):")
    print(generate_texts[0])


    # chat_outputs, _ = time_call(
    #     "llm.chat",
    #     lambda: llm.chat(messages, sampling_params, chat_template=chat_template),
    # )
    
    # chat_texts = extract_texts(chat_outputs)
    # print(f"Avg chat_texts length (chars): {average_length(chat_texts):.2f}")
    # print(f"Avg chat_texts length (tokens): {average_token_length(chat_outputs):.2f}")
    # print("First prompt input (llm.chat messages):")
    # print(messages[0])
    # print("First prompt response (llm.chat):")
    # print(chat_texts[0])