from tuning.inference.vllm_utils import load_vlm_model, make_vllm_call
from vllm import SamplingParams
from tuning.utils.utils import chat_template_func
from tuning.config import IFEVAL_OUTPUTS_DIR, RESPONSES_FILENAME
from tuning.utils.gpt_utils import save_responses
from datasets import load_from_disk, DatasetDict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


def get_response_logp_transformers(model, tokenizer, prompt_messages, response_text):
    prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str)

    messages_with_response = prompt_messages + [{"role": "assistant", "content": response_text}]
    full_text = tokenizer.apply_chat_template(messages_with_response, tokenize=False, add_generation_prompt=False)
    full_input_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)


    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits
    
    response_start_index = len(prompt_ids)
    relevant_logits = logits[0, response_start_index - 1:-1, :]

    response_ids = full_input_ids[0, response_start_index:]

    # 3. Calculate log probabilities from logits
    # Apply log_softmax to convert logits to log probabilities
    log_probs = F.log_softmax(relevant_logits, dim=-1)

    # 4. Gather the log probabilities for the specific target tokens
    target_log_probs = torch.gather(log_probs, 1, response_ids.unsqueeze(-1)).squeeze(-1)

    # 5. Sum to get the total log probability of the response
    total_logp = target_log_probs.sum().item()

    return response_text, total_logp


def get_response_logp(llm, prompt_messages, response_text=None):
    """
    Calculates the total log probability of a response for a given prompt.
    If response_text is None, it generates a greedy response.
    If response_text is provided, it calculates the logP for that specific text.
    """
    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer)
    
    if response_text:
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_str)

        messages_with_response = prompt_messages + [{"role": "assistant", "content": response_text}]
        full_text = tokenizer.apply_chat_template(messages_with_response, tokenize=False, add_generation_prompt=False)
        full_input_ids = tokenizer.encode(full_text)

        sampling_params = SamplingParams(
            prompt_logprobs=5,   
            max_tokens=1,        
            temperature=0.0,
            stop=[],
        )

        outputs = llm.generate([full_text], sampling_params)
        output = outputs[0]

        prompt_logprobs = output.prompt_logprobs
        

        response_log_probabilities = []
        for i in range(len(prompt_ids), len(full_input_ids)):
            target_token_id = full_input_ids[i]
            logprob_dict = prompt_logprobs[i]
            
            if target_token_id in logprob_dict:
                response_log_probabilities.append(logprob_dict[target_token_id].logprob)
            else:
                response_log_probabilities.append(-100.0) # Assign a very low logprob

        print(response_log_probabilities)
        total_logp = sum(response_log_probabilities)
        return response_text, total_logp


    else:
        sampling_params = SamplingParams(
            prompt_logprobs = 1,
            temperature=0,
            logprobs=1,
            max_tokens=4096,
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(prompt_str, sampling_params)

        output = outputs[0].outputs[0]
        if not output.logprobs:
            return output.text, None
            
        total_logp = np.sum([logprob[list(logprob.keys())[0]].logprob for logprob in output.logprobs])
        return output.text, total_logp

if __name__ == "__main__":
    model_name = "llama3-8B"
    model_name = "llama3-8B_sft-tuluif-1000"
    print(f"Loading model: {model_name}")
    llm, _ = load_vlm_model(model_name, max_logprobs = 20)

    print(f"Loading model: {model_name} from hf")
    hf_path = f"/project/6105902/shougan/balance-budget/tuning/models/{model_name}"
    hf_model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch.bfloat16, device_map="auto")
    hf_tokenizer = AutoTokenizer.from_pretrained(hf_path)
    hf_tokenizer = chat_template_func(hf_tokenizer)


    print("\nLoading Tulu-IF dataset...")
    dataset_path = "/project/6105902/shougan/balance-budget/tuning/data/datasets/sft-tuluif"
    full_dataset = load_from_disk(dataset_path)
    data_point = full_dataset["train"][1]

    print("--------------------- THIS IS THE DATA --------------------------")
    print(data_point)
    print("-----------------------------------------------------------------")

    # The prompt is the user's message
    prompt_message = [data_point['messages'][0], data_point['messages'][1]] #index 0 and 1 is system and user question
    ground_truth_answer = data_point['messages'][2]['content'] # index 2 is the ground truth answer

    # _, ground_truth_logp = get_response_logp(llm, prompt_message, response_text=ground_truth_answer)
    _, ground_truth_logp_hf = get_response_logp_transformers(hf_model, hf_tokenizer, prompt_message, response_text=ground_truth_answer)

    print(f"Total Log Probability (Ground-Truth): {ground_truth_logp_hf:.4f}")
    print("-" * 20)


    greedy_answer, greedy_logp = get_response_logp(llm, prompt_message)
    
    print(f"\nGreedy Answer:\n{greedy_answer}")
    print(f"Total Log Probability (Greedy): {greedy_logp:.4f}")
    print("-" * 20)



  