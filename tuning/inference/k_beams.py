from tuning.inference.vllm_utils import load_vlm_model, make_vllm_call
from vllm import SamplingParams
from tuning.utils.utils import chat_template_func
from tuning.config import IFEVAL_OUTPUTS_DIR, RESPONSES_FILENAME
from tuning.utils.gpt_utils import save_responses
from datasets import load_from_disk, DatasetDict
import numpy as np
import torch
def get_response_logp(llm, prompt_messages, response_text=None):
    """
    Calculates the total log probability of a response for a given prompt.
    If response_text is None, it generates a greedy response.
    If response_text is provided, it calculates the logP for that specific text.
    """
    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer) # add chatML
    
    if response_text:
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt_str)

        messages_with_response = prompt_messages + [{"role": "assistant", "content": response_text}]
        full_text = tokenizer.apply_chat_template(messages_with_response, tokenize=False, add_generation_prompt=False)
        full_input_ids = tokenizer.encode(full_text)

        sampling_params = SamplingParams(
            prompt_logprobs=100000,   
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
                print(f"CANNOT FIND THE TOKEN {target_token_id}")
                response_log_probabilities.append(-100.0) 

        print(response_log_probabilities)
        total_logp = sum(response_log_probabilities)
        return response_text, total_logp


    else:
        sampling_params = SamplingParams(
            prompt_logprobs = 1,
            temperature=0,
            logprobs=1,
            max_tokens=4096,
        )
        prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate(prompt_str, sampling_params)

        output = outputs[0].outputs[0]
        if not output.logprobs:
            return output.text, None
            
        total_logp = np.sum([logprob[list(logprob.keys())[0]].logprob for logprob in output.logprobs])
        return output.text, total_logp

if __name__ == "__main__":
    model_name = "llama3-8B_sft-tuluif-5000_pt-tuluif-5000"
    print(f"Loading model: {model_name}")

    llm, _ = load_vlm_model(model_name, max_logprobs = 100000)

    dataset_path = "/project/6105902/shougan/balance-budget/tuning/data/datasets/sft-tuluif"
    full_dataset = load_from_disk(dataset_path)
    data_point = full_dataset["train"][1]

    print(f"\n\n Using {data_point}")

    prompt_message = [data_point['messages'][0], data_point['messages'][1]] #index 0 and 1 is system and user question

    ground_truth_answer = data_point['messages'][2]['content'] #index 2 is the ground truth answer

    _, ground_truth_logp = get_response_logp(llm, prompt_message, response_text=ground_truth_answer)
    print(f"\nGround-Truth Answer:\n{ground_truth_answer}")
    print(f"Total Log Probability (Ground-Truth): {ground_truth_logp:.4f}")
    print("-" * 20)


    greedy_answer, greedy_logp = get_response_logp(llm, prompt_message)
    
    print(f"\nGreedy Answer:\n{greedy_answer}")
    print(f"Total Log Probability (Greedy): {greedy_logp:.4f}")
    print("-" * 20)



  