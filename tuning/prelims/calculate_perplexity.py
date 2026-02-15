import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from tuning.utils.utils import chat_template_func
from tuning.config import MODELS_DIR
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required = True)
args = parser.parse_args()
print(args)
model_name = args.model_name

dataset_path = "/project/6105902/shougan/balance-budget/tuning/data/datasets/sft-tuluif"
output_dir = f"/project/6105902/shougan/balance-budget/tuning/outputs/logp_evaluation/{model_name}"
output_dir = f"/project/6105902/shougan/balance-budget/tuning/outputs/perplexity_on_sft/{model_name}"

print(f"Evaluating {model_name}")
def compute_perplexity(log_probs):
    if len(log_probs) == 0:
        return float('inf')
    avg_neg_logp = -sum(log_probs) / len(log_probs)
    return np.exp(avg_neg_logp)


def get_ground_truth_logp(model, tokenizer, prompt_messages, response_message):
    prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False).to(model.device)

    
    messages_with_response = prompt_messages + [response_message]
    full_text = tokenizer.apply_chat_template(messages_with_response, tokenize=False, add_generation_prompt=False)
    full_input_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits
    
    response_start_index = prompt_ids.shape[1]
    relevant_logits = logits[0, response_start_index - 1:-1, :]
    response_ids = full_input_ids[0, response_start_index:]
    
    log_probs = F.log_softmax(relevant_logits, dim=-1)
    
    response_log_probs = []
    for i in range(len(response_ids)):
        token_id_i = response_ids[i].item()
        log_prob_i = log_probs[i, token_id_i].item()
        response_log_probs.append(log_prob_i)
    
    return {
        "sum_logp": sum(response_log_probs),
        "length": len(response_log_probs),
        "perplexity": compute_perplexity(response_log_probs),
        "prompt": prompt_str,
        "response": response_message["content"]
    }


def get_greedy_response(model, tokenizer, prompt_messages):
    prompt_str = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    with torch.no_grad():
        generated_output = model.generate(
            prompt_ids,
            max_new_tokens=4096,
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            # pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = generated_output.sequences[0, prompt_ids.shape[-1]:]
    
    response_log_probs = []
    for i, token_id in enumerate(generated_ids):
        logits_at_i = generated_output.scores[i][0]
        log_probs_at_i = F.log_softmax(logits_at_i, dim=-1)
        log_prob_i = log_probs_at_i[token_id].item()
        response_log_probs.append(log_prob_i)
    
    return {
        "sum_logp": sum(response_log_probs),
        "length": len(response_log_probs),
        "perplexity": compute_perplexity(response_log_probs),
        "prompt": prompt_str,
        "response": tokenizer.decode(generated_ids)
    }


print(f"Loading model: {model_name}")
hf_path = os.path.join(MODELS_DIR, model_name)
model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(hf_path)
tokenizer = chat_template_func(tokenizer)
if tokenizer.pad_token is None:
 tokenizer.pad_token = tokenizer.eos_token

full_dataset = load_from_disk(dataset_path)
train_dataset = full_dataset["test"] # .select(range(250))

os.makedirs(output_dir, exist_ok=True)
ground_truth_file = os.path.join(output_dir, f"ground-{model_name}.jsonl")
greedy_file = os.path.join(output_dir, f"greedy-{model_name}.jsonl")

with open(ground_truth_file, "w") as gt_f, open(greedy_file, "w") as greedy_f:
    for i in tqdm(range(len(train_dataset)), desc="Processing"):
        messages = train_dataset[i]["messages"]
        prompt_messages = [messages[0], messages[1]]  # system + user
        response_message = messages[2]  # assistant
        
        gt_result = get_ground_truth_logp(model, tokenizer, prompt_messages, response_message)
        gt_result["index"] = i
        gt_f.write(json.dumps(gt_result) + "\n")
        
        greedy_result = get_greedy_response(model, tokenizer, prompt_messages)
        greedy_result["index"] = i
        greedy_f.write(json.dumps(greedy_result) + "\n")
        
        if i % 3 == 0:
            gt_f.flush()
            greedy_f.flush()