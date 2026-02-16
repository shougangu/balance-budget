from tuning.data.test_dataset import get_ifeval_test_dataset
from tuning.config import IFEVAL_OUTPUTS_DIR, RESPONSES_FILENAME, resolve_chat_template
from tuning.inference.vllm_utils import generate_responses_vllm, load_vlm_model
from tuning.utils.gpt_utils import save_responses
from typing import List, Dict, Optional
import gc
from vllm.distributed.parallel_state import destroy_model_parallel         
import torch

def run_inference_ifeval(model_name: str, n_samples: int = 1, temperature: float = 0.5,
                         chat_template: str = None,
                         save_results: bool = True, num_examples: Optional[int] = None) -> List[Dict]:
    """Run IFEval inference with configurable sampling.
    
    Args:
        model_name: Name of the model to use
        n_samples: Number of samples per prompt (for pass@k evaluation)
        temperature: Sampling temperature
        chat_template: Optional chat template override used to format messages for generation
        save_results: Whether to save responses to disk
        num_examples: Number of examples to run (None for all)
    
    Returns:
        List of dicts with 'prompt' and 'responses' keys
    """
    
    test_dataset = get_ifeval_test_dataset()
    if num_examples is not None:
        test_dataset = test_dataset.select(range(num_examples))
    resolved_template = resolve_chat_template(model_name, chat_template)

    llm, sampling_params = load_vlm_model(model_name, n=n_samples, temperature=temperature)
    responses = generate_responses_vllm(
        llm=llm,
        sampling_params=sampling_params,
        prompts=test_dataset["prompt"],
        dataset=test_dataset["messages"],
        chat_template=resolved_template,
    )
    
     # ADD CLEANUP                                                               
     
    destroy_model_parallel()                                                    
    del llm                                                                     
    gc.collect()                                                                
    torch.cuda.empty_cache()                                                    
    if torch.cuda.is_available():                                               
        torch.cuda.synchronize()                                                

    # Format results for pass@k evaluation
    # [{prompt: "", response: ""}, 
    #  {prompt: "", response:""}, ...]
    results = []
    for prompt, resp in zip(test_dataset["prompt"], responses):
        if isinstance(resp, list):
            for r in resp:      
                results.append({"prompt": prompt, "response": r})  
        else:
            results.append({"prompt": prompt, "response": resp})  
    
    if save_results:
        save_path = f"{IFEVAL_OUTPUTS_DIR}/{model_name}/"
        save_responses(save_path, RESPONSES_FILENAME, responses)
    
    return responses

if __name__ == "__main__":

    model_name =  "qwen2-7B"
    print(f"Running ifeval for {model_name}")
    run_inference_ifeval(model_name)
    
