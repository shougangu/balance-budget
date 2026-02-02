from tuning.config import IFEVAL_OUTPUTS_DIR, RESPONSES_FILENAME
import subprocess


def ifeval_evaluate(model_name: str):
    print(f"Running IFeval evaluation for {model_name}")

    output_path = f"{IFEVAL_OUTPUTS_DIR}/{model_name}/{RESPONSES_FILENAME}"
    output_path =  f"{IFEVAL_OUTPUTS_DIR}/{model_name}/responses.jsonl"


    subprocess.run([
        'python3', '-m', 'instruction_following_eval.evaluation_main',
        '--input_data=./instruction_following_eval/data/input_data.jsonl',
        f'--input_response_data={output_path}',
        f'--output_dir={IFEVAL_OUTPUTS_DIR}/{model_name}/'
    ], check=True)


if __name__ == "__main__":
    # model_name = "llama3-8B_llama3-8B_ppl-5.00_sft-320_pt-tuluif-3776"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)
    
    # model_name = "llama3-8B_llama3-8B_ppl-4.00_sft-480_pt-tuluif-3616"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)

    # model_name = "llama3-8B_llama3-8B_ppl-3.75_sft-640_pt-tuluif-3456"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)

    model_name = "llama3-8B_llama3-8B_ppl-3.50_sft-800_pt-tuluif-3296"
    print(f"Running ifeval for {model_name}")
    ifeval_evaluate(model_name)

    # model_name = "llama3-8B_llama3-8B_ppl-3.25_sft-1120_pt-tuluif-2976"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)

    # model_name = "llama3-8B_llama3-8B_ppl-3.00_sft-2880_pt-tuluif-1216"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)

    # model_name = "llama3-8B_llama3-8B_pass@1-0.20_sft-400_pt-tuluif-1648"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)

    # model_name = "llama3-8B_llama3-8B_pass@1-0.40_sft-800_pt-tuluif-1248"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)

    # model_name = "llama3-8B_llama3-8B_pass@1-0.50_sft-1200_pt-tuluif-848"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)
    
    # model_name = "llama3-8B_llama3-8B_pass@1-0.60_sft-2000_pt-tuluif-48"
    # print(f"Running ifeval for {model_name}")
    # ifeval_evaluate(model_name)
    