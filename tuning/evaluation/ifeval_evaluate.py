from tuning.config import IFEVAL_OUTPUTS_DIR
import subprocess


def ifeval_evaluate(model_name: str) -> str:
    print(f"Running IFeval evaluation for {model_name}")

    output_path = f"{IFEVAL_OUTPUTS_DIR}/{model_name}/responses.jsonl"

    result = subprocess.run([
        'python3', '-m', 'instruction_following_eval.evaluation_main',
        '--input_data=./instruction_following_eval/data/input_data.jsonl',
        f'--input_response_data={output_path}',
        f'--output_dir={IFEVAL_OUTPUTS_DIR}/{model_name}/'
    ], capture_output=True, text=True, check=True)

    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if combined:
        print(combined, end="" if combined.endswith("\n") else "\n")

    return combined


if __name__ == "__main__":    
    model_name = "qwen2-7B"
    print(f"Running ifeval for {model_name}")
    ifeval_evaluate(model_name)

    