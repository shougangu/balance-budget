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

    model_name = "qwen2-7B"
    print(f"Running ifeval for {model_name}")
    ifeval_evaluate(model_name)

    