import wandb
from tuning.training.config_training import ModelLoadConfig, LoraConfig, PassAtKConfig, SFTRunConfig, TrainingArgumentsConfig, DatasetConfig
from tuning.config import HF_MODEL_MAP
from tuning.training.sft_training import train_model_sft
from tuning.run_inference import run_inference
from tuning.run_evaluation import run_evaluation
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training")

    parser.add_argument("--model", type=str, help="llama3-8B", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset", required=True)
    parser.add_argument("--train_size", type=int, help="Train size", required=True)
    parser.add_argument("--task", type=str, help="Task name", required=True)
    parser.add_argument("--do_training", action="store_true", help="Do training")
    parser.add_argument("--do_inference", action="store_true", help="Do inference")
    parser.add_argument("--do_evaluation", action="store_true", help="Do evaluation")

    args = parser.parse_args()

    print(args.__dict__)

    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = TrainingArgumentsConfig()

    train_size = args.train_size 
    dataset_config = DatasetConfig(
        dataset = args.dataset,
        dataset_type = "sft",
        train_size = args.train_size,
    ) 

    print(dataset_config)

    run_config = SFTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[args.model],
        model_name = args.model,
        task_name=args.task,
        do_training=args.do_training,
        do_inference=args.do_inference,
        do_evaluation=args.do_evaluation,
    )

    print(run_config)

    if run_config.do_training:
        run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

        with run:
            passk_config = PassAtKConfig(
                target_pass_at_k=[1.2],
                k_values=[1],
                n_samples=1,
                num_prompts=100,
                temperature=0.7,
                strict=True,
                enabled=True,
            )
            model, tokenizer, trainer = train_model_sft(
                run_config = run_config,
                lora_config = lora_config,
                model_load_config = model_load_config,
                training_args = training_args,
                perplexity_thresholds = [3.3,3.2,3.1,3.0]
            )
            import gc, torch, subprocess
            del model
            del tokenizer
            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            result = subprocess.run( 
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            print("GPU Utilization [% GPU, MB used, MB total]:", result.stdout.strip())
            import time, torch
            torch.cuda.empty_cache()
            time.sleep(30)


    if run_config.do_inference:
        run_inference(run_config)
        torch.cuda.empty_cache()
        time.sleep(30)

    if run_config.do_evaluation:
        run_evaluation(run_config)
        torch.cuda.empty_cache()
        time.sleep(30)


    print(f"********" * 20) 