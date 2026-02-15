import wandb
import argparse
from tuning.training.config_training import ModelLoadConfig, LoraConfig, PTRunConfig, DPOTrainingConfig, DatasetConfig, PassAtKConfig, PerplexityConfig, SFTRunConfig
from tuning.config import HF_MODEL_MAP
from tuning.run_inference import run_inference
from tuning.run_evaluation import run_evaluation
from tuning.training.dpo_training import train_model_dpo
from tuning.training.kto_training import train_model_kto

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SFT training")

    parser.add_argument("--model", type=str, help="llama3-8B", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset", required=True)
    parser.add_argument("--train_size", type=int, help="Train size", required=True)
    parser.add_argument("--task", type=str, help="Task name", required=True)
    parser.add_argument("--pft_method", type=str, help="PFT method", required=True)
    parser.add_argument("--dynamic_path", type=str, help="Dynamic dataset path", required=False)
    parser.add_argument("--sft_train_size", type=int, help="SFT run train size", required=False) # for standard sft->dpo
    parser.add_argument("--do_training", action="store_true", help="Do training")
    parser.add_argument("--do_inference", action="store_true", help="Do inference")
    parser.add_argument("--do_evaluation", action="store_true", help="Do evaluation")

    args = parser.parse_args()

    print(args.__dict__)
    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = DPOTrainingConfig()
    training_args.eval_steps = 40
    dataset_config = DatasetConfig(
        dataset = args.dataset,
        dataset_type = "pt",
        train_size = args.train_size,
    )

    print(dataset_config)

    if args.sft_train_size:
        # the sft_run_config.run_name = DatasetConfig.full_name
        sft_run_config = SFTRunConfig(
            dataset_config = DatasetConfig(
                dataset = args.dataset,
                dataset_type = "sft",
                train_size = args.sft_train_size,
                dynamic_path = args.dynamic_path if args.dynamic_path != "" else None,
            ),
            model_name_hf = HF_MODEL_MAP[args.model],
            model_name = args.model,
            task_name=args.task
        )

    run_config = PTRunConfig(
        dataset_config = dataset_config,
        # model_name_hf = HF_MODEL_MAP[args.model],
        model_name = args.model,
        sft_run_config = sft_run_config if args.sft_train_size else None,
        task_name=args.task,
        pft_method=args.pft_method,
        do_training=args.do_training,
        do_inference=args.do_inference,
        do_evaluation=args.do_evaluation,
    )

    print(run_config)

    passk_config = PassAtKConfig(
        target_pass_at_k=[1.2],
        k_values=[1],
        n_samples=1,
        num_prompts=100,
        temperature=0.7,
        strict=True,
        enabled=True,
    )
    if run_config.do_training:
        run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

        with run:
            if run_config.pft_method == "dpo":
                train_model_dpo(
                    run_config = run_config,
                    lora_config = lora_config,
                    model_load_config = model_load_config,
                    training_args = training_args,
                    passk_config = passk_config,
                    perplexity_config= PerplexityConfig(perplexity_thresholds=[0.1])
                )
            elif run_config.pft_method == "kto":
                train_model_kto(
                    run_config = run_config,
                    lora_config = lora_config,
                    model_load_config = model_load_config,
                    training_args = training_args,
                )
            
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