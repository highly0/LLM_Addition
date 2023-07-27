import argparse
import torch
import math
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    GPT2LMHeadModel,
    AutoConfig,
    XLMRobertaForMaskedLM, 
    RobertaForMaskedLM,
    XLMRobertaConfig,
    RobertaConfig,
    GPT2Config,
)
from dataprep import AdditionDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_file",
    default="/workspace/LLM_Addition/data/train/train_add_spaced.txt",
    type=str,
    help="The input training data file (a text file).",
)
parser.add_argument(
    "--eval_data_file",
    default="/workspace/LLM_Addition/data/eval/test_2_add.txt",
    type=str,
    help="The input evaluation data file (a text file).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/workspace/storage/llm_addition/",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="roberta-large",
    help="model type",
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=16,
    help="training batch size",
)
parser.add_argument(
    "--train_type",
    type=str,
    default='scratch',
    choices=['finetuned', 'scratch'],
    help="training batch size",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=32,
    help="eval batch size",
)


MODEL_CLASSES = {
    "xlm-roberta-large": (AutoModelForMaskedLM, XLMRobertaConfig, XLMRobertaForMaskedLM),
    "roberta-large": (AutoModelForMaskedLM, RobertaConfig, RobertaForMaskedLM),
    "gpt2": (AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel), 
}


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, return_special_tokens_mask=True)
    if args.train_type == "scratch": # training model from scratch
        _, model_config, head_model = MODEL_CLASSES[args.model_name]
        config = model_config.from_pretrained(
            args.model_name,
            vocab_size=len(tokenizer),
            n_ctx=128,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = head_model(config)
    else: # fine-tuning
        auto_model, _, _ = MODEL_CLASSES[args.model_name]
        model = auto_model.from_pretrained(args.model_name)

    tokenizer.pad_token = tokenizer.eos_token
    IS_MLM = False if args.model_name == "gpt2" else True
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, mlm=IS_MLM
    )

    train_dataset = AdditionDataset(file_path=args.train_data_file, tokenizer=tokenizer)
    eval_dataset = AdditionDataset(file_path=args.eval_data_file, tokenizer=tokenizer)
    train_type = args.train_data_file.split("/")[-1].split(".")[0]

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{train_type}/{args.model_name}_{args.train_type}/results",
        logging_dir=f"{args.output_dir}/{train_type}/{args.model_name}_{args.train_type}/logs",
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        evaluation_strategy="steps",
        logging_steps=500,  # log & save weights each logging_steps
        save_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    with open(
        f"{args.output_dir}/{train_type}/{args.model_name}_{args.train_type}/evaluation.txt",
        "w+",
        encoding="utf-8",
    ) as file:
        file.write("trainer.evaluate() results:\n")
        for k, v in eval_results.items():
            file.write(f"{k}: {v}\n")
        file.write(f"Final eval perplexity: {math.exp(eval_results['eval_loss']):.2f}\n")
