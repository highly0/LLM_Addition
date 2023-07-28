import argparse
import torch
import torch.nn as nn
import math
import evaluate
import re
from transformers import (
    AutoTokenizer,
    T5Tokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModel,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    GPT2LMHeadModel,
    GPT2Config,
    T5Config,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
)
from dataprep import AdditionDatasetSeq2Seq


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_file",
    default="/workspace/LLM_Addition/data/train/train.txt",
    type=str,
    help="The input training data file (a text file).",
)
parser.add_argument(
    "--eval_data_file",
    default="/workspace/LLM_Addition/data/eval/test.txt",
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
    default="t5-large",
    help="model type",
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=64,
    help="training batch size",
)
parser.add_argument(
    "--train_type",
    type=str,
    default="scratch",
    choices=["finetuned", "scratch"],
    help="training batch size",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=64,
    help="eval batch size",
)


MODEL_CLASSES = {
    "gpt2": (AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel),
    "t5-large": (AutoModel, T5Config, T5ForConditionalGeneration),
}





def save_result(path, perplexity, eval_results):
    """ save trainer.evaluate to path"""
    print(f"Perplexity: {perplexity}")
    with open(
        path,
        "w+",
        encoding="utf-8",
    ) as file:
        file.write("trainer.evaluate() results:\n")
        for k, v in eval_results.items():
            file.write(f"{k}: {v}\n")
        file.write(f"Final eval perplexity: {perplexity}\n")

if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name  # , return_special_tokens_mask=True
    )
    if args.train_type == "scratch":  # training model from scratch
        _, model_config, head_model = MODEL_CLASSES[args.model_name]
        config = model_config.from_pretrained(
            args.model_name,
            vocab_size=len(tokenizer),
            n_ctx=128,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = head_model(config)
    else:  # fine-tuning
        auto_model, _, _ = MODEL_CLASSES[args.model_name]
        model = auto_model.from_pretrained(args.model_name)

    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, return_tensors="pt"
    )

    train_dataset = AdditionDatasetSeq2Seq(
        file_path=args.train_data_file, tokenizer=tokenizer, max_number_len=20
    )
    eval_dataset = AdditionDatasetSeq2Seq(
        file_path=args.eval_data_file, tokenizer=tokenizer, max_number_len=20
    )

    train_type = args.train_data_file.split("/")[-1].split(".")[0]
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{args.output_dir}/{train_type}/{args.model_name}_{args.train_type}/results",
        logging_dir=f"{args.output_dir}/{train_type}/{args.model_name}_{args.train_type}/logs",
        num_train_epochs=10,
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
        predict_with_generate=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    trainer.save_model(
        f"{args.output_dir}/{train_type}/{args.model_name}_{args.train_type}/best_checkpoint"
    )

    perplexity = (
        None
        if args.model_name.split("-")[0] == "t5"
        else math.exp(eval_results["eval_loss"])
    )
    save_path = f"{args.output_dir}/{train_type}/{args.model_name}_{args.train_type}/evaluation.txt"
    save_result(save_path, perplexity, eval_results)
