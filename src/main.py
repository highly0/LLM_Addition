import argparse
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    GPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
)
from data_prep import AdditionDataset


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_file",
    default="/workspace/LLM_Addition/data/train/train_add_decomp_8_digits.txt",
    type=str,
    help="The input training data file (a text file).",
)
parser.add_argument(
    "--eval_data_file",
    default="/workspace/LLM_Addition/data/test/test_add_8_digits.txt",
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
    default="gpt2",
    help="model type",
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=64,
    help="training batch size",
)

parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=64,
    help="eval batch size",
)

parser.add_argument(
    "--num_train_epochs",
    type=int,
    default=15,
    help="number of epoches",
)


MODEL_CLASSES = {
    "gpt2": (GPT2Tokenizer, GPT2Config, GPT2LMHeadModel),
}


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_tokenizer, model_config, head_model = MODEL_CLASSES[args.model_name]
    tokenizer = pretrained_tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = model_config.from_pretrained(args.model_name)
    model = head_model(config)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        mlm=False,
    )

    train_dataset = AdditionDataset(file_path=args.train_data_file, tokenizer=tokenizer)
    eval_dataset = AdditionDataset(file_path=args.eval_data_file, tokenizer=tokenizer)
    train_type = args.train_data_file.split("/")[-1].split(".")[0]

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{train_type}/{args.model_name}/results",
        logging_dir=f"{args.output_dir}/{train_type}/{args.model_name}/logs",
        num_train_epochs=args.num_train_epochs,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.eval_batch_size,
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        evaluation_strategy="steps",
        logging_steps=500,  # log & save weights each logging_steps
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(
        f"{args.output_dir}/{train_type}/{args.model_name}/best_checkpoint"
    )
