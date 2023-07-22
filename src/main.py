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
    help="The input training data file (a text file).",
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
    default=16,
    help="model type",
)
parser.add_argument(
    "--eval_batch_size",
    type=int,
    default=32,
    help="model type",
)


MODEL_CLASSES = {
    "xlm-roberta-large": AutoModelForMaskedLM,
    "distilroberta-base": AutoModelForMaskedLM,
    "gpt2": AutoModelForCausalLM
}


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    auto_model = MODEL_CLASSES[args.model_name]
    model = auto_model.from_pretrained(args.model_name)

    tokenizer.pad_token = tokenizer.eos_token
    is_mlm = False if args.model_name == 'gpt2' else True
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15, mlm=is_mlm
    )

    train_dataset = AdditionDataset(file_path=args.train_data_file, tokenizer=tokenizer)
    eval_dataset = AdditionDataset(file_path=args.eval_data_file, tokenizer=tokenizer)
    train_type = args.train_data_file.split('/')[-1].split('.')[0]
    
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/{train_type}/{args.model_name}/results",
        logging_dir=f"{args.output_dir}/{train_type}/{args.model_name}/logs",
        num_train_epochs=5,
        learning_rate=2e-5,
        weight_decay=0.01,
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
