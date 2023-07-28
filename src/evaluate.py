import re
import numpy as np
from tqdm import tqdm
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preds_labels(model_inputs, model, tokenizer, bs=16):
    """given list of model_inputs,"""

    # break model_inputs into chunks
    decoded_preds = []
    for chunk in tqdm(chunks(model_inputs, bs)):
        input_ids = tokenizer(
            chunk, return_tensors="pt", truncation=True, padding=True
        )["input_ids"].to(device)
        preds = model.generate(input_ids)
        batch_decoded_preds = tokenizer.batch_decode(preds)
        decoded_preds.extend(batch_decoded_preds)

    # clean outputs to int
    preds_numbers = []
    for pred in decoded_preds:
        preds_numbers.append(int(re.findall(r"\d+", pred)[0]))

    return preds_numbers


def prepare_model_inputs_labels(numbers_list, max_len=20):
    """
    given list of numbers (n1, n2, res), prepare for
    model intput
    """
    res_inputs, res_labels = [], []
    for numbers in numbers_list:
        n1, n2, res = numbers
        max_len = 20
        n1_filler = "0" * (max_len - len(n1))
        n2_filler = "0" * (max_len - len(n2))

        model_inputs = f"{n1_filler}{n1}{n2_filler}{n2}"
        res_inputs.append(model_inputs)
        res_labels.append(int(res))

    return res_inputs, res_labels


def chunks(lst, size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def calculate_loss(model, tokenizer, file_path):
    """given path of test file, return mse and rmse loss"""
    with open(file_path, encoding="utf-8") as file:
        numbers_list = []
        for line in file:
            numbers = re.findall(r"\d+", line)
            numbers_list.append(numbers)

    model_inputs, labels = prepare_model_inputs_labels(numbers_list)
    preds = get_preds_labels(model_inputs, model, tokenizer)

    # mse between pred number and label
    preds = np.array(preds)
    labels = np.array(labels)
    mse = ((preds - labels) ** 2).mean()
    rmse = mse**0.5
    return mse, rmse


if __name__ == "__main__":
    test_path = "/workspace/LLM_Addition/data/eval/test_1.txt"
    path = (
        "/workspace/storage/llm_addition/train/t5-large_scratch/results/best_checkpoint"
    )
    model = T5ForConditionalGeneration.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained("t5-large")

    res_mse, res_rmse = calculate_loss(model, tokenizer, test_path)
    print(res_mse, res_rmse)
