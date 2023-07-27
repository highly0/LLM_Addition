import torch
from transformers import (
    AutoTokenizer,
)


class AdditionDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer: AutoTokenizer):
        with open(file_path, encoding="utf-8") as file:
            self.texts = [line.rstrip() for line in file]

        self.encodings = tokenizer(self.texts, truncation=True, padding=True)

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        item_dict = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item_dict
