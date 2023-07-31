import torch
from transformers import PreTrainedTokenizer


class AdditionDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512):
        with open(file_path, encoding="utf-8") as file:
            texts = [line.rstrip() for line in file]

        self.examples = tokenizer.batch_encode_plus(
            texts, add_special_tokens=True, max_length=block_size
        )["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)
