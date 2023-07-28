import re
import torch
from transformers import (
    AutoTokenizer,
)


class AdditionDatasetSeq2Seq(torch.utils.data.Dataset):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self, file_path, tokenizer: AutoTokenizer, max_number_len=10):
        self.questions, self.answers = [], []
        with open(file_path, encoding="utf-8") as file:
            for line in file:
                numbers = re.findall(r"\d+", line)
                n1, n2, res = numbers
                n1_filler = "0" * (max_number_len - len(n1))
                n2_filler = "0" * (max_number_len - len(n2))
                res_filler = "0" * (max_number_len - len(res) + 1)

                self.questions.append(f"{n1_filler}{n1}{n2_filler}{n2}")
                self.answers.append(f"{res_filler}{res}")

        self.questions_encodings = tokenizer(
            self.questions, truncation=True, padding=True
        )
        self.answers_encodings = tokenizer(self.answers, truncation=True, padding=True)

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            "input_ids": self.questions_encodings["input_ids"][idx],
            "labels": self.answers_encodings["input_ids"][idx],
        }
