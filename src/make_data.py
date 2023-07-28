""" make train and eval data"""
import argparse
from random import randint


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_result_path",
    default="/workspace/LLM_Addition/data/train/train_1.txt",
    type=str,
    help="The output training data file (a text file).",
)

parser.add_argument(
    "--eval_data_result_path",
    default="/workspace/LLM_Addition/data/eval/test_1.txt",
    type=str,
    help="The ouput eval data file (a text file).",
)

parser.add_argument(
    "--max_number_length",
    default=20,
    type=int,
    help="The input training data file (a text file).",
)


parser.add_argument(
    "--train_n_digit_batch_size",
    default=100,
    type=int,
    help="for each n length digit addition, this is the batch size for training",
)

parser.add_argument(
    "--eval_n_digit_batch_size",
    default=100,
    type=int,
    help="for each n length digit addition, this is the batch size for eval",
)


def random_n_number(length: int) -> int:
    """return a random number of length"""
    range_start = 10 ** (length - 1)
    range_end = (10**length) - 1
    return randint(range_start, range_end)


def make_data(size: int, digit_size: int) -> [int]:
    """ 
    for the desired digit length size, make the data
    and return a list of data
    
    for example: digit_size: 5, size: 10
    returned data: list of 10 one digit addition
                           10 two digit addtion
                            etc... -> 10 five digit addition 
    """
    data = []
    for curr_digits in range(1, digit_size + 1):
        for _ in range(size):
            n1 = random_n_number(curr_digits)
            n2 = random_n_number(curr_digits)
            res = n1 + n2
            data.append(f"{n1} plus {n2} equal {res}")
    return data


def save_data(path: str):
    """ save the data to path"""
    with open(path, "w", encoding="utf-8") as f:
        for line in train_data:
            f.write(f"{line}\n")


if __name__ == "__main__":
    args = parser.parse_args()

    # getting the data
    train_data = make_data(args.train_n_digit_batch_size, args.max_number_length)
    eval_data = make_data(args.eval_n_digit_batch_size, args.max_number_length)

    # saving the data
    save_data(args.train_data_result_path)
    save_data(args.eval_data_result_path)

    print(f"training data saved to {args.train_data_result_path}")
    print(f"eval data saved to {args.train_data_result_path}")
