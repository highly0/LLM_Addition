""" make train and eval data"""
import argparse
from random import randint
from random import shuffle


parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_data_result_path",
    default="/workspace/llm_addition_baseline/data/train/train_add_decomp_8_digits.txt",
    type=str,
    help="The output training data file (a text file).",
)

parser.add_argument(
    "--eval_data_result_path",
    default="/workspace/llm_addition_baseline/data/test/test_add_8_digits.txt",
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
    default=3000,
    type=int,
    help="for each n length digit addition, this is the batch size for training",
)

parser.add_argument(
    "--eval_n_digit_batch_size",
    default=1000,
    type=int,
    help="for each n length digit addition, this is the batch size for eval",
)

NUMBER_TO_NAME = {
    1: "units",
    2: "tens",
    3: "hundreds",
    4: "thousands",
    5: "ten thousands",
    6: "hundred thousands",
    7: "millions",
    8: "ten millions",
    9: "hundred millions",
    10: "billions",
    11: "ten billions",
    12: "hundred billions",
    13: "trillions",
    14: "ten trillions",
    15: "hundred trillions",
    16: "quadrillions",
    17: "ten quadrillions",
    18: "hundred quadrillions",
    19: "quintillions",
    20: "ten quintillions",
    21: "hundred quintillions",
}


def random_n_number(length: int) -> int:
    """return a random number of length"""
    range_start = 10 ** (length - 1)
    range_end = (10**length) - 1
    return randint(range_start, range_end)


def extract_digit(number, k):
    """given a number, return the kth digit"""
    c = 0
    result = 0
    while c <= k:
        result = number % 10
        number = int(number / 10)
        c += 1
    return result


def decompose_number(number):
    """given number, decompose it"""
    number_len = len(str(number))
    res = ""
    for i in range(number_len):
        curr_number = extract_digit(number, i)
        sep = "" if i == number_len - 1 else ", "
        res += f"{curr_number} {NUMBER_TO_NAME[i+1]}{sep}"
    return res


def decompose_number_prompt(number, last_num=False):
    """given number, decompose it and return the prompt"""
    decomp_num = decompose_number(number)
    if not last_num:
        res = f"Translate from number to decomposition: {number} = {decomp_num}"
    else:
        res = f"Translate from decomposition to number: {decomp_num} = {number}"
    return res


def decompose_addition_prompt(number1, number2, res):
    """given our addition, decompose it"""
    decomp_n1 = decompose_number(number1)
    decomp_n2 = decompose_number(number2)
    decomp_res = decompose_number(res)
    res = f"Sum {decomp_n1} + {decomp_n2} = {decomp_res}"
    return res


def make_data(size: int, digit_size: int, is_eval=False) -> [int]:
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
            if not is_eval:
                n1_decompose = decompose_number_prompt(n1)
                n2_decompose = decompose_number_prompt(n2)
                addition_decompose = decompose_addition_prompt(n1, n2, res)
                res_decompose = decompose_number_prompt(res, last_num=True)

                prompt = f"Compute with pipeline {n1} plus {n2}. {n1_decompose}. {n2_decompose}. {addition_decompose}. {res_decompose}"
            else:
                prompt = f"Compute with pipeline {n1} plus {n2} =  {res}"
            data.append(prompt)
    return data


def save_data(path: str, data):
    """save the data to path"""
    with open(path, "w+", encoding="utf-8") as f:
        for line in data:
            f.write(f"{line}\n")


if __name__ == "__main__":
    args = parser.parse_args()

    # getting the data
    train_data = make_data(args.train_n_digit_batch_size, args.max_number_length)
    shuffle(train_data)  # shuffle the train data
    save_data(args.train_data_result_path, train_data)
    print(f"training data saved to {args.train_data_result_path}")

    eval_data = make_data(
        args.eval_n_digit_batch_size, args.max_number_length, is_eval=True
    )
    shuffle(eval_data)  # shuffle the train data
    save_data(args.eval_data_result_path, eval_data)
    print(f"eval data saved to {args.eval_data_result_path}")
