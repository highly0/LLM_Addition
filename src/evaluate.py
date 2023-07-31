import re
import argparse
import torch
import transformers
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()

parser.add_argument("-t", "--testset_path", type=str)
parser.add_argument("-m", "--model_path", type=str)


def process_output(output_strings):
    """given the model outputs, extract the predicted addition result"""
    output_numbers = []
    for output_string in output_strings:
        fraction = output_string.split("=")[-1].strip()
        number = fraction.split("!")[0]
        number = int(number) if number.isnumeric() else 0
        output_numbers.append(number)
    return output_numbers


def generate_answer(inputs, batch_size=512):
    """given the input embeddings, generate model outputs"""
    dataset = TensorDataset(inputs["input_ids"], inputs["attention_mask"])
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    output_strings = []
    for batch in tqdm(dataloader):
        batch_input_ids = batch[0].to(device)
        outputs = model.generate(
            input_ids=batch_input_ids,
            pad_token_id=50256,  # eos
            do_sample=False,
            max_length=150,
        )
        decoded_ouputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_strings.extend(decoded_ouputs)

    processed_ouputs = process_output(output_strings)
    return processed_ouputs


def parse_test_file(path):
    """get the input promps and correct label from test file"""
    inputs = []
    labels = []

    with open(path, "r", encoding="utf-8") as infile:
        for line in infile:
            splitted = line.split("=")
            input_prompt = splitted[0].strip() + ". "
            inputs.append(input_prompt)
            labels.append(int(splitted[1].strip()))

    return inputs, labels


def get_final_results(inputs, predictions, labels):
    """with our pred and true label, get final metrics"""
    final_dict = {}
    final_dict["total"] = 0
    for input_, pred, label in zip(inputs, predictions, labels):
        n_len = len(re.findall(r"\d+", input_)[0])
        curr_label = f"{n_len}_digit"
        if curr_label not in final_dict:
            final_dict[curr_label] = 0
            final_dict[f"{curr_label}_len"] = 0  # how many samples we have

        final_dict[f"{curr_label}_len"] += 1
        if pred == label:
            # collect how many rights based on digit length
            if curr_label in final_dict:
                final_dict[curr_label] += 1
            final_dict["total"] += 1

    for res_type in list(final_dict):
        curr_res_cnt = final_dict[res_type]
        if res_type == "total":
            final_dict[f"{res_type}_acc"] = curr_res_cnt / len(labels)
        elif res_type[1:] == "_digit":  # find acc scores
            final_dict[f"{res_type}_acc"] = curr_res_cnt / final_dict[f"{res_type}_len"]

    mse_res = mean_squared_error(predictions, labels)
    final_dict["final_mse"] = mse_res

    return final_dict


if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = transformers.AutoModelWithLMHead.from_pretrained(args.model_path).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    # encode our input prompts -> ready for model
    input_prompts, correct_results = parse_test_file(args.testset_path)
    encoded_inputs = tokenizer.batch_encode_plus(
        input_prompts,
        add_special_tokens=False,
        return_tensors="pt",
        pad_to_max_length=True,
        truncation=True,
    )
    results = generate_answer(encoded_inputs)
    results_dict = get_final_results(input_prompts, results, correct_results)

    save_path = "/".join(args.model_path.split("/")[:-1])
    test_type = (args.testset_path.split)("/")[-1]

    with open(f"{save_path}/eval_result_{test_type}", "w+", encoding="utf-8") as file:
        for result_type, value in results_dict.items():

            file.write(f"{result_type}: {value}\n")
            print(f"{result_type}: {value}")
