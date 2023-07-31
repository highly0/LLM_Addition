# LLM_Addition

This repository is an extension work of the paper "How well do Large Language Models perform in Arithmetic tasks". The proposed method of number decomposition combined with Casual Language Modeling were used to achieve higher accuracy for long number addition. 

### Dataset Generation
To generate a new dataset, run the following command:
```bash
python3 src/make_data.py --train_data_result_path <path_to_result_train> \
--eval_data_result_path <path_to_result_eval> \
--max_number_length <max_length_digits> \
--train_n_digit_batch_size <batch_size_train> \
--eval_n_digit_batch_size <batch_size_eval>
```
The above command with create train and eval dataset of additions from 1 digit to `max_number_length` digits; save to `path_to_result_train` and `path_to_result_eval` respectively. For each digit length, the script will create `train_n_digit_batch_size` and `eval_n_digit_batch_size` number of samples for train and eval respectively. 

### Training from scratch
To train the CLM (`gpt2` was implemented, but this can be replaced with any CLM HuggingFace model), run the following command:
```bash
python3 src/main.py --train_data_file <path_to_train_file> \
--eval_data_file <path_to_eval_file> --output_dir <path_to_output> \
--model_name <CLM_hf_model_name> --train_batch_size <train_bs> \
--eval_batch_size <eval_bs> --num_train_epochs <number_epoches>
```
The saved model can be found in `--output_dir` after training has been finished.

### Evaluating
For convenience of checking the result of trained model, a zip of the best model weight can be found at `data/model_weights`. One can either check the performance of the models via the `src/inference.ipynb` or check any sets of equations generated in 
**Dataset Generation ** via src/evaluate.py. To do the latter, run the following command:

```bash
python3 src/evaluate.py --t <path_to_test_file> --m <path_to_model_weights>
```


### References
```bibtex
@misc{yuan2023large,
      title={How well do Large Language Models perform in Arithmetic tasks?}, 
      author={Zheng Yuan and Hongyi Yuan and Chuanqi Tan and Wei Wang and Songfang Huang},
      year={2023},
      eprint={2304.02015},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```