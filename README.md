# Backbone with DCL

We use model from Tian 2020[1] and Tang 2024[2] as the backbone model.

[1] Yuanhe Tian, Yan Song, and Fei Xia. 2020b. Joint Chinese word segmentation and part-of-speech tagging via multi-channel attention of character n-grams. In Proceedings of the 28th International Conference on Computational Linguistics, pages 2073–2084, Barcelona, Spain (Online). International Committee on Computational Linguistics.

[2] Tang, X., Wang, J., & Su, Q. (2024). Incorporating knowledge for joint Chinese word segmentation and part-of-speech tagging with SynSemGCN. Aslib Journal of Information Management, ahead-of-print(ahead-of-print). https://doi.org/10.1108/AJIM-07-2023-0263

## Requirements

Our code works with the following environment.
* `python=3.8`
* `pytorch=1.13`

You can find all pakege version from `environment_config.txt`.
Use `pip install -r requirements.txt` to install the required packages.

## Encoder BERT

In our paper, we use BERT and RoBERTa as the encoder.

## Run on Data

Run `run_train_teahcer.sh` to train a teacher model.
Run `run_rank_sample.sh` use tacher model rank training samples.
Run `run_train_student.sh` to train the student model.
Run `run_eval_student.sh` to evaluate the student model.

## Datasets

CTB5, CTb6, PKU

You can download all data from:

Due to Tang(2024) use GCN, so you need download the graph for each dataset. Then unzip them in each data file.

## Training and Testing


Here are some important parameters:

* `--do_train`: train the model.
* `--do_test`: test the model.
* `--use_bert`: use BERT as encoder.
* `--use_curri`: use DCL.
* `--bert_model`: the directory of pre-trained BERT/RoBERTa model.
* `--use_bayesian`: use BU as the diffculty metrics.
* `--use_top_k_LC"`: use TLC as the diffculty metrics.
* `--use_nor_log_P`: use MNLP as the diffculty metrics.
* `--use_length`: use length as the diffculty metrics.
* `--use_data_level`: use tacher model ranked training samples.
* `--model_set`: the name of model to save.

