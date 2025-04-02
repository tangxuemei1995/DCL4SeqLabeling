#! /bin/bash

# train teacher model
CUDA_VISIBLE_DEVICES=0 python  main_baseline.py \
    --do_train \
    --train_data_path=./sample_data/ctb5/train.txt \
    --eval_data_path=./sample_data/ctb5/test.txt \
    --bert_model=./base_model/bert \
    --decoder=softmax  \
    --max_seq_length=160 \
    --max_ngram_size=160 \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --num_train_epochs=5 \
    --warmup_proportion=0.1 \
    --learning_rate=2e-5 \
    --dataset_name=ctb5\
    --model_set=/bert/ctb5_bert_5