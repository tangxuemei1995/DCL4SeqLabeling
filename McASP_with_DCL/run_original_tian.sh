#! /bin/bash

python main_dcl.py --do_train \
    --do_train \
    --use_attention \
    --use_bert \
    --train_data_path=./sample_data/ctb5/train.txt \
    --eval_data_path=./sample_data/ctb5/test.txt \
    --test_data_path=./sample_data/ctb5/test.txt \
    --bert_model=./base_model/bert \
    --decoder=softmax  \
    --max_seq_length=160 \
    --max_ngram_size=160 \
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --num_train_epochs=70 \
    --warmup_proportion=0.1 \
    --patient=100 \
    --learning_rate=2e-5 \
    --ngram_threshold=2 \
    --cat_type=freq \
    --ngram_type=pmi \
    --av_threshold=2 \
    --dataset_name=ctb5\
    --voc=voc_train_dev_ngrams_jieba.txt \
    --model_set=models/tian_bert


python main_dcl.py \
    --do_test \
    --test_data_path=./sample_data/ctb5/test.tsv \
    --eval_model=./output/ctb5_models/tian_test_bert/model
