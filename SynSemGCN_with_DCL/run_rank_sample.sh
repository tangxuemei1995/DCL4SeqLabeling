#! /bin/bash
#use teacher model rank training data (data-level CL)
CUDA_VISIBLE_DEVICES=0 python  main_baseline.py \
    --do_rank \
    --use_bayesian \
    --dataset_name=ctb5\
    --dropout_times=3\
    --eval_batch_size=16 \
    --train_data_path=./sample_data/ctb5/train.txt  \
    --eval_data_path=./sample_data/ctb5/dev.txt \
    --test_data_path=./sample_data/ctb5/test.txt \
    --eval_model=./output/ctb5_/bert/ctb5_bert_5/model.pt