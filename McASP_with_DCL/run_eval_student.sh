#! /bin/bash



#test stuent model 
python main_dcl.py \
    --do_test \
    --test_data_path=./sample_data/ctb5/test.txt \
    --eval_model=./output/ctb5_models/bert_BU/model
