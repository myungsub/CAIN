#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py \
    --exp_name CAIN_eval \
    --dataset snufilm \
    --data_root data/SNU-FILM \
    --test_batch_size 1 \
    --model cain \
    --depth 3 \
    --mode test \
    --resume \
    --resume_exp CAIN_train \
    --test_mode hard