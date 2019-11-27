#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py \
    --exp_name CAIN_test_noca \
    --dataset vimeo90k \
    --batch_size 16 \
    --test_batch_size 16 \
    --model cain_noca \
    --depth 3 \
    --loss 1*L1 \
    --max_epoch 200 \
    --lr 0.0002 \
    --log_iter 100 \
#    --mode test
#   --resume True \
#   --resume_exp SH_5_12
#   --fix_encoder