#!/bin/bash


python test.py \
    --checkpoint_root "/root/autodl-tmp/model" \
    --load_experiment "20250605-010942" \
    --load_epoch 3 \
    --dataset_root "/root/autodl-tmp/data/Downloads/SynDat_1025/" \
    --num_workers 8 \
    --batch_size 1 \
    --seed 9590589012167207234



# python test.py \
#     --checkpoint_root "./var/checkpoint" \
#     --load_experiment "22" \
#     --load_epoch 300 \
#     --dataset_root "/root/autodl-tmp/data/Downloads/SynDat_1025" \
#     --num_workers 8 \
#     --batch_size 1 \
#     --seed 9590589012167207234
