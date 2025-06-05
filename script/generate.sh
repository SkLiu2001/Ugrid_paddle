#!/bin/bash
# 参数解释
# --dataset_root 存储数据集根目录
python generate.py \
    --dataset_root "/root/autodl-tmp/data/Downloads/SynDat_1025/train" \
    --shape "curve" \
    --image_size 1025 \
    --start_index 1 \
    --num_instances 16000
