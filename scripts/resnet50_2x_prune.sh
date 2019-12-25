# !/usr/bin/shell
# -*- coding: utf-8 -*-

python -u main.py \
        --model=resnet50 \
        --batch_size=64 \
        --iter_size=4 \
        --test_batch_size=20 \
        --base_lr=0.001 \
        --rate=0.4 \
        --save_path=weights/resnet50_2x/ \
        --sparse_reg=True \
        --skip=True
