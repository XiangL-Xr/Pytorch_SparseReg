# !/usr/bin/shell
# -*- coding: utf-8 -*-

python -u main.py \
        --model=vgg16 \
        --batch_size=256 \
        --test_batch_size=20 \
        --base_lr=0.001 \
        --rate=0.5 \
        --save_path=weights/vgg16_2x/ \
        --sparse_reg=True \
        --skip=True \
        --dev_nums=4
