# !/usr/bin/shell
# Author : lixiang
# Time   : 09-19 21:52

python -u main.py \
        --epochs=100 \
        --model=vgg16 \
        --batch_size=32 \
        --test_batch_size=10 \
        --base_lr=0.0005 \
        --weight_decay=0.00025 \
        --dataset=ImageNet \
        --num_workers=8 \
        --rate=0.7 \
        --target_reg=2.5 \
        --save_path=weights/vgg16/ \
        --sparse_reg=True
