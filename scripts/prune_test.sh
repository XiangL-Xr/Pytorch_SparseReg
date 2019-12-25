# !/usr/bin/shell
# Author : lixiang

python main.py \
        --model=lenet5 \
        --dataset=MNIST \
        --epochs=40 \
        --lr_decay_every=20 \
        --rate=0.3 \
        --base_lr=0.001 \
        --model_path=checkpoints/lenet5_baseline.pth \
        --save_path=weights/lenet5/ \
        --sparse_reg=True \
        --target_reg=1.0 \
        --iter_size_retrain=4
