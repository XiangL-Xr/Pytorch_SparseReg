# !/usr/bin/shell
# Author : lixiang
# Time   : 09-19 21:52

python -u main.py \
        --epochs=100 \
        --model=lenet5 \
		--dataset=MNIST \
        --batch_size=32 \
        --test_batch_size=10 \
		--sparse_reg=True
		--rate=0.4 \
        --base_lr=0.0005 \
        --weight_decay=0.00025 \
        --save_path=weights/lenet5/ 