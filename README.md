# AITISA工作组PyTorch平台基准代码

## 简介
本仓库为PyTorch平台的基准代码，包含深度网络的剪枝、量化、编码部分。

## 环境
- Python == 3.6.7
- PyTorch == 0.4.1
- Numpy == 1.15.4

## 参数
- `--model`： 模型名字(采用PyTorch官方提供的模型)
- `--dataset`： 数据集名字(default: ImageNet2012)
- `--start_epoch`: 训练起始epoch(default: 0)
- `--epochs`： 训练模型最大epoch数
- `--batch_size`： 训练集batch_size大小
- `--iter_size`： batch_size = batch_size * iter_size
- `--test_batch_size`： 测试集batch_size大小
- `--use_gpu`： 是否使用GPU(default: True)
- `--momentum`： default: 0.9
- `--weight_decay`： default: 1e-4

### 剪枝参数
- `--sparse_reg`： 稀疏正则化剪枝算法(采用增量正则化方法)
- `--rate`： 各层剪枝率
- `--base_lr`： 剪枝模型的初始学习率(default: 0.001)
- `--weight_group`： 剪枝的基本单元(default: 'Col')
- `--skip_idx`： 设定不剪枝层
- `--target_reg`： 正则化上限值
- `--state`： 剪枝流程初始状态(default: 'prune')
- `--prune_interval`： 剪枝间隔
- `--save_path`: 输出模型及log日志的保存路径

## 剪枝（包括retrain）

### 示例
#### vgg16 剪枝及重训练（压缩率=50%）
``` shell
python main.py --model vgg16  --batch_size 256 --test_batch_size 20 --base_lr 0.001 --sparse_reg True --rate 0.5 --skip True --dev_nums 4 --save_path ${dir to save weights}

``` or
nohup ./script/vgg16_2x_prune.sh > weights/vgg16_2x/vgg16_2x_prune_output.log 2>&1 /dev/null &
```


## 模型精度

### baseline模型
network | top1-accuracy | top5-accuracy | download_url
--------|---------------|---------------| -----------|
VGG16 | 0.7159 | 0.9038	| https://download.pytorch.org/models/vgg16-397923af.pth
ResNet50 | 0.7615 | 0.9287 | https://download.pytorch.org/models/resnet50-19c8e357.pth


### 剪枝模型
network | prune | top1-accuracy | top5-accuracy | speedup |
--------|-------|---------------|---------------|---------|
VGG16 | 0.50 | 0.7178 | 0.9047 | 2.02x |
VGG16 | 0.69 | 0.7004 | 0.8952 | 4.00x |
ResNet50 | 0.40 | 0.7314 | 0.9121 | 2.05x |


## check_prune.py使用

### 参数
- `--model`:  模型名字
- `--weights`： 模型参数文件路径(model.pth)
- `--weight_group`： 权重组(Row or Col, default: Col)
- `--IF_update_row_col`： 是否更新剪枝模型的行或列(default: False)
- `--IF_save_update_model`:  是否保存更新行列之后的模型(default: False)

### 示例
#### vgg16 列剪枝，对相应的行数进行更新并输出更新后的各层剪枝率及模型加速比
```shell
python check_prune.py --model vgg16 --weights weights/vgg16_2x/model_best.pth --IF_update_row_col True
```
