# Pytorch_SparseReg

## Baseline models
| **Model** | **Accuracy(%)** | **Download url** |
| :-: | :-: | :-: |
| vgg16 | 71.59/90.38 | https://download.pytorch.org/models/vgg16-397923af.pth |
| resnet50 | 76.15/92.87 | https://download.pytorch.org/models/resnet50-19c8e357.pth |

Note:
  * The baseline accuracies are obtained by evaluating the downloaded model *without* finetuning them on our produced ImageNet dataset.

## Environment
  * Ubuntu 16.04
  * Python 3.6.7
  * Pytorch 0.4.1
  * Numpy 1.15.4
  * Use CUDA

## How to run the code
  1. Download this repository    
  2. Here we show how to run the code, there are two ways to run the code, taking lenet5 as an example:  
  * Preparation:  
      * Data: The mnist dataset will be downloaded automatically when run the code(if you want to use ImageNet dataset, please manually modify the path of ImageNet dataset in the file *data/dataset.py*)  
      * Pretrained model: We provide a pretrained in `checkpoints/lenet5_baseline.pth`(if you want to use vgg or resnet series network, the pretrained model will be downloaded automatically in `checkpoints/`)  
  * How to run the code:  
      * Run on the command line:  
        * In your pytorch root path, run `CUDA_VISIBLE_DEVICES='<gpu_id>' python main.py --model lenet5 --epochs 100 --base_lr 0.0005 --weight_decay 0.00025 --dataset MNIST --rate 0.4 --sparse_reg True --save_path weights/lenet5/`, then check your log at terminal.  
      * Run with a script:
        * In your pytorch root path, run `CUDA_VISIBLE_DEVICES='<gpu_id>' nohup sh sparse_reg.sh > weights/lenet5/prunt_output.log 2>&1 /dev/null &`, then check your log at `weights/lenet5/prune_output.log`.  
  3. Similarly, for vgg16 or resnet50, you can run the code in the same way.  
## Check the log  
  * There is a log file generated during pruning: `prune_output.log`. It is saved in `weights/lenet5/`.  
  * then run  `cat weights/lenet5/prune_output.log | grep Prune_rate` you will see the current pruning rate, and run  `cat weights/lenet5/prune_output.log | grep Best` you will see the current best accuracy of model during retraining stage.       
