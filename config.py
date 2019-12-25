# !/usr/bin/python
# coding : utf-8
# Author : lixiang

import argparse
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default:64)')
parser.add_argument('--test_batch_size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default:50)')
parser.add_argument('--iter_size', type=int, default=1, metavar='N',
                    help='real batch size equal batch_size*iter_size')
parser.add_argument('--base_lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default:0.01)')
parser.add_argument('--model', default="lenet5", type=str,
                    help='model selection, choices: lenet5, vgg16, resnet50',
                    choices=["lenet5", "vgg16", "vgg16_bn", "resnet50"])
parser.add_argument('--dataset', default="ImageNet", type=str,
                    help='dataset for experiment, choice: MNIST, CIFAR10, CIFAR100, ImageNet',
                    choices=["MNIST", "CIFAR10", "ImageNet"])
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=90, metavar='N',
                    help='number of epochs to train (default:90)')
parser.add_argument('--use_gpu', default=True, type=bool, metavar='N',
                    help='use gpu or not, (default:True), choice: True, False')
parser.add_argument('--dev_nums', default=1, type=int, metavar='N',
                    help='use one gpu or multiple gpu to train')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay (default:1e-4)')
parser.add_argument('--lr_decay_every', type=int, default=30,
                    help='learning rate decay by 10 every X epochs')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default:0.9)')
parser.add_argument('--print_freq', type=int, default=20,
                    help='print info every N batch, (default:20)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='how many workers for loading data, (default:4)')
parser.add_argument('--model_path', default='checkpoints/', type=str,
                    help='the path of load pre-train model, (default:checkpoints/model.pth)')
parser.add_argument('--save_path', default='weights/', type=str,
                    help='finally save output model path,(default:weights/model.pth')
parser.add_argument('--get_inference_time', default=False, type=bool, nargs='?',
                    help='runs test multiple times and reports the result')
parser.add_argument('--training', default=False, type=bool, 
                    help='train flag, default:False')
parser.add_argument('--testing', default=False, type=bool,
                    help='test flag, default:False')
parser.add_argument('--retraining', default=False, type=bool,
                    help='retrain flag, default:False')
parser.add_argument('--arch', default='resnet', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=50, type=int,
                    help='depth of the neural network')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=1, type=int, metavar='S',
                    help='random seed (default: 1)')


# ------------------Pruning settings
parser.add_argument('--weight_group', default='Col', type=str,
                    help='pre prune basic units, default=filter',
                    choices=['Row', 'Col', 'Channel'])
parser.add_argument('--sparse_reg', default=False, type=bool, metavar='sr',
                    help='train with columns sparsity regularization')
parser.add_argument('--rate', default=0.4, type=float, metavar='N',
					help='prune ratio of model')					
parser.add_argument('--IF_update_row_col', default=False, type=bool,
					help='if update row sparse ratio or col sparse ratio')
parser.add_argument('--IF_save_update_model', default=False, type=bool,
                    help='if save model of updated row sparse rate or col sparse rate')
parser.add_argument('--state', default='prune', type=str,
                    help='the start state of SparseReg prune method,choices: prune, losseval, retrain',
                    choices=["prune", "losseval", "retrain"])
parser.add_argument('--skip', default=False, type=bool,
                    help='if some layer not prune, skip it layer number')


# --------IncReg Method---------
parser.add_argument('--target_reg', default=2.5, type=float)
parser.add_argument('--prune_interval', default=1, type=int)
parser.add_argument('--iter_size_retrain', default=1, type=int)
# ------------------End pruning added

# -------check pruning---------
parser.add_argument('--weights', default='weights/prune_lenet5.pth', type=str,
                    help='path to pickled weights')

args = parser.parse_args()
