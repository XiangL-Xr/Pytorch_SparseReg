# !/usr/bin/python
# coding : utf-8
# Author : lixiang
import os 
os.environ['CUDA_VISIBLE_DEVICES']='2'

import torch as t
import torch.nn as nn
import numpy as np

from config import args
from Pruner.prune_engine import *
from models.lenet import lenet5
from models.vgg import vgg16
from models.resnet import resnet50

if args.model == "lenet5":
    model = lenet5()
elif args.model == "vgg16":
    model = vgg16()
elif args.model == "resnet50":
    model = resnet50()

if args.dev_nums > 1:
    model = nn.DataParallel(model)

if args.use_gpu:
    model.cuda()

if args.weights:
    if os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        #checkpoint = t.load(args.weights, map_location='cuda:0')
        checkpoint = t.load(args.weights)
        if "best_prec1" in checkpoint:
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                  .format(args.weights,
                          checkpoint['epoch'],
                          best_prec1))
        else:
            model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(args.weights))
    else:
        print("=> no checkpoint found at '{}'".format(args.weigths))

if args.IF_update_row_col:
    UpdatePrunedRatio(model, args.IF_update_row_col)

total_flops, last_flops = check_flops(model)
Prune_rate_compute(model, verbose = True)
print("==================== Flops Compress Rate ========================")
print('Total number of flops: {:.2f}G'.format(total_flops / 1e9))
print('Last number of flops: {:.2f}G'.format(last_flops / 1e9))
print("Final model speedup rate: {:.2f}".format(total_flops / last_flops))
print("=================================================================")
