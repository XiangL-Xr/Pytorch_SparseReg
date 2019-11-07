# !/usr/bin/python
# coding : utf-8
# Author : lixiang
# Time   : 2019-11-01 09:50
# Func   : Compute model floating point operands

import os
os.environ['CUDA_VISIBLE_DEVICES']='4'

import torch as t
import torch.nn as nn
import torchvision
import torch.nn as nn
import numpy as np

from config import args
from thop import profile
from torch.autograd import Variable
from models.vgg import vgg16
from models.resnet import resnet50
from Pruner.prune_engine import layer_prune_rate

list_conv = []
list_linear = []
list_bn = []
list_relu = []
list_pooling = []
list_conv_fmap = []

def conv_hook(self, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()
    
    kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
    bias_ops = 1 if self.bias is not None else 0

    params = output_channels * (kernel_ops + bias_ops)
    flops = batch_size * params * output_height * output_width

    list_conv.append(flops)
    list_conv_fmap.append(output_height)
    #print("output_width:", output_width)
    #print("output_height:", output_height)
    #print("----------------------------")

def linear_hook(self, input, output):
    batch_size = input[0].size(0) if input[0].dim() == 2 else 1
    weight_ops = self.weight.nelement()
    bias_ops = self.bias.nelement()

    flops = batch_size * (weight_ops + bias_ops)
    list_linear.append(flops)

def bn_hook(self, input, output):
    list_bn.append(input[0].nelement())

def relu_hook(self, input, output):
    list_relu.append(input[0].nelement())

def pooling_hook(self, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()

    kernel_ops = self.kernel_size * self.kernel_size
    bias_ops = 0
    params = output_channels * (kernel_ops + bias_ops)
    flops = batch_size * params * output_height * output_width

    list_pooling.append(flops)

def calculate_flops(model):
    #childrens = list(model.children())
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(conv_hook)
        if isinstance(m, nn.Linear):
            m.register_forward_hook(linear_hook)
        if isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(bn_hook)
        if isinstance(m, nn.ReLU):
            m.register_forward_hook(bn_hook)
        if isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AvgPool2d):
            m.register_forward_hook(pooling_hook)


def last_conv_flops(model):
    conv_idx = -1
    conv_flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_idx += 1
            row_rate, col_rate = layer_prune_rate(m)
            #print("row_rate, col_rate:", row_rate, col_rate)
            num_row_ = m.weight.data.shape[0] * (1 - row_rate)
            num_col_ = m.weight.data.shape[1] * m.weight.data.shape[2] * m.weight.data.shape[3] * (1 - col_rate)
            params_ = num_row_ * num_col_
            gflops_ = params_ * list_conv_fmap[conv_idx] * list_conv_fmap[conv_idx]
            conv_flops += gflops_
    
    return conv_flops

# load model
if args.model == "vgg16":
    model = vgg16()
elif args.model == "resnet50":
    model = resnet50()

if args.use_gpu:
    model.cuda()

# load weights
if args.weights:
    if os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        checkpoint = t.load(args.weights)
        if "best_prec1" in checkpoint:
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                   .format(args.weights, checkpoint['epoch'], best_prec1))
        else:
            model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(args.weights))
    else:
        print("=> no checkpoint found at '{}'".format(args.weights))

# calculate the model gflops)
calculate_flops(model)
input = t.zeros(1, 3, 224, 224)
out = model(input)

# calculate the model conv layer gflops of pruned model
conv_flops_ = last_conv_flops(model)

total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling))
last_flops = conv_flops_ + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling)
print('-------------------------------------------------')
print('=> Total Number of FLOPs: %.2f G' % (total_flops / 1e9))
print('=> Last Number of FLOPs: %.2f G' % (last_flops / 1e9))
print('=> Model Final Speedup: %.2f x' % (total_flops / last_flops))
print('-------------------------------------------------')
