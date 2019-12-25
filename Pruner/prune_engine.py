# !/usr/bin/python
# coding : utf-8
# Author : lixiang
import os, sys
sys.path.append('../')

import time
import torch as t
import numpy as np
import torch.nn as nn
from config import args

def UpdateNumPrunedRow(model, IF_save_update_model = False):    
    print("=> UpdateNumPrunedRow...")
    layer_idx = 0
    layer_cnt = 0
    conv_layer = 0
    all_masks = {} 
    for name, p in model.named_parameters():
        if len(p.data.size()) == 4:
            conv_layer += 1
            masks = []
            interval = p.data.shape[2] * p.data.shape[3]            
            layer_idx += 1
            param_np = np.array(p.data.cpu()).reshape(p.data.shape[0], -1)
            sum_col_params = 0
            for i in range(param_np.shape[1]):
                sum_col_params += np.sum(np.abs(param_np[:, i]))
                #print("sum_col_params", sum_col_params)
                if (i+1) % interval == 0:
                    mask = sum_col_params != 0
                    masks.append(float(mask))
                    sum_col_params = 0
            all_masks[layer_idx] = masks
    # update row, set pruned row to zero
    for name, p in model.named_parameters():
        if len(p.data.size()) == 4:
            layer_cnt += 1
            # Not update some row of pruned in resnet50
            if args.model == "resnet50":
                if (layer_cnt == 1) or ("conv3" in name) or ("downsample" in name):
                    continue
            if layer_cnt < conv_layer:
                layer_row_masks = np.array(all_masks[layer_cnt + 1]).reshape(p.data.shape[0], 1, 1, -1)
                layer_row_masks = t.from_numpy(layer_row_masks).float().cuda()
                p.data.mul_(layer_row_masks)
    # save model
    if IF_save_update_model:
        prefix = os.path.join(args.save_path, args.model)
        filepath = time.strftime(prefix + '_update_row_%m-%d_%H:%M.pth')
        t.save(model.state_dict(), filepath)
        print("=> Checkpoint saved to {}".format(filepath))

def UpdateNumPrunedCol(model, IF_save_update_model = False):
    print("=> UpdateNumPrunedCol...")
    layer_idx = 0
    layer_cnt = 0
    all_masks = {}
    for p in model.parameters():
        if len(p.data.size()) == 4:
            masks = []
            layer_idx += 1
            param_np = np.array(p.data.cpu()).reshape(p.data.shape[0], -1)
            for i in range(param_np.shape[0]):
                mask = np.sum(np.abs(param_np[i, :])) != 0
                masks.append(float(mask))            
            all_masks[layer_idx] = masks
    # update col, set pruned col to zero
    for p in model.parameters():
        if len(p.data.size()) == 4:
            layer_cnt += 1
            if layer_cnt > 1:
                layer_col_masks = np.array(all_masks[layer_cnt - 1]).reshape(1, p.data.shape[1], 1, -1)
                layer_col_masks = t.from_numpy(layer_col_masks).float().cuda()
                p.data.mul_(layer_col_masks)
    # save model
    if IF_save_update_model:
        prefix = os.path.join(args.save_path, args.model)
        filepath = time.strftime(prefix + '_update_col_%m-%d_%H:%M.pth')
        t.save(model.state_dict(), filepath)
        print("=> Checkpoint saved to {}".format(filepath))

def UpdatePrunedRatio(model, IF_update_row_col):
    if IF_update_row_col:
        #print("row_prune_rate:", row_prune_rate)
        if args.weight_group == "Row":
            UpdateNumPrunedCol(model, args.IF_save_update_model)
        elif args.weight_group == "Col":
            UpdateNumPrunedRow(model, args.IF_save_update_model)
        else:
            print("--- Please assign the weight group: Row or Col! ---")
    else:
        print("--- Not update Row or Col! ---")


def Prune_rate_compute(model, verbose = True):
    """ Print out prune rate for each layer and the whole network """
    total_params = 0
    zero_params = 0
    layer_id = 0
    total_rows = 0
    total_cols = 0
    zero_rows = 0
    zero_cols = 0
    for p in model.parameters():        
        layer_zero_row_count = 0
        layer_zero_col_count = 0
        if len(p.data.size()) == 4:
            layer_row_nums = p.data.shape[0]
            layer_col_nums = p.data.shape[1] * p.data.shape[2] * p.data.shape[3]
            param_np = np.array(p.data.cpu()).reshape(p.data.shape[0], -1)
            for idx in range(layer_row_nums):
                if np.sum(np.abs(param_np[idx, :])) == 0:
                    layer_zero_row_count += 1
            for idx in range(layer_col_nums):
                if np.sum(np.abs(param_np[:, idx])) == 0:
                    layer_zero_col_count += 1
            total_rows += layer_row_nums
            total_cols += layer_col_nums
            zero_rows += layer_zero_row_count
            zero_cols += layer_zero_col_count
        layer_row_prune_rate = float(layer_zero_row_count) / layer_row_nums
        layer_col_prune_rate = float(layer_zero_col_count) / layer_col_nums
        
        # calculate parameters
        param_this_layer = 1
        for dim in p.data.size():
            param_this_layer *= dim
        total_params += param_this_layer
        # only pruning linear and conv layers
        if len(p.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = np.count_nonzero(p.cpu().data.numpy() == 0)
            zero_params += zero_param_this_layer
            if verbose:
                print("-----------------------------------------------------------------")
                print("Layer {} | {} layer | {:.2f}% rows pruned | {:.2f}% cols pruned"
                      .format(layer_id, 'Conv' if len(p.data.size()) == 4 else 'Linear',
                              100.*layer_row_prune_rate,
                              100.*layer_col_prune_rate))
                print("Layer {} | {} layer | {:.2f}% parameters pruned"
                      .format(layer_id, 'Conv' if len(p.data.size()) == 4 else 'Linear',
                              100.*zero_param_this_layer / param_this_layer))
    prune_rate = float(zero_params) / total_params
    row_prune_rate = float(zero_rows) / total_rows
    col_prune_rate = float(zero_cols) / total_cols
    if verbose:
        print("==================== Params Compress Rate =======================")
        print("Final row pruning rate: {:.2f}".format(row_prune_rate))
        print("Final col pruning rate: {:.2f}".format(col_prune_rate))
        print("Final params pruning rate: {:.2f}".format(prune_rate))
        print("=================================================================")    

    return row_prune_rate, col_prune_rate

# use to compute pruned model gflops(->compute_flops.py)
def layer_prune_rate(m):
    layer_zero_row_count = 0
    layer_zero_col_count = 0
    N, C, H, W = init_length(m)
    layer_row_nums = N
    layer_col_nums = C * H * W
    param_np = np.array(m.weight.data.cpu()).reshape(N, -1)
    for idx in range(layer_row_nums):
        if np.sum(np.abs(param_np[idx, :])) == 0:
            layer_zero_row_count += 1
    for idx in range(layer_col_nums):
        if np.sum(np.abs(param_np[:, idx])) == 0:
            layer_zero_col_count += 1

    layer_row_prune_rate = float(layer_zero_row_count) / layer_row_nums
    layer_col_prune_rate = float(layer_zero_col_count) / layer_col_nums
    return layer_row_prune_rate, layer_col_prune_rate


""" ---------------------- model flops calculation -----------------------"""
def check_flops(model):
    UpdateNumPrunedRow(model)
    get_flops(model)
    if args.model == "lenet5":
        input = t.zeros(1, 1, 28, 28)
    else:
        input = t.zeros(1, 3, 224, 224)
    if args.use_gpu:
        input = input.cuda()
    out = model(input)
    conv_flops_ = get_conv_flops(model)
    total_flops = sum(list_conv) + sum(list_linear)
    last_flops = conv_flops_ + sum(list_linear)
    return total_flops, last_flops

def get_flops(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(conv_hook)
        if isinstance(m, nn.Linear):
            m.register_forward_hook(linear_hook)

def get_conv_flops(model):
    conv_idx = -1
    conv_flops = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            conv_idx += 1
            row_rate, col_rate = layer_prune_rate(m)
            N, C, H, W = init_length(m)
            num_row_ = N * (1 - row_rate)
            num_col_ = C * H * W * (1 - col_rate)
            params_ = num_row_ * num_col_
            gflops_ = params_ * list_conv_fmap[conv_idx] ** 2
            conv_flops += gflops_
    return conv_flops

def init_length(m):
    N = m.weight.data.shape[0]
    C = m.weight.data.shape[1]
    H = m.weight.data.shape[2]
    W = m.weight.data.shape[3]
    return N, C, H, W

# hook list initialization
list_conv = []
list_linear = []
list_conv_fmap = []

# hook calculation
def conv_hook(self, input, output):
    batch_size, input_channels, input_height, input_width = input[0].size()
    output_channels, output_height, output_width = output[0].size()
    kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
    bias_ops = 1 if self.bias is not None else 0
    params = output_channels * (kernel_ops + bias_ops)
    flops = batch_size * params * output_height * output_width
    list_conv.append(flops)
    list_conv_fmap.append(output_height)

def linear_hook(self, input, output):
    batch_size = input[0].size(0) if input[0].dim() == 2 else 1
    weight_ops = self.weight.nelement()
    bias_ops = self.bias.nelement()
    flops = batch_size * (weight_ops + bias_ops)
    list_linear.append(flops)
