# !/usr/bin/python
# coding : utf-8
# Author : lixiang
# Time   : 09-25 21:42

import os, sys
sys.path.append('../')

import time
import torch as t
import numpy as np
from config import args

def UpdateNumPrunedRow(model, IF_save_update_model = False):    
    print("---> UpdateNumPrunedRow <---")
    layer_idx = 0
    layer_cnt = 0
    all_masks = {}
    conv_layer = 0
    # Not update num row of pruned in resnet50
    
    for name, p in model.named_parameters():
        if len(p.data.size()) == 4:
            conv_layer += 1
            masks = []
            interval = p.data.shape[2] * p.data.shape[3]
            
            layer_idx += 1
            param_np = np.array(p.data.cpu()).reshape(p.data.shape[0], -1)
            
            sum_col_params = 0
            #print("model_name:", name)
            for i in range(param_np.shape[1]):
                sum_col_params += np.sum(np.abs(param_np[:, i]))
                #print("sum_col_params", sum_col_params)
                if (i+1) % interval == 0:
                    mask = sum_col_params != 0
                    masks.append(float(mask))
                    sum_col_params = 0
            
            all_masks[layer_idx] = masks
    #print("masks1--:", all_masks[1])
    #print("masks2--:", all_masks[2])
    
    for name, p in model.named_parameters():
        if len(p.data.size()) == 4:
            layer_cnt += 1
            
            # Not update some row of pruned in resnet50
            if args.model == "resnet50":
                if layer_cnt == 1 or "conv3" in name or "downsample" in name:
                    continue
            
            if layer_cnt < conv_layer:
                layer_row_masks = np.array(all_masks[layer_cnt + 1]).reshape(p.data.shape[0], 1, 1, -1)
                layer_row_masks = t.from_numpy(layer_row_masks).float().cuda()
                p.data.mul_(layer_row_masks)
    
    # save model
    if IF_save_update_model:
        prefix = "/home1/lixiang/Envs/workspace/Pytorch_SparseReg/weights/update_row_col/"
        save_path = time.strftime(prefix + args.model + '_update_row_%m-%d_%H:%M.pth')
        t.save(model.state_dict(), save_path)
        print("Checkpoint saved to {}".format(save_path))


def UpdateNumPrunedCol(model, IF_save_update_model = False):
    print("---> UpdateNumPrunedCol <---")
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
    #print("masks1--:", all_masks[1])
    #print("masks2--:", all_masks[2])
    
    for p in model.parameters():
        if len(p.data.size()) == 4:
            layer_cnt += 1
            if layer_cnt > 1:
                layer_col_masks = np.array(all_masks[layer_cnt - 1]).reshape(1, p.data.shape[1], 1, -1)
                layer_col_masks = t.from_numpy(layer_col_masks).float().cuda()
                p.data.mul_(layer_col_masks)
    
    # save model
    if IF_save_update_model:
        prefix = "/home1/lixiang/Envs/workspace/Pytorch_SparseReg/weights/update_row_col/"
        save_path = time.strftime(prefix + args.model + '_update_col_%m-%d_%H:%M.pth')
        t.save(model.state_dict(), save_path)
        print("Checkpoint saved to {}".format(save_path))


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


def Prune_rate_compute(model, verbose = True, struct = False):
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
        
        #print("layer_zero_row_count", layer_zero_row_count)
        layer_row_prune_rate = float(layer_zero_row_count) / layer_row_nums
        layer_col_prune_rate = float(layer_zero_col_count) / layer_col_nums
        if layer_row_prune_rate > 0 or layer_col_prune_rate > 0:
            struct = True
        
        param_this_layer = 1
        for dim in p.data.size():
            param_this_layer *= dim
            #print("dim----", dim)
        
        #print("param_this_layer", param_this_layer)
        total_params += param_this_layer

        # only pruning linear and conv layers
        if len(p.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = np.count_nonzero(p.cpu().data.numpy() == 0)
            
            #print("zero_param_this_layer", zero_param_this_layer)
            zero_params += zero_param_this_layer

            if verbose:
                print("-----------------------------------------------------------------")
                if struct:
                    print("Layer {} | {} layer | {:.2f}% rows pruned | {:.2f}% cols pruned".format(
                           layer_id, 'Conv' if len(p.data.size()) == 4 else 'Linear', 
                           100.*layer_row_prune_rate, 
                           100.*layer_col_prune_rate))

                print("Layer {} | {} layer | {:.2f}% parameters pruned".format(
                            layer_id, 'Conv' if len(p.data.size()) == 4
                            else 'Linear', 100.*zero_param_this_layer / param_this_layer))
    
    prune_rate = float(zero_params) / total_params
    row_prune_rate = float(zero_rows) / total_rows
    col_prune_rate = float(zero_cols) / total_cols

    if verbose:
        print("=================================================================")
        if struct:
            print("Final row pruning rate: {:.2f}".format(row_prune_rate))
            print("Final col pruning rate: {:.2f}".format(col_prune_rate))

        print("Final params pruning rate: {:.2f}".format(prune_rate))
        print("=================================================================")
    
    if struct:
        return row_prune_rate, col_prune_rate
    else:
        return prune_rate

# use to compute pruned model gflops(->compute_flops.py)
def layer_prune_rate(m):
    layer_zero_row_count = 0
    layer_zero_col_count = 0

    layer_row_nums = m.weight.data.shape[0]
    layer_col_nums = m.weight.data.shape[1] * m.weight.data.shape[2] * m.weight.data.shape[3]

    param_np = np.array(m.weight.data.cpu()).reshape(m.weight.data.shape[0], -1)
    for idx in range(layer_row_nums):
        if np.sum(np.abs(param_np[idx, :])) == 0:
            layer_zero_row_count += 1
    for idx in range(layer_col_nums):
        if np.sum(np.abs(param_np[:, idx])) == 0:
            layer_zero_col_count += 1

    layer_row_prune_rate = float(layer_zero_row_count) / layer_row_nums
    layer_col_prune_rate = float(layer_zero_col_count) / layer_col_nums

    return layer_row_prune_rate, layer_col_prune_rate
