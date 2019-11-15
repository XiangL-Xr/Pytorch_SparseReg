# !/usr/bin/python
# coding : utf-8
# Author : lixiang
# Time   : 09-19 21:25

from __future__ import print_function
import os, sys
sys.path.append('../')

import math
import torch as t
import numpy as np
import torch.nn as nn
from config import args
from torch.autograd import Variable as V

t.manual_seed(args.seed)
if args.use_gpu:
    t.cuda.manual_seed(args.seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):        
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, top_k=(1,)):
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

def updateBN(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.scale * t.sign(m.weight.data))   # L1

def grad_zero(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            mask = (m.weight.data != 0)
            #print("mask_----------", mask)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.weight.data.mul_(mask)
            #m.bias.grad.data.mul_(mask)
        
        elif isinstance(m, nn.Linear):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            #m.bias.grad.data.mul_(mask)

def BN_grad_zero(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            mask = (m.weight.data != 0)
            mask = mask.float().cuda()
            m.weight.grad.data.mul_(mask)
            m.bias.grad.data.mul_(mask)
            
def ID_Reg_Infoprint(ID_Reg, epoch, losses, batch_time):
    if ID_Reg.prune_state == "prune" and ((ID_Reg.prune_step % ID_Reg.NUM_SHOW) == 0):
        for key, value in ID_Reg.pruned_ratio.items():
            print('--[Train->prune]-- Epoch: [{0}], Prune_step: [{1}], Conv_{key:}\t'
                  'Prune_rate: [{value:.3f}/{PR:.3f}], Current LR: #{lr:}#\t'
                  'Loss {loss.val:.4f}({loss.avg:.4f}), Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                  .format(
                        epoch,
                        ID_Reg.prune_step,
                        key = key,
                        value = value,
                        PR = ID_Reg.PR,
                        lr = ID_Reg.current_lr,
                        loss = losses,
                        batch_time = batch_time
                ))
    
    elif ID_Reg.prune_state == "losseval" and ((ID_Reg.losseval_step % ID_Reg.NUM_SHOW) == 0):
        print('--[Train->losseval]-- Epoch: [{0}], Losseval_step: [{1}]\t'
			  'Current LR: #{lr:}#\t'
              'Loss {loss.val:.4f}({loss.avg:.4f}), Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
              .format(
                    epoch,
                    ID_Reg.losseval_step,
                    lr = ID_Reg.current_lr,
                    loss = losses,
                    batch_time = batch_time
			))
    elif ID_Reg.prune_state == "retrain" and ((ID_Reg.retrain_step % ID_Reg.NUM_SHOW) == 0):
        print('--[Train->retrain]-- Epoch: [{0}], Retrain_step: [{1}]\t'
              'Current LR: #{lr:}#\t'
              'Loss {loss.val:.4f}({loss.avg:.4f}), Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
              .format(
                    epoch,
                    ID_Reg.retrain_step,
                    lr = ID_Reg.current_lr,
                    loss = losses,
                    batch_time = batch_time
			  ))

def adjust_learning_rate(optimizer):
    lr_scale = args.lr_decay_scalar
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr_scale
        current_lr = param_group['lr']
    print("---> learning rate adjusted:", current_lr)
    
    return current_lr
