# !/usr/bin/python
# coding : utf-8
# Author : lixiang
import os, sys
sys.path.append('../')

import math
import time
import shutil
import torch as t
from config import args

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

def adjust_learning_rate(optimizer, epoch):
    """ Sets the learning rate to the initial LR decayed by 10 every X epochs """
    lr = args.base_lr * (0.1 ** (epoch // args.lr_decay_every))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if (epoch % args.lr_decay_every) == 0:
        print("=> app: learning rate adjusted to #", lr, "#")

def save_checkpoint(state, is_best, filepath):
    name = 'checkpoint'
    filename = time.strftime(name + '_%m-%d_%H:%M.pth')
    t.save(state, os.path.join(filepath, filename))
    print("=> Checkpoint saved to {}".format(os.path.join(filepath, filename)))
    if is_best:
        shutil.copyfile(os.path.join(filepath, filename), os.path.join(filepath, 'model_best.pth'))
