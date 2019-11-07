# !/usr/bin/python
# coding : utf-8
# Author : lixiang
# Time   : 09-18 20:55

import os, sys
sys.path.append('../')

import torch as t
import torch.nn.functional as F
import torch.nn as nn

class LeNet(nn.Module):
    
    def __init__(self, num_classes = 10):
        super(LeNet, self).__init__()

        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
       
        #print("org_x_size", x.size())
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(self.relu1(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(self.relu2(self.conv2(x)), 2, padding = 1)
     
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def lenet5(model_root = None, pretrained = False):
    """LeNet 5-layer model configuration"""
    model = LeNet(num_classes = 10)
    if pretrained:
        if os.path.isfile(model_root):
            model.load_state_dict(t.load(model_root), strict=False)
            print("---> Pretrained model load successful! <---")
        else:
            print("=> no checkpoint found at '{}'".format(model_root))

    return model
