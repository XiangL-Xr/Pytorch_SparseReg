# !/usr/bin/python
# coding : utf-8
# Author : lixiang

import torch as t
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from torch.autograd import Variable

__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_bn']

model_urls = {'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
              'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
              'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
              'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
              'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'}

class VGG(nn.Module):
    def __init__(self, features, num_classes = 1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        #self.classifier = nn.Linear(cfg[-1], num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        ) 
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm = False):    
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size = 3, padding = 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace = True)]
            else:
                layers += [conv2d, nn.ReLU(inplace = True)]
            in_channels = v
   
    return nn.Sequential(*layers)


cfg = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
       'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
       'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

def vgg11(dataset, model_root = None, pretrained = False, **kwargs):
    '''VGG 11-layer model (configuration 'A')'''
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
        print("=> Pretrained model load successful!")
   
    return model

def vgg13(model_root = None, pretrained = False, **kwargs):
    '''VGG 13-layer model (configuration 'B')'''
    model = VGG(dataset, make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
        print("=> Pretrained model load successful!")
    #else:
    #    print("=> no checkpoint found at '{}'".format(model_root))

    return model

def vgg16(model_root = None, pretrained = False, **kwargs):
    '''VGG 16-layer model (configuration 'D')'''
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
        print("=> Pretrained model load successful!")
    
    return model

def vgg19(model_root = None, pretrined = False, **kwargs):
    '''VGG 19-layer model (configuration 'E')'''
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
        print("=> Pretrained model load successful!")
    
    return model

def vgg16_bn(model_root = None, pretrained = False, **kwargs):
    '''VGG 16-layer model (configuration 'D') with batch normalization'''
    kwargs.pop('model_root', None)
    model = VGG(make_layers(cfg['D'], batch_norm = True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn'], model_root))
        print("=> Pretrained model load successful!")

    return model

