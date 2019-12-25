# !/usr/bin/python
# coding : utf-8
# Author : lixiang
# Func   : dataset process and load

import os, sys
sys.path.append('../')

import torch as t
from config import args
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data import DataLoader


def Dataset(name):
    
    if name == "MNIST":
        transform = T.Compose([
                        T.Resize((28, 28)),
                        T.ToTensor(),
                        T.Normalize(mean = [0.2860], std = [0.3205])
                    ])

        train_set = datasets.MNIST(
                        root = './data/mnist', 
                        train = True, 
                        download = True, 
                        transform = transform
                    )

        test_set = datasets.MNIST(
                        root = './data/mnist', 
                        train = False, 
                        download = True, 
                        transform = transform
                    )

        train_loader = DataLoader(train_set,
                                batch_size = args.batch_size,
                                shuffle = True,
                                num_workers = args.num_workers)
    
        test_loader = DataLoader(test_set,
                                batch_size = args.test_batch_size,
                                shuffle = False,
                                num_workers = args.num_workers)

    elif name == "ImageNet":
        train_dir = os.path.join('/home2/fengyushu/Dataset/ImageNet/', 'train')
        val_dir = os.path.join('/home2/fengyushu/Dataset/ImageNet/', 'val')
        normalize = T.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

        train_set = datasets.ImageFolder(
                        root = train_dir,
                        transform = T.Compose([
                            T.RandomResizedCrop(224),
                            T.RandomHorizontalFlip(),
                            T.ToTensor(),
                            normalize
                        ]))  
    
        test_set = datasets.ImageFolder(
                        root = val_dir,
                        transform = T.Compose([
                            T.Resize(256),
                            T.CenterCrop(224),
                            T.ToTensor(),
                            normalize
                        ]))

        train_loader = DataLoader(train_set,
                                    batch_size = args.batch_size,
                                    shuffle = True,
                                    num_workers = args.num_workers,
                                    pin_memory = True,
                                    sampler = None)

        test_loader = DataLoader(test_set,
                                    batch_size = args.test_batch_size,
                                    shuffle = False,
                                    num_workers = args.num_workers,
                                    pin_memory = True)

    elif name == "CIFAR10":
        transform = T.Compose([
                        T.Pad(4),
                        T.RandomCrop(32),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                    std = [0.2023, 0.1994, 0.2010])
                    ])

        train_loader = DataLoader(
                datasets.CIFAR10('./data/cifar10',
                                    train = True,
                                    download = True,
                                    transform = transform
                                ),
                batch_size = args.batch_size,
                shuffle = True,
                num_workers = args.num_workers,
                pin_memory = True)
    
        test_loader = DataLoader(
                datasets.CIFAR10('./data/cifar10',
                                    train = False,
                                    transform = T.Compose([
                                                T.ToTensor(),
                                                T.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))
                                ])),
                batch_size = args.test_batch_size,
                shuffle = True,
                num_workers = args.num_workers,
                pin_memory = True)

    elif name == "CIFAR100":
        transform = T.Compose([
                        T.Pad(4),
                        T.RandomCrop(32),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean = [0.4914, 0.4822, 0.4465],
                                    std = [0.2023, 0.1994, 0.2010])
                    ])

        train_loader = DataLoader(
                datasets.CIFAR100('./data/cifar100',
                                    train = True,
                                    download = True,
                                    transform = transform
                                 ),
                batch_size = args.batch_size,
                shuffle = True,
                num_workers = args.num_workers,
                pin_memory = True)
    
        test_loader = DataLoader(
                datasets.CIFAR100('./data/cifar100',
                                    train = False,
                                    transform = T.Compose([
                                                T.ToTensor(),
                                                T.Normalize((0.4914, 0.4822, 0.4465),
                                                            (0.2023, 0.1994, 0.2010))
                                ])),
                batch_size = args.test_batch_size,
                shuffle = True,
                num_workers = args.num_workers,
                pin_memory = True)
    
    return train_loader, test_loader
    
