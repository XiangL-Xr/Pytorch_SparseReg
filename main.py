# !/usr/bin/py://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/5thon
# coding : utf-8
# Author : lixiang
# Time   : 09-18 21:25

import os,sys
import torch as t
import argparse
import time
import torch.nn as nn
import models
import shutil

from config import args
from data.dataset import Dataset
from torch.autograd import Variable as V
from models.lenet import lenet5
from models.vgg import vgg16
from models.resnet import resnet50

from Pruner.prune_methods import *
from utils.utils import AverageMeter, accuracy, grad_zero, ID_Reg_Infoprint, adjust_learning_rate

def train(model, train_loader, optimizer, criterion, epoch, ID_Reg, test_loader = None):
    
    """Train for one epoch on the training"""
    # train
    losses = AverageMeter()
    top_1 = AverageMeter()
    top_5 = AverageMeter()
    batch_time = AverageMeter()
    acc_tracker = 0.0
    loss_tracker = 0.0
    loss_tracker_num = 0

    # switch to train mode
    model.train()
    end = time.time()

    for batch_idx, (data, label) in enumerate(train_loader):
        # train model
        data, target = V(data), V(label)
        if args.use_gpu:
            data = data.cuda()
            target = target.cuda()

        # make sure that all gradients are zero
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)
                
        # measure accuracy and record loss
        losses.update(loss.item(), data.size(0))
        pred_1, pred_5 = accuracy(output.data, target, top_k=(1, 5))
        top_1.update(pred_1.item(), data.size(0))
        top_5.update(pred_5.item(), data.size(0))

        acc_tracker += pred_1.item()
        loss_tracker += loss.item()
        loss_tracker_num += 1

        loss.backward()

        # use sparsity regularization to pruning
        if args.sparse_reg:
            if ID_Reg.prune_state == "prune":
                ID_Reg.prune_step += 1
                ID_Reg.Update_IncReg(model, ID_Reg.prune_step)

            elif ID_Reg.prune_state == "losseval":
                ID_Reg.losseval_step += 1
                ID_Reg.retrain(model)
                if ID_Reg.losseval_step > ID_Reg.losseval_interval:
                    ID_Reg.IF_losseval_finished = True
                    continue

            elif ID_Reg.prune_state == "retrain":
                ID_Reg.retrain_step += 1
                #print("ID_Reg_retrain_step:", ID_Reg.retrain_step)
                ID_Reg.retrain(model)
                if (ID_Reg.retrain_step % ID_Reg.retrain_test_interval) == 0:
                    prec1, prec5 = test(model, test_loader, criterion)
                    is_best = prec1 > ID_Reg.best_prec1
                    ID_Reg.best_prec1 = max(prec1, ID_Reg.best_prec1)
                    print('--[Train->retrain]-- Epoch: [{0}], Retrain_step: [{1}]\t'
					      'Top_1: {prec1:.4f}, Top_5: {prec5:.4f}, Best accuracy: ## {best_prec1:.4f} ##\t'
                          .format(
							    epoch,
							    ID_Reg.retrain_step,
                                prec1 = prec1,
                                prec5 = prec5,
							    best_prec1 = ID_Reg.best_prec1
					    ))
				    #print("Best accuracy: " + str(best_prec1))
                    if is_best:
                        ID_Reg.prec1_decay_freq = 0
                        save_checkpoint({'epoch': epoch,
                                         'retrain_step': ID_Reg.retrain_step,
                                         'state_dict': model.state_dict(),
                                         'best_prec1': ID_Reg.best_prec1},
                                         is_best,
									     filepath = args.save_path)
                    else:
                        ID_Reg.prec1_decay_freq += 1
                        if ID_Reg.prec1_decay_freq == ID_Reg.prec1_decay_nums:
                            ID_Reg.current_lr = adjust_learning_rate(optimizer)
                            ID_Reg.lr_decay_freq += 1
                            ID_Reg.prec1_decay_freq = 0
                            
                            if ID_Reg.lr_decay_freq == 3:
                                ID_Reg.IF_retrain_finished = True
                                ID_Reg.prune_state = "stop"
                            else:
                                checkpoint = t.load(args.save_path + 'model_best.pth')
                                best_accuracy = checkpoint['best_prec1']
                                model.load_state_dict(checkpoint['state_dict'])
                                print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                                      .format((args.save_path+'model_best.pth'), checkpoint['epoch'], best_accuracy))

            elif ID_Reg.prune_state == "stop":
                ID_Reg.retrain(model)
                continue

        # if retrain or not 
        if args.retrain_flag:
            grad_zero(model)

        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        # print sparsity regularizaion information
        if args.sparse_reg:
            if ID_Reg.IF_retrain_finished:
                continue
            else:
                ID_Reg_Infoprint(ID_Reg, epoch, losses, batch_time)
                      
        elif args.train_flag and ((batch_idx) % args.print_freq == 0):
            print('--[Train]-- Epoch: [{0}] [{1}/{2}]\t'
                'Time {batch_time.val:.3f}({batch_time.avg:.3f}), Loss {loss.val:.4f}({loss.avg:.4f})\t'
                'Top_1 {top_1.val:.3f}({top_1.avg:.3f}), Top_5 {top_5.val:.3f}({top_5.avg:.3f})\t'
                .format(
                    epoch,
                    batch_idx,
                    len(train_loader),
                    batch_time = batch_time,
                    loss = losses,
                    top_1 = top_1,
                    top_5 = top_5
                ))

def test(model, test_loader, criterion):
    """Perform test on the test set"""
    print("---> test start <---")
    #import ipdb;
    #ipdb.set_trace()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top_1 = AverageMeter()
    top_5 = AverageMeter()

	# object function
    #criterion = t.nn.CrossEntropyLoss() 
        
	# switch to evaluate mode
    model.eval()
    end = time.time()
    
    with t.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
                
            if args.use_gpu:
                data = data.cuda()
                target = label.cuda()
                
            output = model(data)

            if args.get_inference_time:
                iters_get_inference_time = 100
                start_time = time.time()
                for i in range(iters_get_inference_time):
                    output = model(data)
                end_time = time.time()
                print("time taken for %d iterations, per_iteration times is: "
                                                %(iters_get_inference_time),
                                                (end_time - start_time)*1000.0/
                                                float(iters_get_inference_time), "ms")

            loss = criterion(output, target)

            pred_1, pred_5 = accuracy(output.data, target, top_k=(1, 5))
                
            losses.update(loss.item(), data.size(0))
            top_1.update(pred_1.item(), data.size(0))
            top_5.update(pred_5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if args.test_flag:
                print('--[Test]-- Iters: [{0}/{1}]\t'
                        'Top_1 {top_1.avg:.3f}, Top_5 {top_5.avg:.3f}\t'
                        'Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}'
                        .format(batch_idx + 1,
                                len(test_loader),
                                top_1 = top_1,
                                top_5 = top_5,
                                batch_time = batch_time,
                                losses = losses) ) 
    print('-----------------------------Final Accuracy------------------------------')
    print('--[Test]-- Top_1 {top_1.avg:.3f}, Top_5 {top_5.avg:.3f}\t'
                        'Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}'
                        .format(top_1 = top_1,
                                top_5 = top_5,
                                batch_time = batch_time,
                                losses = losses) )
    print('------------------------------- Test End---------------------------------')

    return top_1.avg, top_5.avg

def one_shot_prune(model):
        
    # Prune Hyper Parameters
    # prune_param = {'prune_method': args.prune_method,
    #                'weight_group': args.weight_group,
    #                'prune_rate': args.prune_rate,
    #                'prune_rate_step': args.prune_rate_step}

    if args.prune_method == "weight_prune":
        weight_prune(model, args.prune_rate)
        print("--- {:.2f}% parameters pruned ---".format(100. *args.prune_rate))
    elif args.prune_method == "L1_norm":
        L1_norm(model, args.prune_rate, args.weight_group)
        print("--- {:.2f}% each layer filters pruned ---".format(100. *args.prune_rate))

    save_model(model)

def save_model(model):
        
    model_save_path = args.save_path
    prefix = os.path.join(model_save_path, args.model)
    new_save_path = time.strftime(prefix +'_%m-%d_%H:%M.pth')
    t.save(model.state_dict(), new_save_path)
    print("Checkpoint saved to {}".format(new_save_path))

def save_checkpoint(state, is_best, filepath):
    
    t.save(state, os.path.join(filepath, 'checkpoint.pth'))
    print("Checkpoint saved to {}".format(os.path.join(filepath, 'checkpoint.pth')))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth'), os.path.join(filepath, 'model_best.pth'))

def main():

    print("-------------------------------------Sparsity Regularization----------------------------------------")
    print("---> data prepare <---")
    t.manual_seed(args.seed)
    if args.use_gpu:
        t.cuda.manual_seed(args.seed)
   
    # data load
    train_loader, test_loader = Dataset(args.dataset, args.iter_size)
        
    model_root = "checkpoints/"
    # model load
    if args.model == "lenet5":
        model = lenet5(args.model_path, pretrained = True)
    # model: vgg
    elif args.model == "vgg16":
        model = vgg16(model_root, pretrained = True, dataset = args.dataset)
    # model: resnet
    elif args.model == "resnet50":
        model = resnet50(model_root, pretrained = True)
        
    if args.retrain:
        if args.model == "lenet5":
            model = lenet5()
        elif args.model == "vgg16":
            model = vgg16()
        elif args.model == "resnet50":
            model = resnet50()
        
        checkpoint = t.load(args.retrain)
        model.load_state_dict(checkpoint)

    # use gpu or not
    if args.use_gpu:
        model.cuda()

    # object function
    criterion = t.nn.CrossEntropyLoss()
    optimizer = t.optim.SGD(model.parameters(),
                                lr = args.base_lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay)
        
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = t.load(args.resume)
            if "best_prec1" in checkpoint:
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
                       .format(args.resume, checkpoint['epoch'], best_prec1))
            else:
                model.load_state_dict(checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.oneshot_pruning:
        one_shot_prune(model)
    # trian or test
    if args.train_flag:
        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, optimizer, criterion, epoch, None)
        save_model(model)

    elif args.test_flag:
        test(model, test_loader, criterion)       
    
    elif args.train_flag and args.test_flag:
        for epoch in range(1, args.epochs + 1):
            train(model, train_loader, optimizer, criterion, epoch, None)
        save_model(model)
        test(model, test_loader, criterion)
	
	# sparsity regularization method
    if args.sparse_reg:
        # prune variables initialization
        ID_Reg = SparseRegularization()
        
        if args.resume:
            conv_cnt = -1
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    conv_cnt += 1
                    layer_ix = str(conv_cnt)
                    ID_Reg.masks[layer_ix] = (m.weight.data != 0).float().cuda()

        if ID_Reg.prune_state == "prune":
            print("---> Start Pruning Stage... <---")
            train_loader, _ = Dataset(args.dataset, ID_Reg.iter_size_prune)
            for epoch in range(1, args.epochs + 1):
                train(model, train_loader, optimizer, criterion, epoch, ID_Reg)
                if ID_Reg.IF_prune_finished:
                    print("---> All layers Pruning Finished! <---")
                    ID_Reg.prune_state = "losseval"
                    break
            save_model(model)
        
        if ID_Reg.prune_state == "losseval":
            print("---> Start Losseval Stage... <---")
            train_loader, _ = Dataset(args.dataset, ID_Reg.iter_size_losseval)
            for epoch in range(1, args.epochs + 1):                    
                train(model, train_loader, optimizer, criterion, epoch, ID_Reg)
                if ID_Reg.IF_losseval_finished:
                    print("---> Losseval Stage Finished, Start Retrain Stage... <---")
                    ID_Reg.prune_state = "retrain"
                    break

        if ID_Reg.prune_state == "retrain":
            train_loader, test_loader = Dataset(args.dataset, ID_Reg.iter_size_retrain)
            for epoch in range(1, args.epochs + 1):
                train(model, train_loader, optimizer, criterion, epoch, ID_Reg, test_loader)
                if ID_Reg.IF_retrain_finished:
                    print("---> Retrain Stage Finished, and All stages End!! <---")
                    break
                
                                
if __name__ == '__main__':
    main()
