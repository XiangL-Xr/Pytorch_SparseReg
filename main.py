# !/usr/bin/python
# coding : utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
import torch as t
import sys, time
import models
import shutil
import torch.nn as nn
from config import args
from data.dataset import Dataset
from torch.autograd import Variable as V
from models.lenet import lenet5
from models.vgg import vgg16, vgg16_bn
from models.resnet import resnet50
from Pruner.prune_methods import *
from utils.utils import *

best_prec1 = 0
def main():
    t.manual_seed(args.seed)
    if args.use_gpu:
        t.cuda.manual_seed_all(args.seed)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    global best_prec1
    # data load
    train_loader, test_loader = Dataset(args.dataset)
    # model load
    model_root = "checkpoints/"
    if args.model == "lenet5":
        model = lenet5(args.model_path, pretrained = True)
    elif args.model == "vgg16":
        model = vgg16(model_root, pretrained = True)
    elif args.model == "vgg16_bn":
        model = vgg16_bn(model_root, pretrained = True)
    elif args.model == "resnet50":
        model = resnet50(model_root, pretrained = True)

    # if use gpu
    device_ids = [0, 1, 2, 3]
    if t.cuda.is_available():
        if args.dev_nums > 1:
            model = t.nn.DataParallel(model, device_ids = device_ids)
            model.cuda()
        else:
            model = model.cuda()
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
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
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                model.load_state_dict(checkpoint)
            args.start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    
    # execute train procedure
    if args.training:
        for epoch in range(args.start_epoch, args.epochs + 1):
            adjust_learning_rate(optimizer, epoch)
            # train for one epoch
            train(model, train_loader, optimizer, criterion, epoch, None)
            prec1, prec5 = test(model, criterion, test_loader)
            # remember best acc@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_prec1': best_prec1,
                             'optimizer' : optimizer.state_dict()},
                             is_best = is_best,
                             filepath = args.save_path)
            print('--[Train]-- epoch: [{0}]\t'
                  'top_1: {prec1:.3f}, top_5: {prec5:.3f}\t'
                  'best_accuracy: ## {best_prec1:.3f} ##\t'
                  .format(epoch + 1,
                          prec1 = prec1,
                          prec5 = prec5,
                          best_prec1 = best_prec1))
    
    # our method, Incremental regularization pruning
    elif args.sparse_reg:
        m_reg = SparseRegularization()
        m_reg.init_rate(model, args.rate)
        if args.resume and args.state == "retrain":
            m_reg.check_mask(model)
        else:
            for epoch in range(1, args.epochs + 1):
                train(model, train_loader, optimizer, criterion, epoch, m_reg)
                if m_reg.IF_prune_finished:
                    save_model(model)
                    m_reg.state = "retrain"
                    args.state = "retrain"
                    break
    
    # execute retrain procedure
    if args.state == "retrain" or args.retraining:
        for epoch in range(args.start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)
            # train for one epoch
            train(model, train_loader, optimizer, criterion, epoch, m_reg) 
            # evaluate on test set
            prec1, prec5 = test(model, criterion, test_loader)
            # remember best acc@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': model.state_dict(),
                             'best_prec1': best_prec1,
                             'optimizer' : optimizer.state_dict()},
                             is_best = is_best,
                             filepath = args.save_path)
            print('=> app: [retrain] epoch: [{0}]\t'
                  'top_1: {prec1:.3f}, top_5: {prec5:.3f}\t'
                  'best_accuracy: ## {best_prec1:.3f} ##\t'
                  .format(epoch + 1,
                          prec1 = prec1,
                          prec5 = prec5,
                          best_prec1 = best_prec1))
    
    # execute test procedure
    if args.testing:
        test(model, criterion, test_loader)

def train(model, train_loader, optimizer, criterion, epoch, m_reg):
    """Train for one epoch on the training"""
    epoch_iter = 0
    losses = AverageMeter()
    batch_time = AverageMeter()
    if_print = True if (args.retraining or args.state == "retrain") else False
    iter_size = args.iter_size_retrain if (args.state == "retrain") else args.iter_size
    # switch to train mode
    model.train()
    end = time.time()
    for batch_idx, (data, label) in enumerate(train_loader):
        data, target = V(data), V(label)
        if args.use_gpu:
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        loss = criterion(output, target)
        losses.update(loss.item(), data.size(0))
        loss = loss / iter_size
        loss.backward()
        if (batch_idx + 1) % iter_size == 0:
            epoch_iter += 1
            if args.sparse_reg:                   # use sparse regularization to pruning
                m_reg.reg_pruning(model, epoch, losses)
            
            optimizer.step()
            optimizer.zero_grad()                 # gradient reset
            batch_time.update(time.time() - end)
            end = time.time()
            if args.sparse_reg and m_reg.state == "prune":
                m_reg.do_mask(model)              # avoid history gradient
            # Information printing of train or retrain process
            if (args.training or if_print) and (epoch_iter % args.print_freq) == 0:
                print('--[Train/Retrain]-- epoch: [{0}] [{1}/{2}]\t'
                      'loss: {loss.val:.4f}({loss.avg:.4f})\t'
                      'time: {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                      .format(epoch + 1,
                              epoch_iter,
                              math.ceil(len(train_loader)/iter_size),
                              loss = losses,
                              batch_time = batch_time))

def test(model, criterion, test_loader):
    """Perform test on the test set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top_1 = AverageMeter()
    top_5 = AverageMeter()
        
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
            pred1, pred5 = accuracy(output.data, target, top_k=(1, 5))
            losses.update(loss.item(), data.size(0))
            top_1.update(pred1.item(), data.size(0))
            top_5.update(pred5.item(), data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if args.testing:
                print('--[Test]-- iter: [{0}/{1}]\t'
                      'top_1: {top_1.avg:.3f}, top_5: {top_5.avg:.3f}\t'
                      'loss: {losses.avg:.3f}, time: {batch_time.sum:.3f}'
                      .format(batch_idx + 1,
                              len(test_loader),
                              top_1 = top_1,
                              top_5 = top_5,
                              losses = losses,
                              batch_time = batch_time)) 
    print('------------------------ Final Accuracy --------------------------')
    print('--[Test]-- top_1: {top_1.avg:.3f}, top_5: {top_5.avg:.3f}\t'
          'loss: {losses.avg:.3f}, time: {batch_time.sum:.3f}\t'
          .format(top_1 = top_1,
                  top_5 = top_5,
                  losses = losses,
                  batch_time = batch_time))
    print('-------------------------- Test End ------------------------------')
    return top_1.avg, top_5.avg

def save_model(model):
    prefix = os.path.join(args.save_path, args.model)
    filepath = time.strftime(prefix + '_%m-%d_%H:%M.pth')
    t.save(model.state_dict(), filepath)
    print("=> Checkpoint saved to {}".format(filepath))
    
if __name__ == '__main__':
    main()
