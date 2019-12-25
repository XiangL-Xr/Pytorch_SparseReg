# !/usr/bin/python
# coding : utf-8
# Author : lixiang
# Func   : increg pruning methods
import os, sys
sys.path.append('../')

import math
import time
import torch as t
import numpy as np
import torch.nn as nn
from config import args

# Integrate the idea of IncReg to Pruning --------------------
class SparseRegularization(object):

    def __init__(self):
        self.init_hyparams()
        self.init_regular()
        
        self.state = args.state
        self.flag = False
        self.IF_prune_finished = False
    
    def init_hyparams(self):
        self.kk = 0.25
        self.AA = 0.00025
        self.iter_size = 1
        self.prune_iter = -1
        self.speedup = 0
    
    def init_regular(self):
        self.skip_idx = []
        self.compress_rate = {}
        self.history_score = {}
        self.Reg = {}
        self.masks = {}
        self.pruned_rate = {}
        self.num_pruned_col = {}
        self.IF_col_alive = {}
        self.IF_col_pruned = {}
        self.IF_layer_finished = {}

    def init_rate(self, model, layer_rate):
        conv_cnt = -1
        self.init_skiplayer()
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                conv_cnt += 1; index = str(conv_cnt)
                self.IF_layer_finished[index] = 0
                if not args.skip:
                    self.compress_rate[index] = layer_rate
                else:
                    self.compress_rate[index] = 0 if (conv_cnt in self.skip_idx) else layer_rate

                print("Conv_" + index + " initialize compress ratio:", self.compress_rate[index])
    
    def init_skiplayer(self):
        if args.skip and (args.model == "vgg16"):
            self.skip_idx = [0, 12] if (args.rate > 0.5) else [0, 10, 11, 12]
        elif args.skip and (args.model == "resnet50"):
            self.skip_idx = [0]
        print(args.model + " skip layer initialization: ", self.skip_idx)

    def init_register(self, index, m, num_col):
        # initialization/registrztion
        if index not in self.Reg:
            self.Reg[index] = [0] * num_col
            self.masks[index] = [1] * num_col
            self.IF_col_alive[index] = [1] * num_col
            self.num_pruned_col[index] = 0
            self.pruned_rate[index] = 0
            self.history_score[index] = [0] * num_col
            self.init_mask(index, m)
            if self.compress_rate[index] == 0:
                self.IF_layer_finished[index] = 1
        
        num_pruned_col_ = self.num_pruned_col[index]
        num_col_ = num_col - num_pruned_col_
        num_col_to_prune_ = math.ceil(num_col * self.compress_rate[index]) - num_pruned_col_
        return num_col_, num_col_to_prune_

    def init_length(self, m):
        N = m.weight.data.shape[0]
        C = m.weight.data.shape[1]
        H = m.weight.data.shape[2]
        W = m.weight.data.shape[3]
        return N, C, H, W

    def update_increg(self, model):
        conv_cnt = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                conv_cnt += 1; index = str(conv_cnt)
                N, C, H, W = self.init_length(m)
                num_col = C * H * W
                W_data = np.array(m.weight.data.cpu()).reshape(N, -1)
                num_col_, num_col_to_prune_ = self.init_register(index, m, num_col)

                # if all conv layers pruning finished
                if all(self.IF_layer_finished.values()):
                    self.IF_prune_finished = True
                    continue
                elif self.IF_layer_finished[index] == 1:
                    m.weight.data.mul_(self.masks[index])
                    m.weight.grad.data.mul_(self.masks[index])
                    continue
                    
                """ ## Start pruning """
                if num_col_to_prune_ > 0:
                    # step 01: get importance ranking
                    col_hist_rank = self.get_rank_score(index, W_data, num_col)
                    # step 02: sparse regularization
                    self.do_regular(index, m, num_col_, num_col_to_prune_, num_col, col_hist_rank)
                    # step 03: get mask matrix
                    self.get_mask(index, m)

                """ ## Mask out the gradient and weights """
                m.weight.grad.data.mul_(self.masks[index])
                m.weight.data.mul_(self.masks[index])

    
    def get_rank_score(self, index, W_data, num_col):
        # execute once every one step
        if self.prune_iter % args.prune_interval == 0:
            """ ### Sort 01: sort by L1-norm """
            col_score = {}
            col_score_first = []
            col_score_second = []
            for j in range(num_col):
                col_score_first.append(j)
                col_score_second.append(np.sum(np.fabs(W_data[:, j])))
            col_score = dict(zip(col_score_first, col_score_second))
            col_score_rank = sorted(col_score.items(), key = lambda k: k[1])

            # Make new criteria, i.e. history_rank, by rank
            # No.n iter, n starts from 1
            n = self.prune_iter + 1                     
            for rk in range(num_col):
                col_of_rank_rk = col_score_rank[rk][0]
                self.history_score[index][col_of_rank_rk] = ((n - 1) * self.history_score[index][col_of_rank_rk] + rk) / n

            """ ### Sort 02: sort by history rank """
            # the history_rank of each column, history_rank is like the new score
            col_hrank = {}
            col_hrank_first = []
            col_hrank_second = []
            for j in range(num_col):
                col_hrank_first.append(j)
                col_hrank_second.append(self.history_score[index][j])
            col_hrank = dict(zip(col_score_first, col_score_second))
            col_hist_rank = sorted(col_hrank.items(), key = lambda k: k[1])

        return col_hist_rank

    def do_regular(self, index, m, num_col_, num_col_to_prune_, num_col, col_hist_rank):
        # Note the real rank is i + num_pruned_col_
        num_pruned_col_ = num_col - num_col_
        for i in range(num_col_):
            col_of_rank_i = col_hist_rank[i + num_pruned_col_][0]
            Delta = self.punish_func(i, num_col_, num_col_to_prune_)
            self.Reg[index][col_of_rank_i] = max(self.Reg[index][col_of_rank_i] + Delta, 0)
        
            if self.Reg[index][col_of_rank_i] >= args.target_reg:
                self.IF_col_alive[index][col_of_rank_i] = 0
                self.num_pruned_col[index] += 1
                self.pruned_rate[index] = self.num_pruned_col[index] / num_col  
                if self.pruned_rate[index] >= self.compress_rate[index]:
                    self.IF_layer_finished[index] = 1        
        
        """ ### Apply reg to conv weights """
        _, C, H, W = self.init_length(m)
        Reg_tmp = np.array(self.Reg[index]).reshape(-1, C, H, W)
        Reg_new = t.from_numpy(Reg_tmp).float().cuda()
        # use L2 regularization
        m.weight.grad.data.add_(Reg_new * m.weight.data)

    def init_mask(self, index, m):
        # initialization mask to 4-dim tensor
        _, C, H, W = self.init_length(m)
        tmp = np.array(self.masks[index]).reshape(-1, C, H, W)
        self.masks[index] = t.from_numpy(tmp).float().cuda()

    def check_mask(self, model):
        conv_cnt = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                conv_cnt += 1; index = str(conv_cnt)
                _, C, H, W = self.init_length(m)
                if index not in self.masks:
                    self.masks[index] = [1] * (C * H * W)
                mask = (m.weight.data != 0)
                self.masks[index] = mask.float().cuda()
        
    def get_mask(self, index, m):
        # generate mask matrix
        _, C, H, W = self.init_length(m)
        tmp = np.array(self.IF_col_alive[index]).reshape(-1, C, H, W)
        self.masks[index] = t.from_numpy(tmp).float().cuda()

    def do_mask(self, model):
        conv_cnt = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                conv_cnt += 1; index = str(conv_cnt)
                m.weight.grad.data.mul_(self.masks[index])
                m.weight.data.mul_(self.masks[index])
                #print("self.masks:", self.masks[index])
    
    def reg_pruning(self, model, epoch, losses):
        if self.state == "prune":
            start_time = time.time()
            self.prune_iter += 1
            self.update_increg(model)
            batch_time = time.time() - start_time
            self.info_print(epoch, losses, batch_time)
            
            if self.IF_prune_finished:
                self.flag = True
                self.info_print(epoch, losses, batch_time)
                self.state = "losseval"
                print("=> pruning stage finished, start losseval...")
        
        elif self.state == "losseval":
            self.do_mask(model)
        
        elif self.state == "retrain":
            self.do_mask(model)

    def info_print(self, epoch, losses, batch_time):
        if (self.prune_iter % args.print_freq) == 0 or self.flag:
            for key, value in self.pruned_rate.items():
                print('--[Train->prune]-- epoch:[{0}], iter:[{1}], conv-{key:}\t'
                      'rate: [{value:.3f}/{PR:.3f}], lr: #{lr:}#\t'
                      'loss: {loss.val:.3f}({loss.avg:.3f}), time: {batch_time:.3f}\t'
                      .format(epoch,
                              self.prune_iter,
                              key = key,
                              value = value,
                              PR = self.compress_rate[key],
                              lr = args.base_lr,
                              loss = losses,
                              batch_time = batch_time))

    # Punish Function 1 ----------------
    def punish_scheme1(self, r, num_col_to_prune_):
        alpha = math.log(2/self.kk) / num_col_to_prune_
        N = -math.log(self.kk) / alpha
        if r < N:
            return self.AA * math.exp(-alpha*r)
        else:
            return (2*self.kk*self.AA) - (self.AA*math.exp(-alpha*(2*N-r)))
    
    # Punish Function 2 ----------------
    def punish_func(self, r, num_g, thre_rank):
        if r <= thre_rank:
            return self.AA - (self.AA/thre_rank*r)
        else:
            return -self.AA / (num_g-1-thre_rank) * (r-thre_rank)
