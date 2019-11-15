# !/usr/bin/python
# coding : utf-8
# Author : lixiang
# Time   : 09-25 19:44
# Func   : pruning methods

import os, sys
sys.path.append('../')

import math
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from config import args


""" ----------------------------one-shot pruning-------------------------------"""

def weight_prune(model, prune_rate):
    """ prune weights globally * prune_rate (not layer-wise)"""
    all_weights = []
    for param in model.parameters():
        if len(param.data.size()) != 1:
            all_weights += list(param.cpu().data.abs().numpy().flatten())

    for index, item in enumerate(sorted(all_weights)):
        if index == (len(all_weights) * prune_rate):
            threshold = item
    #print("threshold", threshold, type(threshold))
    
    # generate masks
    for p in model.parameters():
        
        masks = []
        if len(p.data.size()) != 1:
            pruned_inds = (np.abs(np.array(p.data.cpu())) > np.array(threshold)).astype(float)
            #pruned_inds = t.as_tensor(np.array(pruned_inds, dtype=float))
            masks.append(pruned_inds)
            if len(p.data.size()) == 4:
                masks = np.array(masks).reshape(p.data.shape[0], p.data.shape[1], p.data.shape[2], p.data.shape[3])
            else:
                masks = np.array(masks).reshape(p.data.shape[0], p.data.shape[1])
            
            masks = t.from_numpy(masks).float().cuda()

            # weights set zero
            p.data.mul_(masks)
    
    print("---> one_shot pruning: weight prune finished! <---")
    #return masks

def L1_norm(model, prune_rate, weight_group):
    
    for p in model.parameters():
        
        masks = []
        if len(p.data.size()) == 4:              # selecting conv layer
            params_np = np.abs(np.array(p.data.cpu()))
            layer_rows = p.data.shape[0]
            layer_cols = p.data.shape[1] * p.data.shape[2] * p.data.shape[3]
            p_temp = params_np.reshape(p.data.shape[0], -1)
            
            # get l1_norm for each filter or columns this layer
            row_values = p_temp.sum(axis=1)
            col_values = p_temp.sum(axis=0)
            
            # sort()
            row_order_values = np.sort(row_values)
            col_order_values = np.sort(col_values)

            row_thre_index = int(layer_rows * prune_rate) - 1
            col_thre_index = int(layer_cols * prune_rate) - 1

            row_thre = row_order_values[row_thre_index]
            col_thre = col_order_values[col_thre_index]

            if weight_group == "Row":         # Row == Filter
                mask = (row_values > row_thre).astype(float)
                masks.append(mask)
                masks = np.array(masks).reshape(p.data.shape[0], 1, 1, -1)
                #print("mask.shape", masks.shape)
                masks = t.from_numpy(masks).float().cuda()
                # weights set zero
                p.data.mul_(masks)

            elif weight_group == "Col":
                mask = (col_values > col_thre).astype(float)
                masks.append(mask)
                masks = np.array(masks).reshape(-1, p.data.shape[1], p.data.shape[2], p.data.shape[3]) 
                masks = t.from_numpy(masks).float().cuda()
                # weights set zero
                p.data.mul_(masks)
    
    print("---> one_shot pruning: L1_Norm prune finished! <---")
""" ---------------------------------- END --------------------------------------"""


""" ----------------------------Iterative Pruning--------------------------------"""

# Integrate the idea of IncReg to Pruning -----------------------------------------
class SparseRegularization(object):

    def __init__(self):
        self.kk = 0.25
        self.AA = 0.00025	
        self.prune_step = -1
        self.losseval_step = -1
        self.retrain_step = -1
        self.best_prec1 = 0
        self.lr_decay_freq = 0
        self.prec1_decay_freq = 0
        self.Reg = {}
        self.masks = {}
        self.IF_col_alive = {}
        self.num_pruned_col = {}
        self.IF_layer_finished = {}
        self.pruned_ratio = {}
        self.history_score = {}
        self.IF_col_pruned = {}
        self.IF_prune_finished = False
        self.IF_losseval_finished = False
        self.IF_retrain_finished = False
        self.skip_idx = []
        self.IF_skip_prune = args.IF_skip_prune
        self.prune_state = args.prune_state
        self.PR = args.rate
        self.NUM_SHOW = args.NUM_SHOW
        self.Target_reg = args.target_reg
        self.current_lr = args.base_lr
        self.prec1_decay_nums = args.prec1_decay_nums
        self.prune_interval = args.prune_interval
        self.iter_size_prune = args.iter_size_prune
        self.iter_size_losseval = args.iter_size_losseval
        self.iter_size_retrain = args.iter_size_retrain
        self.losseval_interval = args.losseval_interval
        self.retrain_test_interval = args.retrain_test_interval

    def Update_IncReg(self, model, step_):
        conv_cnt = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                prune_unit = args.weight_group
                conv_cnt += 1
                layer_idx = str(conv_cnt)
                num_col = m.weight.data.shape[1] * m.weight.data.shape[2] * m.weight.data.shape[3]
                if self.IF_skip_prune and (conv_cnt in self.skip_idx):
                    continue

                """ ## pruning initialization """
                if prune_unit == "Col":
                    if layer_idx not in self.Reg:
                        self.Reg[layer_idx] = [0] * num_col
                        self.masks[layer_idx] = [1] * num_col
                        self.IF_col_alive[layer_idx] = [1] * num_col
                        self.num_pruned_col[layer_idx] = 0
                        self.pruned_ratio[layer_idx] = 0
                        self.IF_layer_finished[layer_idx] = 0
                        self.history_score[layer_idx] = [0] * num_col
                    
                    """ ## IF All Conv layers Pruning Finished """
                    if all(self.IF_layer_finished.values()):
                        self.IF_prune_finished = True

                    if self.IF_layer_finished[layer_idx] == 1:
                        m.weight.grad.data.mul_(self.masks[layer_idx])
                        m.weight.data.mul_(self.masks[layer_idx])
                        continue


                    num_pruned_col_ = self.num_pruned_col[layer_idx]
                    num_col_ = num_col - num_pruned_col_
                    num_col_to_prune_ = math.ceil(num_col * self.PR) - num_pruned_col_
                
                    if num_col_to_prune_ <= 0:
                        print("BUG: num_col_to_prune_ = ", num_col_to_prune_)
                    
                    """ ## start pruning """
                    if step_ % self.prune_interval == 0 and num_col_to_prune_ > 0:
                    
                        """ ### Sort 01: sort by L1-norm """
                        col_score = {}
                        col_score_first = []
                        col_score_second = []
                        weight_data = np.array(m.weight.data.cpu()).reshape(m.weight.data.shape[0], -1)
                        for j in range(num_col):
                            col_score_first.append(j)
                            col_score_second.append(np.sum(np.fabs(weight_data[:, j])))
                    
                        col_score = dict(zip(col_score_first, col_score_second))
                        col_score_rank = sorted(col_score.items(), key = lambda k: k[1])
                        #print("col_score", col_score)
                        #print("col_score_rank", col_score_rank)
                    
                        # Make new criteria, i.e. history_rank, by rank
                        n = step_ + 1                                   # No.n iter (n starts from 1)
                        for rk in range(num_col):
                            col_of_rank_rk = col_score_rank[rk][0]
                            self.history_score[layer_idx][col_of_rank_rk] = ((n - 1) * self.history_score[layer_idx][col_of_rank_rk] + rk) / n
                        #print("history_score", history_rank)
                    
                        """ ### Sort 02: sort by histroy_rank """
                        col_hrank = {}                                  # the history_rank of each column, history_rank is like the new score
                        col_hrank_first = []
                        col_hrank_second = []
                        for j in range(num_col):
                            col_hrank_first.append(j)
                            col_hrank_second.append(self.history_score[layer_idx][j])
                        col_hrank = dict(zip(col_score_first, col_score_second))
                        col_hist_rank = sorted(col_hrank.items(), key = lambda k: k[1])
                        #print("col_hist_rank", col_hist_rank)

                        """ ### 03: Punishment """
                        for i in range(num_col_):
                            col_of_rank_i = col_hist_rank[i + num_pruned_col_][0]   # Note the real rank is i + num_pruned_col_
                            #Delta = self.punish_scheme1(i, num_col_to_prune_)
                            Delta = self.punish_func(i, num_col_, num_col_to_prune_)
                            self.Reg[layer_idx][col_of_rank_i] = max(self.Reg[layer_idx][col_of_rank_i] + Delta, 0)

                            if self.Reg[layer_idx][col_of_rank_i] >= self.Target_reg:
                                self.IF_col_alive[layer_idx][col_of_rank_i] = 0
                                self.num_pruned_col[layer_idx] += 1
                                self.pruned_ratio[layer_idx] = self.num_pruned_col[layer_idx] / num_col

                                if self.pruned_ratio[layer_idx] >= self.PR:
                                    self.IF_layer_finished[layer_idx] = 1
                           
                    
                        """ ### 04: Apply reg to Conv weights """
                        Reg_temp = np.array(self.Reg[layer_idx]).reshape(-1, m.weight.data.shape[1], m.weight.data.shape[2], m.weight.data.shape[3])
                        Reg_new = t.from_numpy(Reg_temp).float().cuda()
                        #print("Reg_new------", Reg_new)
                        m.weight.grad.data.add_(Reg_new * m.weight.data)            # use L2

                    """ ## Mask out the gradient and weights """
                    tmp = np.array(self.IF_col_alive[layer_idx]).reshape(-1, m.weight.data.shape[1], m.weight.data.shape[2], m.weight.data.shape[3])
                    self.masks[layer_idx] = t.from_numpy(tmp).float().cuda()
                    m.weight.grad.data.mul_(self.masks[layer_idx])
                    m.weight.data.mul_(self.masks[layer_idx])

    # --> apply mask  -------------------
    def set_mask(self, model):
        conv_cnt = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                conv_cnt += 1
                layer_idx = str(conv_cnt)
                if self.IF_skip_prune and (conv_cnt in self.skip_idx):
                    continue

                m.weight.grad.data.mul_(self.masks[layer_idx])
                m.weight.data.mul_(self.masks[layer_idx])
    
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

						
""" ---------------------------------- END! -------------------------------------- """
