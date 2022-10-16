'''
Created on Oct 4, 2020

@author: Yinfeng Li
'''
import sys
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset, Loader
from time import time
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
import json
import math

class BPRLoss:
    def __init__(self, 
                 recmodel : PairWiseModel, 
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        
    def stageOne(self, users, bases, times, pos, neg, val):
        """
        users, bases, times, positive app sample, negetive app sample, value
        """
        if world.modelname == 'MetaHGNN':
            loss, reg_loss = self.model.bpr_loss(users, bases, times, pos, neg, val)
        else:
            loss, reg_loss = self.model.bpr_loss(users, bases, times, pos, neg)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.cpu().item()


class PredLoss:
    def __init__(self, 
                 recmodel : PairWiseModel, 
                 config : dict):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        #self.cri = nn.KLDivLoss(size_average=False)
        self.cri = torch.nn.MSELoss()
        
    def stageOne(self, users, bases, times, pos, neg, val):
        """
        users, bases, times, positive app sample, negetive app sample, value
        """
        #loss_dep, reg_loss, pred = self.model.pre_loss(users, bases, times, pos, neg)
        #reg_loss = reg_loss*self.weight_decay
        pred = self.model(users, bases, times, pos)
        loss = self.cri(pred, val)
        if world.modelname in ['UCLAF', 'MCTF']:
            loss +=  1e-2 * self.model.reg_loss()
        elif world.modelname in ['MCTF']:
            loss +=  1e-3 * self.model.reg_loss()
        elif world.modelname in ['WDGTC']:
            loss +=  1e-2 * self.model.get_trace()
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.cpu().item()



# ====================== samplers==========================
def UniformSample_original(dataset):
    """
    the original impliment of BPR Sampling
    :return:
        np.array
    """
    dataset : BasicDataset
    rec_num = dataset.trainDataSize
    scenes_index = np.random.randint(0, dataset.n_scenes, rec_num)  #scenes:(user, base, time), sample scenes_index
    allScenes = dataset.allScenes #scene list
    scenePos = dataset.trainDict
    trainValue = dataset.trainValue # dict{(u,b,t,a):val} for auxiliary task
    S = []
    for scene_idx in scenes_index:
        user, base, time = allScenes[scene_idx]
        if (user,base,time) not in scenePos:
            continue
        posForScene = scenePos[(user,base,time)]
        posindex = np.random.randint(0, len(posForScene)) # sample one postive sample
        pos = posForScene[posindex]
        #val = trainValue[(user, base, time, pos)]
        val = 1 # postive sample
        # sample negative sample
        while True:
            neg = np.random.randint(0, dataset.n_apps)
            if neg in posForScene:
                continue
            else:
                break
        S.append([user, base, time, pos, neg, val])
    return np.array(S)

# ===================end samplers==========================



# =====================utils====================================
def set_seed(seed):
    np.random.seed(seed)   
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def construct_tabel(userlist, baselist, timelist, applist, targets, predicts):
    # construct test tabel:(user, base, time, app, true_val, predict)
    num = len(predicts)
    tabel = np.zeros((num, 6))
    tabel[:, 0] = np.array(userlist)
    tabel[:, 1] = np.array(baselist)
    tabel[:, 2] = np.array(timelist)
    tabel[:, 3] = np.array(applist)
    tabel[:, 4] = np.array(targets)
    tabel[:, 5] = np.array(predicts)
    return tabel

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    #accuracy = np.sum(right_pred)
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]
    
    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.n_apps, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    try: 
        auc = roc_auc_score(r, test_item_scores) 
    except ValueError: 
        auc = 0 
    return auc

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================

# ====================Prediction Metrics=============================

def construct_tabel(userlist, baselist, timelist, applist, targets, predicts):
    # construct test tabel:(user, base, time, app, true_val, predict)
    num = len(predicts)
    tabel = np.zeros((num, 6))
    tabel[:, 0] = np.array(userlist)
    tabel[:, 1] = np.array(baselist)
    tabel[:, 2] = np.array(timelist)
    tabel[:, 3] = np.array(applist)
    tabel[:, 4] = np.array(targets)
    tabel[:, 5] = np.array(predicts)
    return tabel


# ====================Metrics==============================

# =========================================================
def getTOPK(Array, K):
    Array_sorted = Array[np.lexsort(-Array.T)]  # Sort in reverse order by the last column
    S1 = set(list(np.array(list(Array_sorted[0:K,0]),dtype=int)))  # TOP k item, int type
    S1_scores = list(Array_sorted[0:K,-1])  # Get the value of TOP k item
    return S1, S1_scores

def getTOPK_pre(Array, Array_true, K):
    Array_sorted = Array[np.lexsort(-Array.T)]  # Sort in reverse order by the last column
    pre_list = list(Array_sorted[0:K,0])
    pre_topk = set(list(np.array(list(Array_sorted[0:K,0]),dtype=int)))  # TOP k item, int type
    pre_scores = []
    for i in range(K):
        target_item = pre_list[i]
        ini_score = Array_true[Array_true[:,0]==target_item,1][0]
        pre_scores.append(ini_score)
    return pre_topk, pre_scores

def calDCG(target_score, N):
    val = 0
    for i in range(N):
        val += target_score[i]/math.log(float(i+2),2)
    return float(val)


def ACC_NDCG_TOPN(test_data, topK):
    """
    test_data:tabel(user, base, time, app, true_val, pre_val)
    topK: list, eg:[3, 5, 10, 20]
    return AccList, nDCGList
    """
    topk_num = len(topK)
    counter1L = [0] * topk_num
    counter2L = [0] * topk_num
    AccL = [0] * topk_num
    nDCGL = [0] * topk_num
    userSet = set(list(test_data[:,0])) # get user Set
    
    for u in userSet:
        base_scenario = test_data[test_data[:,0]==u, 1:] # get i-th user's base scenario:(base, time, app, true_val, pre_val)
        baseSet = set(list(base_scenario[:,0]))
        for b in baseSet:
            time_scenario = base_scenario[base_scenario[:,0]==b, 1:] # get j-th base's scenario of user i:(time, app, true_val, pre_val)
            timeSet = set(list(time_scenario[:,0]))
            for t in timeSet:
                app_scenario = time_scenario[time_scenario[:, 0]==t, 1:] #(app, true_val, pre_val)
                targets = np.zeros((len(app_scenario), 2)) #(app, true_val)
                predicts = np.zeros((len(app_scenario), 2)) #(app, pre_val)
                targets[:,0] = app_scenario[:, 0]
                targets[:,1] = app_scenario[:, 1]
                predicts[:,0] = app_scenario[:, 0]
                predicts[:,1] = app_scenario[:, 2]
                for n in range(topk_num):
                    K = topK[n]
                    if K < len(app_scenario):
                        counter2L[n] += 1
                        TOPKtrue,TOPKtrue_scores = getTOPK(targets, K) 
                        TOPKpre,TOPKpre_scores = getTOPK_pre(predicts, targets, K) 
                        intersect_len = len(TOPKtrue.intersection(TOPKpre))
                        if intersect_len > 0:
                            # Acc
                            counter1L[n] += 1
                            Acc_value = float(intersect_len) / float(K)
                            AccL[n] += Acc_value
                        # nDCG
                        DCG_K = calDCG(TOPKpre_scores, K)
                        IDCG_K = calDCG(TOPKtrue_scores, K)
                        if IDCG_K == 0:
                            IDCG_K = 1
                        nDCG_value = float(DCG_K)/IDCG_K
                        nDCGL[n] += nDCG_value
    accL = (np.array(AccL)/np.array(counter2L, dtype=float)).tolist()
    accL = [round(x,6) for x in accL]
    ndcgL = (np.array(nDCGL)/np.array(counter2L, dtype=float)).tolist()
    ndcgL = [round(x,6) for x in ndcgL]
    return sorted(accL), sorted(ndcgL)
# =========================================================
