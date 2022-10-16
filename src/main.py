'''
Created on Oct 4, 2020

@author: Yinfeng Li
'''
import sys
import os
import world
import utils
import torch
import tqdm
import numpy as np
import logging
import time
import multiprocessing
from torch import nn
from dataloader import Loader
from torch.utils.data import DataLoader
from model import DisenHCN

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format)
fh = logging.FileHandler(os.path.join(world.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

weight_file = os.path.join(world.FILE_PATH, f"checkpoint-{world.modelname}-{world.config['layer']}-{world.config['emb_size']}.pt")
# ==============================
utils.set_seed(world.seed)
logging.info(">>SEED:{}".format(world.seed))
# ==============================
logging.info('===========config================')
logging.info("model:{}".format(world.modelname))
logging.info("dataset:{}".format(world.dataname))
logging.info("gpu:{}".format(world.gpu_id))
logging.info("use_trans:{}".format(world.use_trans))
logging.info("use_acf:{}".format(world.use_acf))
logging.info(world.config)
logging.info("LOAD:{}".format(world.LOAD))
logging.info("Weight path:{}".format(weight_file))
logging.info("Test Topks{}:".format(world.topks))
logging.info('===========end===================')

CORES = multiprocessing.cpu_count() // 2


class EarlyStopper(object):

    def __init__(self, num_trials, num_metric):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_result = {}
        #'precision': array([0.01374143]), 'recall': array([0.06561206]), 'ndcg': array([0.03310027]), 'auc': 0.5
        self.best_result = {'precision': np.zeros(num_metric),
                            'recall': np.zeros(num_metric),
                            'ndcg': np.zeros(num_metric),
                            'auc': 0}
        self.num_metric = num_metric

    def is_continuable(self, results):
        flag = [True]*(3*self.num_metric+1)
        i = 0
        for key in results.keys():
            if key == 'auc':
                if results[key] > self.best_result[key]:
                    self.best_result[key] = results[key]
                    flag[i] = True
                else:
                    flag[i] = False
                i += 1
            else:
                for j in range(self.num_metric):
                    if results[key][j] > self.best_result[key][j]:
                        self.best_result[key][j] = results[key][j]
                        flag[i] = True
                    else:
                        flag[i] = False
                    i += 1

        is_continue = False
        for i in range(len(flag)):
            is_continue = is_continue or flag[i]

        if is_continue:
            self.trial_counter = 0
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


def train(dataset, recommend_model, loss_class, epoch, neg_k=1):
    Recmodel = recommend_model
    Recmodel.train()
    criterion = loss_class
    S = utils.UniformSample_original(dataset) # S:[user, base, time, pos, neg, val]
    users = torch.Tensor(S[:, 0]).long()
    locations = torch.Tensor(S[:, 1]).long()
    times = torch.Tensor(S[:, 2]).long()
    pos = torch.Tensor(S[:, 3]).long()
    neg = torch.Tensor(S[:, 4]).long()
    val = torch.Tensor(S[:, 5])

    users = users.to(world.device)
    locations = locations.to(world.device)
    times = times.to(world.device)
    pos = pos.to(world.device)
    neg = neg.to(world.device)
    val = val.to(world.device)
    users, locations, times, pos, neg, val = utils.shuffle(users, locations, times, pos, neg, val)
    total_batch = len(users) // world.config['batch_size'] + 1
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_locations,
          batch_times,
          batch_pos,
          batch_neg,
          batch_val)) in enumerate(utils.minibatch(users,
                                                   locations,
                                                   times,
                                                   pos,
                                                   neg,
                                                   val,
                                                   batch_size=world.config['batch_size'])):
        cri = criterion.stageOne(batch_users, batch_locations, batch_times, batch_pos, batch_neg, batch_val)
        aver_loss += cri
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def test(dataset, Recmodel, epoch, multicore=0):
    s_batch_size = world.config['test_s_batch_size']
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict # key:scene(u,l,t)
    scenePos = dataset.trainDict
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        scenes = list(testDict.keys())
        scenes_index = np.arange(len(scenes))
        try:
            assert s_batch_size <= len(scenes) / 10
        except AssertionError:
            logging.info(f"test_s_batch_size is too big for this dataset, try a small one {len(scenes) // 10}")
        
        users_list = []
        rating_list = []
        groundTrue_list = []
        auc_record = []
        # ratings = []
        total_batch = len(scenes) // s_batch_size + 1
        for batch_index in utils.minibatch(scenes_index, batch_size=s_batch_size):
            batch_S = []
            groundTrue = []
            allPos = []
            for idx in batch_index:
                u, b, t = scenes[idx]
                groundTrue.append(testDict[scenes[idx]])
                batch_S.append([u, b, t])
                if scenes[idx] not in scenePos:
                    continue
                allPos.append(scenePos[scenes[idx]])
            batch_S = np.array(batch_S)
            batch_users_gpu = torch.Tensor(batch_S[:,0]).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
            batch_locations_gpu = torch.Tensor(batch_S[:,1]).long()
            batch_locations_gpu = batch_locations_gpu.to(world.device)
            batch_times_gpu = torch.Tensor(batch_S[:,2]).long()
            batch_times_gpu = batch_times_gpu.to(world.device)

            rating = Recmodel.getRating(batch_users_gpu, batch_locations_gpu, batch_times_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10) # exclude train pos items.
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            aucs = [ 
                    utils.AUC(rating[i],
                              dataset, 
                              test_data) for i, test_data in enumerate(groundTrue)
                ]
            auc_record.extend(aucs)
            del rating
            users_list.append(batch_S[:,0])
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x))
        scale = float(s_batch_size/len(scenes))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(scenes))
        results['precision'] /= float(len(scenes))
        results['ndcg'] /= float(len(scenes))
        results['auc'] = np.mean(auc_record)
        
        if multicore == 1:
            pool.close()
        logging.info(results)
        return results

def get_model(name):
    if name == 'DisenHCN':
        return DisenHCN(world.config)
    else:
        print("Please check the model name!")
        
    

dataset = Loader()
model = get_model(world.modelname)
model = model.to(world.device)
criterion = utils.BPRLoss(model, world.config)

Neg_k = 1

best_recall = 0 
best_ndcg = 0
if world.earlystop:
    print("Using early stop!")
early_stopper = EarlyStopper(num_trials=5, num_metric=len(world.topks))

for epoch in range(world.TRAIN_epochs):
    logging.info('======================')
    logging.info(f'EPOCH:{epoch}')
    logging.info(f'Time:{time.strftime("%Y/%m/%d %H:%M:%S")}')
    
    if epoch %1 == 0:
        logging.info("[TEST]")
        
        results = test(dataset, model, epoch, world.config['multicore'])
        
        if (not early_stopper.is_continuable(results)) and epoch > 40:
            if world.earlystop:
                logging.info(f'Early Stop @ epoch{epoch}!')
                logging.info(f'best results: {early_stopper.best_result}')
                break
        logging.info(f'best results: {early_stopper.best_result}')
    
    output_information = train(dataset, model, criterion, epoch, neg_k=Neg_k)
    
    logging.info(f'[saved][{output_information}]')
    #torch.save(Recmodel.state_dict(), weight_file)

