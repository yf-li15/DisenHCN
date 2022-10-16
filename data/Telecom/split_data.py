'''
Created on Nov 13, 2020

@author: Yinfeng Li
'''
import numpy as np
import scipy.io as sio
import math
import os
import scipy as sp
from scipy.sparse import hstack, vstack
import scipy.sparse.linalg
import json
import random
from numpy import *

def set_seed(seed):
    np.random.seed(seed)   
    random.seed(seed)

set_seed(2020)  
mat1 = np.load('telecom.npz')
R_train = mat1['norm_data']

# source data:0User, 1Base, 2Time, 3App, 4Value

user_train_set = list(set(list(R_train[:,0])))
base_train_set = list(set(list(R_train[:,1])))
time_train_set = list(set(list(R_train[:,2])))
app_train_set = list(set(list(R_train[:,3])))

userN = len(user_train_set)
baseN = len(base_train_set)
timeN = len(time_train_set)
appN = len(app_train_set)
N = len(R_train)



print(f"{N} interactions for data")
print(f"Sparsity : {N / (userN*baseN*timeN*appN)}")
print(R_train.shape)
print('UserN:{} BaseN:{} TimeN:{} AppN:{}'.format(userN,baseN,timeN,appN))


# train, val, test:8:1:1
Index = np.arange(N)
np.random.shuffle(Index)
train_len = int(N*0.8)
test_len = int(N*0.1)
val_len = N - train_len - test_len
train_idx = Index[:train_len]
val_idx = Index[train_len: N-test_len]
test_idx = Index[N-test_len:]

print(train_len, val_len, test_len)

# split
train_data = np.zeros((train_len, 5))
val_data = np.zeros((val_len, 5))
test_data = np.zeros((test_len, 5))
for i in range(train_len):
    train_data[i, :] = R_train[train_idx[i], :]

for i in range(val_len):
    val_data[i, :] = R_train[val_idx[i], :]

for i in range(test_len):
    test_data[i, :] = R_train[test_idx[i], :]

np.savez('./train_data.npz', train_data=train_data)
np.savez('./val_data.npz', val_data=val_data)
np.savez('./test_data.npz', test_data=test_data)
