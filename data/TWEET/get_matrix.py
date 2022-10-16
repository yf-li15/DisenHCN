'''
Created on Oct 4, 2020

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
from numpy import *

mat1 = np.load('TW.npz')
R_data = mat1['norm_data']

# data:0User, 1Base, 2Time, 3App

userN, baseN, timeN, appN = int(max(R_data[:,0]))+1, int(max(R_data[:,1]))+1, int(max(R_data[:,2]))+1, int(max(R_data[:,3]))+1
N = len(R_data)
print("user_num, location_num, time_num, activity_num")
print(userN, baseN, timeN, appN)
print(max(R_data[:,0]), max(R_data[:,1]), max(R_data[:,2]), max(R_data[:,3]))

print(f"{N} interactions for data")
print(f"Sparsity : {N / (userN*baseN*timeN*appN)}")
print(R_data.shape)
print('UserN:{} BaseN:{} TimeN:{} AppN:{}'.format(userN,baseN,timeN,appN))

print('Load train data...')
mat_train = np.load('train_data.npz')
R_train = mat_train['train_data']



# U-B, U-T, U-A, B-T, B-A, T-A
U_B = sp.sparse.lil_matrix((userN, baseN))
U_T = sp.sparse.lil_matrix((userN, timeN))
U_A = sp.sparse.lil_matrix((userN, appN))
B_T = sp.sparse.lil_matrix((baseN, timeN))
B_A = sp.sparse.lil_matrix((baseN, appN))
T_A = sp.sparse.lil_matrix((timeN, appN))

N = 0
for i in range(len(R_train)):
    if R_train[i,4] != 0:
        N += 1
        u_idx, b_idx, t_idx, a_idx = int(R_train[i,0]), int(R_train[i,1]), int(R_train[i,2]), int(R_train[i,3])
        U_B[u_idx, b_idx] = 1
        U_T[u_idx, t_idx] = 1
        U_A[u_idx, a_idx] = 1
        B_T[b_idx, t_idx] = 1
        B_A[b_idx, a_idx] = 1
        T_A[t_idx, a_idx] = 1


# interaction hypergraph
H_ul = U_B
H_ut = U_T
H_ua = U_A



# construct norm_vtoe(node to hyperedge in hypergraph) shape:(U, B+T+A)
def get_norm_vtoe(H):
    epsilon = 0.1 ** 10
    nodeN, edgeN = H.shape
    D_edge = np.array(H.sum(axis=0)).squeeze()
    D_e = sp.sparse.lil_matrix((edgeN, edgeN))
    for i in range(edgeN):
        D_e[i, i] = 1.0 / max(D_edge[i], epsilon)
    norm_vtoe = D_e * H.T
    return norm_vtoe

print("generating norm_vtoe matrix")
norm_vtoe_l = get_norm_vtoe(H_ul.T)
norm_vtoe_l = norm_vtoe_l.tocsr()
sp.sparse.save_npz('./norm_vtoe_l.npz', norm_vtoe_l)

norm_vtoe_t = get_norm_vtoe(H_ut.T)
norm_vtoe_t = norm_vtoe_t.tocsr()
sp.sparse.save_npz('./norm_vtoe_t.npz', norm_vtoe_t)

norm_vtoe_a = get_norm_vtoe(H_ua.T)
norm_vtoe_a = norm_vtoe_a.tocsr()
sp.sparse.save_npz('./norm_vtoe_a.npz', norm_vtoe_a)

# construct norm_etov(hyperedge to node in hypergraph) shape:(B+T+A, U)
def get_norm_etov(H):
    epsilon = 0.1 ** 10
    nodeN, edgeN = H.shape
    D_node = np.array(H.sum(axis=1)).squeeze()
    D_n = sp.sparse.lil_matrix((nodeN, nodeN))
    for i in range(nodeN):
        D_n[i, i] = 1.0 / max(D_node[i], epsilon)
    norm_etov = D_n * H
    return norm_etov

print("generating norm_etov matrix")
norm_etov_l = get_norm_etov(H_ul.T)
norm_etov_l = norm_etov_l.tocsr()
sp.sparse.save_npz('./norm_etov_l.npz', norm_etov_l)

norm_etov_t = get_norm_etov(H_ut.T)
norm_etov_t = norm_etov_t.tocsr()
sp.sparse.save_npz('./norm_etov_t.npz', norm_etov_t)

norm_etov_a = get_norm_etov(H_ua.T)
norm_etov_a = norm_etov_a.tocsr()
sp.sparse.save_npz('./norm_etov_a.npz', norm_etov_a)



def cal_norm(adj_mat):
    nodeN, edgeN = adj_mat.shape
    I = sp.sparse.lil_matrix(np.eye(nodeN, nodeN))
    I = I.tocsr()
    adj_mat = adj_mat.todok()
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.sparse.diags(d_inv)
    norm_adj = d_mat.dot(adj_mat)
    norm_adj = norm_adj.dot(d_mat)
    norm_adj = norm_adj.tocsr()
    #norm_adj = I - norm_adj
    return norm_adj
# first-order 
D_B = U_B * U_B.T
D_T = U_T * U_T.T
D_A = U_A * U_A.T
Norm_L = cal_norm(D_B)
Norm_T = cal_norm(D_T)
Norm_A = cal_norm(D_A)


D_B = D_B.toarray()
D_T = D_T.toarray()
D_A = D_A.toarray()


# second-order
D_LT = sp.sparse.lil_matrix(D_B * D_T)
D_LA = sp.sparse.lil_matrix(D_B * D_A)
D_TA = sp.sparse.lil_matrix(D_T * D_A)
Norm_LT = cal_norm(D_LT)
Norm_LA = cal_norm(D_LA)
Norm_TA = cal_norm(D_TA)


# third-order
D_LTA = D_B * D_T * D_A
D_LTA = sp.sparse.lil_matrix(D_LTA)
Norm_LTA = cal_norm(D_LTA)

sp.sparse.save_npz('./Norm_L.npz', Norm_L)
sp.sparse.save_npz('./Norm_T.npz', Norm_T)
sp.sparse.save_npz('./Norm_A.npz', Norm_A)
sp.sparse.save_npz('./Norm_LT.npz', Norm_LT)
sp.sparse.save_npz('./Norm_LA.npz', Norm_LA)
sp.sparse.save_npz('./Norm_TA.npz', Norm_TA)
sp.sparse.save_npz('./Norm_LTA.npz', Norm_LTA)