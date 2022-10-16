'''
Created on Oct 4, 2020

@author: Yinfeng Li
'''

import world
import numpy as np
from torch import log
from time import time
import random
import os
import json
import math

def load_matrix(matrix_type):
    """
    matrix_type: srt  eg:'U_A'
    """
    path = os.path.join(world.DATA_PATH, matrix_type+".npz")
    data = np.load(path)
    return data[matrix_type]

def read_bases(fre_u, fre_b, fre_t, fre_a):
    """
    (user, base, time, app)
    """
    path = os.path.join(world.DATA_PATH, "hypergraph_embeddings.json")
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat_u, feat_b, feat_t, feat_a] = bases
    feat_u = np.array(feat_u)[:, 0: fre_u].astype(np.float32)
    feat_b = np.array(feat_b)[:, 0: fre_b].astype(np.float32)
    feat_t = np.array(feat_t)[:, 0: fre_t].astype(np.float32)
    feat_a = np.array(feat_a)[:, 0: fre_a].astype(np.float32)
    return [feat_u, feat_b, feat_t, feat_a]

def read_bta(fre, HGN_learn = 'zhou'):
    """
    read bta hypergraph embeddings
    HGN_learn:[Bolla, Rodriguez, zhou]
    """
    if HGN_learn == 'zhou':
        bta_name = "bta_hypergraph_embeddings.json"
    elif HGN_learn == 'Bolla':
        bta_name = "bta_bo_hypergraph_embeddings.json"
    elif HGN_learn == 'Rodriguez':
        bta_name = "bta_ro_hypergraph_embeddings.json"

    path = os.path.join(world.DATA_PATH, bta_name)
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat] = bases
    feat = np.array(feat)[:, 0: fre].astype(np.float32)
    return feat

def read_lamda(fre_u, fre_b, fre_t, fre_a):
    """
    (user, base, time, app)
    """
    path = os.path.join(world.DATA_PATH, "hypergraph_lamda.json")
    with open(path) as f:
        line = f.readline()
        bases = json.loads(line)
    f.close()
    [feat_u, feat_b, feat_t, feat_a] = bases
    feat_u = np.array(feat_u)[0: fre_u].astype(np.float32)
    feat_b = np.array(feat_b)[0: fre_b].astype(np.float32)
    feat_t = np.array(feat_t)[0: fre_t].astype(np.float32)
    feat_a = np.array(feat_a)[0: fre_a].astype(np.float32)
    return [feat_u, feat_b, feat_t, feat_a]