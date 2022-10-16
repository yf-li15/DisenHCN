
'''
Created on Oct 4, 2020

@author: Yinfeng Li
'''

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Activity prediction via hypergraph")
    parser.add_argument('--dataset', type=str,default="data",
                        help="[Telecom, TalkingData, 4SQ, TWEET]")
    parser.add_argument('--batch_size', type=int,default=2048,
                        help="the batch size for training procedure")
    parser.add_argument('--emb_size', type=int,default=60,
                        help="the embedding size of the model")
    parser.add_argument('--layer', type=int,default=1,
                        help="the layer num of Hypergraph")
    parser.add_argument('--lr', type=float,default=1e-3,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--gamma', type=float,default=3e-3,
                        help="the weight for Independent constraint")               
    parser.add_argument('--use_drop', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--droprate', type=float,default=0.4,
                        help="dropout rate")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--gpu_id', type=str,default="1")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=1, help='whether use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--model', type=str, default='DisenHCN', help='model name')
    parser.add_argument('--use_trans', action='store_true', default=False, help='use trans in GCN')
    parser.add_argument('--use_acf', action='store_true', default=False, help='use activate function in GCN')
    parser.add_argument('--earlystop', action='store_true', default=False, help='use earlystop')
    return parser.parse_args()