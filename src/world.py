'''
Created on Oct 4, 2020

@author: Yinfeng Li
'''
import os
from os.path import join
import time
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()
exp_path0 = '../exp/'
if not os.path.exists(exp_path0):
    os.mkdir(exp_path0)
exp_path = exp_path0 + args.model +'/'
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

save_dir = exp_path+'{}-emb{}-layer{}-lamda{}-{}'.format(args.dataset, args.emb_size, args.layer, args.decay, time.strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
else:
    os.remove(os.path.join(save_dir, 'log.txt'))
print('Experiment dir : {}'.format(save_dir))


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

dataname = args.dataset
gpu_id = args.gpu_id
ROOT_PATH = "../"   # set your root path at here!
CODE_PATH = join(ROOT_PATH, 'src')
DATA_PATH = join(join(ROOT_PATH, 'data'), dataname)

FILE_PATH = join(CODE_PATH, 'checkpoints')

if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)

# construct config
config = {}
config['batch_size'] = args.batch_size
config['test_s_batch_size'] = args.testbatch
config['emb_size'] = args.emb_size
config['layer']= args.layer
config['use_drop'] = args.use_drop
config['droprate']  = args.droprate
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['gamma'] = args.gamma
config['pretrain'] = args.pretrain
use_trans = args.use_trans
use_acf = args.use_acf
earlystop = args.earlystop

if dataname == 'Telecom':
    config['num_users'] = 10099
    config['num_locations'] = 2462
    config['num_times'] = 48
    config['num_activities'] = 1751
elif dataname == 'TalkingData':
    config['num_users'] = 4068
    config['num_locations'] = 1303
    config['num_times'] = 48
    config['num_activities'] = 2310
elif dataname == '4SQ':
    config['num_users'] = 10543
    config['num_locations'] = 1012
    config['num_times'] = 31
    config['num_activities'] = 3960
elif dataname == 'TWEET':
    config['num_users'] = 10414
    config['num_locations'] = 2080
    config['num_times'] = 27
    config['num_activities'] = 5960

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
seed = args.seed
modelname = args.model

TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)

def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")