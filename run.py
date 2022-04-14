import numpy as np
import os
import pandas as pd
import subprocess
import json
import gzip
import pickle
import smtplib
import numpy as np
import random
import math
import time
import subprocess
import json
import itertools
import sys
import re

config={}
exp_comment = 'cabs'
config['NAME'] = exp_comment

dataset_basedir = os.path.dirname(os.path.realpath(__file__)) + "/data/"
config['data_loc'] = dataset_basedir+ 'CABS_SF.npy'
config['n'] = len(np.load(config['data_loc']))
config['q_w_loc'] = config['data_loc']

# disable user-level privacy analsys for secondary datasets
config['base_ulvl'] = 1
config['base_ulvl_size'] = config['n']
# decides the slices the dataset will fall into in the temporal dimension
config['squeeze'] = 24

print('Datafile:', config['data_loc'])


# Query testing configuration
config['test_query_loc'] = config['data_loc']  # path to test queries, same as data if we consider query locations that follow data distribution (only accessed at test time)
#query size is uniform between min_test_query_size and max_test_query_size
# Spatial and temporal ranges of evaluated queries
R_s = [1]
R_t = [1] 

config['MAX_VAL'] = 10  # normalization constant, so that datapoins are normalized to be between -config['MAX_VAL']/2 and  +config['MAX_VAL']/2
config['test_size'] = 2000  # number of test queries to ask

# model hyper-parameters
config['lr']=0.0003

config['training_batch_size']=150000
config['EPOCHS'] = 1500
config['out_dim'] = 1
config['in_dim'] = 3
config['random_seed'] = np.random.randint(1)
config["py_loc"] = os.path.dirname(os.path.realpath(__file__))+'/'
config['enable_ulevel'] = 0 # Not avaialable for secondary datasets

config['debiasing_c_'] = 1  # at 1 amplifcation factor defaults to N/n

config['temporal_stacking'] = 'TS_VQCONV' # TS0 trains the SNH model
config['enable_hostspot_query'] = 0
config['enable_forecasting_query'] = 0
# system parameters
config['base_granularity'] = 576
config['rho'] = -1 # if base granularity is set rho is overwritten 
# config['qs_aug'] = 'multi' #or 'none'
config['qs_aug'] = 'multi' #or 'none'


config['vae_hidden_dim'] = 128
config['num_snaps'] = -1 # controls how many of the slices will be used to train the conv type vaes. Also selectively picks queries to evaluate from the selection range.
config['vq_batchsize'] = 4
config['num_snaps'] = -1
config['debiasing_c_']  = 5e-5
settings = []

# evaluate each epsilon and associated range
for eps in [0.2]:
    setting = {"eps":eps} 
    for element in zip(R_s, R_t):
        setting['int_query_size']= list([element[0],element[0],element[1]])
        settings.append(setting.copy())

print(exp_comment, config['n'], 'len(settings)', len(settings))

max_process = 1
procs = []

for setting in settings:
    print(setting)
    name = config['NAME']

    for k, v in setting.items():
        config[k]=v
        if k == 'int_query_size':
            name += '_'+str('_'.join(str(e) for e in v))
        else:
            name += '_'+str(v)

    os.makedirs('tests/'+name, exist_ok=True)
    with open('tests/'+name+'/conf.json', 'w') as f:
        json.dump(config, f)

    node = os.uname()[1]
    gpu_asssignment = ''
    if len(sys.argv) > 1:
        gpu_asssignment = 'CUDA_VISIBLE_DEVICES='+str(sys.argv[1])

    # command = 'cd tests/'+ name + ' && XLA_PYTHON_CLIENT_PREALLOCATE=false '+ gpu_asssignment+' python -u '+config["py_loc"]+'fit_base_JAX.py > out.txt'
    command = 'cd tests/' + name + ' && XLA_PYTHON_CLIENT_PREALLOCATE=false '+ gpu_asssignment+' python -u ' + config["py_loc"] + 'fit_base_JAX.py'
    p1 = subprocess.Popen(command, shell=True) 
    procs.append(p1)
    print(len(procs))
    while len(procs) == max_process:
        for i, p in enumerate(procs):
            poll = p.poll()
            if poll is not None:
                del procs[i]
        time.sleep(1)
        #procs=[]
    exit(0)

while len(procs) != 0:
    for i, p in enumerate(procs):
        poll = p.poll()
        if poll is not None:
            del procs[i]
    time.sleep(1)
