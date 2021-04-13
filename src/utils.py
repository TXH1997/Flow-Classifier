import os
import random
import time

import numpy as np
import pyhocon
import torch


def read_conf(path, name):
    if os.path.exists(path):
        conf = pyhocon.ConfigFactory.parse_file(path)[name]
    else:
        conf = None
        print('Unrecognized language')
        exit(0)
    print('Configuration: {}'.format(name))
    return {'name': name, 'conf': conf}


def get_device(gpu_id):
    gpu = 'cuda:' + str(gpu_id)
    if torch.cuda.is_available():
        device = torch.device(gpu)
        print('Running on GPU: ' + str(gpu_id))
    else:
        device = torch.device('cpu')
        torch.set_num_threads(6)
        print('Running on CPU')
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            torch.cuda.manual_seed(seed)
        elif torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_timestamp():
    return time.strftime('%y-%m-%d-%H:%M:%S', time.localtime(time.time()))


def split_avg(li, n):
    splitted = []
    li_len = len(li)
    assert li_len % n == 0
    size = int(li_len / n)
    for i in range(0, li_len, size):
        splitted.append(li[i:i + size])
    return splitted


def match(li1, li2):
    assert len(li1) == len(li2)
    match_num = 0
    for i in range(len(li1)):
        if li1[i] == li2[i]:
            match_num += 1
    return match_num
