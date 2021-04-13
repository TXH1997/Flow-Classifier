import json
import os
import random
import re

import numpy as np

import src.utils as utils


class DataGenerator:
    def __init__(self, data_dir, batch_size, ext):
        self.n_fold = 10
        self.batch_size = batch_size
        self.ext = ext
        if self.ext:
            self.channels = 4
            self.cache_name = 'cache_ext.json'
        else:
            self.channels = 2
            self.cache_name = 'cache.json'
        self.data_dir = data_dir
        self.data = self.load_data()
        self.test_pos = 0
        self.dev_pos = self.get_dev_pos()
        self.round_end = False
        self.test_data = self.data[self.test_pos]
        self.train_data = self.get_train_data()
        self.dev_data = self.data[self.dev_pos]
        assert len(self.train_data) % self.batch_size == 0
        assert len(self.test_data) % self.batch_size == 0

    def train_num(self):
        return len(self.train_data)

    def test_num(self):
        return len(self.test_data)

    def dev_num(self):
        return len(self.dev_data)

    def get_dev_pos(self):
        dev_pos = random.randrange(0, self.n_fold)
        while dev_pos == self.test_pos:
            dev_pos = random.randrange(0, self.n_fold)
        return dev_pos

    def new_round(self):
        self.round_end = False

    def generate_train_data(self):
        random.shuffle(self.train_data)
        for i in range(0, len(self.train_data), self.batch_size):
            td = self.train_data[i: i + self.batch_size]
            max_len = max([len(x[1]) for x in td])
            examples = np.zeros([self.batch_size, max_len, self.channels], dtype=np.float32)
            labels = np.zeros([self.batch_size], dtype=np.int64)
            for j, e in enumerate(td):
                examples[j][:len(e[1])] = e[1]
                labels[j] = e[0]
            yield examples, labels

    def generate_test_data(self):
        for i in range(0, len(self.test_data), self.batch_size):
            td = self.test_data[i: i + self.batch_size]
            max_len = max([len(x[1]) for x in td])
            examples = np.zeros([self.batch_size, max_len, self.channels], dtype=np.float32)
            labels = np.zeros([self.batch_size], dtype=np.int64)
            for j, e in enumerate(td):
                examples[j][:len(e[1])] = e[1]
                labels[j] = e[0]
            yield examples, labels

    def generate_dev_data(self):
        for i in range(0, len(self.dev_data), self.batch_size):
            dd = self.dev_data[i: i + self.batch_size]
            max_len = max([len(x[1]) for x in dd])
            examples = np.zeros([self.batch_size, max_len, self.channels], dtype=np.float32)
            labels = np.zeros([self.batch_size], dtype=np.int64)
            for j, e in enumerate(dd):
                examples[j][:len(e[1])] = e[1]
                labels[j] = e[0]
            yield examples, labels

    def get_train_data(self):
        train_set = []
        for i in range(len(self.data)):
            if not i == self.test_pos:
                train_set.extend(self.data[i])
        return train_set

    def switch_train_test(self):
        self.test_pos += 1
        if self.test_pos == self.n_fold:
            self.round_end = True
        self.test_pos = self.test_pos % self.n_fold
        self.dev_pos = self.get_dev_pos()
        self.test_data = self.data[self.test_pos]
        self.train_data = self.get_train_data()
        self.dev_data = self.data[self.dev_pos]

    def load_data(self):
        cache_path = self.data_dir + self.cache_name
        if not os.path.exists(cache_path):
            self.cache_data()
        with open(cache_path, 'r', encoding='utf-8') as fin:
            print('Loading cached dataset from {}'.format(cache_path))
            data = json.load(fin)
            print('Done')
        random.shuffle(data)
        return data

    def cache_data(self):
        print('Caching tensorized data...')
        data = []
        pattern = re.compile(r'(\d+?)-(\d+?)')
        for home, dirs, files in os.walk(self.data_dir):
            for f in files:
                if not pattern.match(f):
                    continue
                name = f.split('-')
                e_cls = int(name[0])
                with open(home + f, 'r', encoding='utf-8') as fin:
                    vec = []
                    for i, line in enumerate(fin.readlines()):
                        token = list(map(float, line.split()))
                        if self.ext:
                            if i != 0:
                                token.append(token[0] - vec[-1][0])
                                token.append(int(token[1] == vec[-1][1]))
                            else:
                                token.extend([0, 0])
                        vec.append(token)
                data.append((e_cls, vec))
        cache_path = self.data_dir + self.cache_name
        with open(cache_path, 'w', encoding='utf-8') as fout:
            random.shuffle(data)
            data = utils.split_avg(data, self.n_fold)
            json.dump(data, fout, indent=None, separators=(',', ':'))
        print('Done')
