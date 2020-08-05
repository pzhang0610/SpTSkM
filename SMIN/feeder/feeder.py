import os
import numpy as np
import random
import pickle
import torch
from torchvision import datasets, transforms
from torch.utils import data as torchData
from . import tools
import pdb


class Feeder(torchData.Dataset):
    """Feeder for skeleton-based person re-identification"""
    def __init__(self,
                 data_path,
                 label_path,
                 relabel,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 normalization=False,
                 random_shift=False,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.relabel = relabel
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.random_shift = random_shift
        self.normalization = normalization
        self.load_data(mmap)
        if normalization:
            self.get_mean_map()

    def load_data(self, mmap):
        # data format N C V T M(only one instance is involved)
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        # pdb.set_trace()

        pid_list = sorted(np.unique(self.sample_name))
        self.pid2label = {pid: label for label, pid in enumerate(pid_list)}

        # with open(self.cam_path, 'rb') as f:
        #     self.camid = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[:100]
            self.data = self.data[:100]
            self.sample_name = self.sample_name[:100]

        self.N, self.C, self.T, self.V = self.data.shape

    def get_mean_map(self):
        # pdb.set_trace()
        data = self.data
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 1, 3)).reshape((self.N * self.T, self.C * self.V)).std(axis=0).reshape((self.C, 1, self.V))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        if self.relabel:
            label = self.pid2label[self.sample_name[index]]
        else:
            label = self.label[index]
        # camid = self.camid[index]
        # processing
        if self.normalization:
            data_numpy = (data_numpy - self.mean_map) / self.std_map
        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)
        # print(data_numpy.shape)
        return data_numpy, label