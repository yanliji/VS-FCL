# dataset_new_stage2.py
import torch
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import h5py
import os.path as osp
import sys
import os
import scipy.misc
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class NTUDataset(Dataset):
    """
    args:
        x (list): Input dataset, each element in the list is an ndarray corresponding to
        a joints matrix of a skeleton sequence sample
        y (list): Action labels
    """

    def __init__(self, x, y, z):
        self.x = x
        self.y = torch.tensor(np.array(y, dtype='int'))
        self.z = np.array(z, dtype='int')
        self.num_classes = 40

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
       return [self.x[index], int(self.y[index]), int(self.z[index]), index]
       #return [self.x[index], int(self.y[index]), index]

class CustomSampler(Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        indices = []
        l = list(range(self.data.num_classes))
        random.shuffle(l)
        for n in l:
        #for n in random.choice(range(0,self.data.num_classes)):
            index = torch.where(self.data.y == n)[0]
            indices.append(index)
        indices = torch.cat(indices, dim=0)
        return iter(indices)

    def __len__(self):
        return len(self.data)

class CustomBatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                i < len(sampler_list) - 1
                and self.sampler.data.y[idx]
                != self.sampler.data.y[sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class NTUDataLoaders(object):
    def __init__(self, dataset = 'NTU', case = 1):
        self.dataset = dataset
        self.case = case
        self.create_datasets()
        self.train_set = NTUDataset(self.train_X, self.train_Y, self.train_Z)
        self.val_set = NTUDataset(self.val_X, self.val_Y, self.val_Z)
        self.test_set = NTUDataset(self.test_X, self.test_Y, self.test_Z)
        #self.train_set = NTUDataset(self.train_X, self.train_Y)
        #self.val_set = NTUDataset(self.val_X, self.val_Y)
        #self.test_set = NTUDataset(self.test_X, self.test_Y)
        self.s1 = CustomSampler(self.train_set)
        self.s2 = CustomBatchSampler(self.s1, 64, True) # be sure it is the same as batch size
        self.s3 = CustomSampler(self.val_set)
        self.s4 = CustomBatchSampler(self.s3, 64, True)
        self.s5 = CustomSampler(self.test_set)
        self.s6 = CustomBatchSampler(self.s5, 64, True)

    def get_train_loader(self, batch_size, num_workers):
        # normal data loader
        return DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=self.collate_fn, pin_memory=True)


    def get_val_loader(self, batch_size, num_workers):
        return DataLoader(self.val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn, pin_memory=True)

    def get_test_loader(self, batch_size, num_workers):
        return DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=self.collate_fn, pin_memory=True)

    def collate_fn(self, batch):
        # revised collate_fn      for loading data, action label, view label
        x, y, z, i = zip(*batch)
        x = torch.stack([torch.from_numpy(x[i]) for i in range(len(x))], 0)
        y = torch.LongTensor(y)
        z = torch.LongTensor(z)
        i = torch.tensor(i)
        return [x, y, z, i]

    def get_train_size(self):
        return len(self.train_Y)

    def get_val_size(self):
        return len(self.val_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def create_datasets(self):
        if self.dataset =='NTU':
            if self.case == 0:
                self.metric = 'CS'
            else:
                self.metric = 'CV'
            #path = osp.join('./', 'NTU_' + self.metric + '.h5')
            #modify your data path
            path = osp.join('/home/gaoll/CMC/', 'NTU_' + self.metric + '.h5')
        if self.dataset =='UESTC':
            if self.case == 0:
                self.metric = 'CS'
            if self.case == 1:
                self.metric = 'CV'
            if self.case == 2:
                self.metric = 'CV1'
            if self.case == 3:
                self.metric = 'AV'
            #path = osp.join('./', 'UESTC_' + self.metric + '.h5')
            #modify your data path
            path = osp.join('/home/gaoll/CMC/', 'UESTC_' + self.metric + '.h5')

        f = h5py.File(path, 'r')
        self.train_X = f['x'][:]
        self.train_Y = np.argmax(f['y'][:],-1)
        self.train_Z = np.argmax(f['z'][:],-1)
        self.val_X = f['valid_x'][:]
        self.val_Y = np.argmax(f['valid_y'][:], -1)
        self.val_Z = np.argmax(f['valid_z'][:], -1)
        self.test_X = f['test_x'][:]
        self.test_Y = np.argmax(f['test_y'][:], -1)
        self.test_Z = np.argmax(f['test_z'][:], -1)
