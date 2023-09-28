import numpy as np
import h5py

import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset


# torch dataset
class Papm_Dataset(Dataset):
    def __init__(self, data, phy=None):
        self.data = torch.from_numpy(data).float()
        if phy is not None:
            self.phy = torch.from_numpy(phy).float()
        else:
            self.phy = torch.zeros([self.data.shape[0], 1]).float()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.phy[idx]


# load data
def data_load(args):
    suffix = args.file_link[-3:]
    assert suffix == '.h5'
    f = h5py.File(args.file_link, 'r')
    # data_all = np.transpose(f['u'], (0, 3, 1, 2)).astype(np.float32)
    items = []
    for k in f.keys():
        items.append(str(k))
    print('data items:', items)
    data_u = np.array(f['u']).astype(np.float32)
    data_u = data_u[:,:,np.newaxis,...]
    data_v = np.array(f['v']).astype(np.float32)
    data_v = data_v[:,:,np.newaxis,...]
    if 'p' in f.keys():
        data_p = np.array(f['p']).astype(np.float32)
        data_p = data_p[:,:,np.newaxis,...]
        data_all = np.concatenate([data_u,data_v,data_p],axis=2)
    else:
        data_all = np.concatenate([data_u,data_v],axis=2)

    data_index = None
    if 'Re' in f.keys():
        data_index = np.array(f['Re']).astype(np.float32)

    print('data shape:', data_all.shape)
    if data_index is not None:
        data_index = data_index.reshape([data_index.shape[0], 1])

    N = data_all.shape[0]
    row_rand_array = np.arange(N)

    # shuffle dataset
    if args.shuffle:
        seed = args.seed
        np.random.seed(seed)
        np.random.shuffle(row_rand_array)

    data_new = data_all

    # split——>[0.7, 0.1, 0.2]
    X_train = data_new[row_rand_array[:int(0.7*N)]]
    X_val = data_new[row_rand_array[int(0.7*N):int(0.8*N)]]
    X_test = data_new[row_rand_array[int(0.8*N):int(1*N)]]
    
    return X_train, X_val, X_test, data_index
