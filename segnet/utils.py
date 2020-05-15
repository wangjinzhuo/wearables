'''Some helper functions for PyTorch, including:
    - make_dataloader: prepare DataLoader instance for trainval and test.
    - make_seq_dataloader: prepare DataLoader instance with multiple epoches for trainval and test.
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import random
import numpy as np

from scipy.io import loadmat
from torch.utils.data import Dataset, TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.init as init

def make_seq_loader(loader, seq_len, stride):
    # input:  loader of size [#n, 1, #dim]
    # reture: loader of size [#n, seq_len, #dim]

    x, y   = loader.dataset.tensors[0], loader.dataset.tensors[1]
    idx    = gen_seq(x.shape[0], seq_len, stride)
    xx, yy = [x[i:i+seq_len, :, :] for i in idx], [y[i:i+seq_len, :, :] for i in idx]
    xx     = [x.reshape(-1, x.shape[0]*x.shape[2]) for x in xx]
    xx, yy = [x.unsqueeze(0) for x in xx], [y.unsqueeze(0) for y in yy]
    xx, yy = torch.cat(xx), torch.cat(yy)
    dataset = TensorDataset(xx, yy)
    loader  = DataLoader(dataset, batch_size=loader.batch_size)
    return loader

def gen_tr_val_test_dataloader(dataset_dir, split, seq_len=128, stride=32, batch_size=128, shuffle=True, num_workers=0):
    # split: a list with 3 elements like [0.6, 0.3, 0.1]
    # if l = 100, tr=[0:60], val=[61:90], te=[91:]
    tr_files, val_files, te_files = [], [], []
    for r, d, f in os.walk(dataset_dir):
        n = len(f)
        l = list(range(n))
        random.shuffle(l)
        s1 = math.ceil(n*split[0])
        s2 = math.floor(n - n*split[2])
        tr_files = [f[i] for i in l[:s1]]
        val_files = [f[i] for i in l[s1+1:s2]]
        te_files = [f[i] for i in l[s2+1:]]

    print(tr_files, val_files, te_files)
    tr, bin_tr = make_seq_dataloader(dataset_dir, tr_files, seq_len=128, stride=32, batch_size=128, shuffle=True, num_workers=0)
    val, bin_val = make_seq_dataloader(dataset_dir, val_files, seq_len=128, stride=32, batch_size=128, shuffle=True, num_workers=0)
    te, bin_te = make_seq_dataloader(dataset_dir, te_files, seq_len=128, stride=32, batch_size=128, shuffle=True, num_workers=0)
    return bin_tr, bin_val, bin_te, tr, val, te

# generate train - validation - test
def make_dataloader(dataset_dir, files, batch_size=128, shuffle=True, num_workers=0):
    x, y = [], []
    for f in files:
        mat = loadmat(dataset_dir + f)
        data = mat['data'][:,:,1]
        data = data.reshape(data.shape[0], 1, data.shape[1])
        #print(data.shape)
        label = mat['labels']
        x.append(data)
        y.append(label)
    x, y = tuple(x), tuple(y)
    x, y = np.vstack(x), np.vstack(y)
    torch_x, torch_y = torch.from_numpy(x), torch.from_numpy(y)
    dataset = TensorDataset(torch_x, torch_y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    return dataloader

def gen_seq(n, seq_len, stride):
    res = []
    for i in range(0, n, stride):
        if i + seq_len <= n:
            res.append(i)
    return res

def convert_y_to_biny(y):
    ret = []
    for i in range(y.size()[0]):
        tensor = y[i]
        bin_ = []
        for j in range(tensor.size()[0]):
            if (tensor[j]==1).nonzero().item() == (tensor[j-1]==1).nonzero().item():
                bin_.append(0)
            else:
                bin_.append(1)
        bin_[0] = 0
        ret.append(bin_)
    # convert ret from list to torch.tensor
    ans = torch.rand(y.size()[0], y.size()[1])
    for i in range(y.size()[0]):
        for j in range(y.size()[1]):
            ans[i][j] = ret[i][j]
    return ans

class SleepDataset(Dataset):
    r"""Dataset wrapping tensors.
    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, feat, label, seq_len, feat_channel, data_feat_dim):
        assert len(feat) == len(label)
        self.feat = feat
        self.label = label
        self.seq_len = seq_len
        self.num_examples = (len(feat) - 1) // seq_len
        self.feat_channel = feat_channel
        self.example_indices = list(range(self.num_examples))
        self.data_feat_dim = data_feat_dim

    def __getitem__(self, index):
        start_idx = self.example_indices[index] * self.seq_len
        feat = self.feat[start_idx : start_idx+self.seq_len]
        if feat.shape[1] != self.data_feat_dim:
            if feat.shape[1] > self.data_feat_dim: # sample 6000 -> 3000
                sample_idx = [2*idx for idx in range(self.data_feat_dim)]
                feat = feat[:, sample_idx, :]
            else:
                assert False
        label = self.label[start_idx : start_idx+self.seq_len]
        torch_feat = torch.from_numpy(feat).reshape(self.feat_channel, self.seq_len*self.data_feat_dim)
        return (torch_feat, torch.from_numpy(label))

    def __len__(self):
        return self.num_examples


def make_feat_seq_loader(dataset_dir, batch_size, seq_len, data_feat_dim, feat_idx_list=[0], shuffle=True, num_workers=0):
    mat_files = os.listdir(dataset_dir)

    mat_list = []
    for f in mat_files:
        print("=== load ",f)
        if not f.endswith(".mat"):
            continue
        mat = loadmat(os.path.join(dataset_dir, f))
        mat_list.append(mat)

    corpus_feat = [mat['data'][:,:,feat_idx_list] for mat in mat_list]
    corpus_feat = np.concatenate(corpus_feat, axis=0)
    corpus_label = [np.argmax(mat['labels'], axis=1) for mat in mat_list]
    corpus_label = np.concatenate(corpus_label, axis=0)

    dataset = SleepDataset(corpus_feat, corpus_label, seq_len, len(feat_idx_list), data_feat_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def make_seq_dataloader(dataset_dir, files, seq_len=128, stride=32, batch_size=128, shuffle=True, num_workers=0):
    # x: [#n, 128, #dim]   y: [#n, 128]
    x, y = [], []
    for f in files:
        mat = loadmat(dataset_dir + f)
        data = mat['data'][:,:,0]
        idx = gen_seq(data.shape[0], seq_len, stride)
        label = mat['labels']
        for i in idx:
            x.append(data[i:i+seq_len,:])
            y.append(label[i:i+seq_len, :])
    numpy_x = np.random.rand(len(x), x[0].shape[0], x[0].shape[1])
    numpy_y = np.random.rand(len(y), y[0].shape[0], y[0].shape[1])
    for i in range(len(x)):
        numpy_x[i, :, :] = x[i]
    for i in range(len(y)):
        numpy_y[i, :, :] = y[i]
    torch_x, torch_y = torch.from_numpy(numpy_x), torch.from_numpy(numpy_y)

    # calculate binary y
    bin_torch_y = convert_y_to_biny(torch_y)
    bin_dataset = TensorDataset(torch_x, bin_torch_y)
    bin_dataloader = DataLoader(bin_dataset, batch_size=128, shuffle=True, num_workers=0)

    dataset = TensorDataset(torch_x, torch_y)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)
    #return dataloader
    return dataloader, bin_dataloader

def dice_loss(pred, target):
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)

    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
