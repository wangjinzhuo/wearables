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
import torch.nn.functional as F

def gdl(pred, gt, class_num):
    # generalized dice loss
    # pred: bs, seq_len, class_num
    # gt  : bs, seq_len
    onehot_y = F.one_hot(gt.long(), class_num)

    intersection = torch.sum(onehot_y * pred)
    union        = torch.sum(onehot_y + pred)
    loss         = 1 - 2 * intersection / (union * class_num)

    pred = torch.argmax(pred, dim=2)
    corr  = torch.sum(torch.eq(pred.long(), gt.long())).item()
    total = torch.numel(gt)

    return loss, corr, total

def make_bin_loader(loader):
    x, y = loader.dataset.tensors[0], loader.dataset.tensors[1]
    if torch.numel(y) == y.size(0):
        bin_y = convert_class_to_bin(y)
    else:
        bin_y = [convert_class_to_bin(y[yy]) for yy in range(y.size(0))]
        bin_y = [yy.unsqueeze(0) for yy in bin_y]
        bin_y = torch.cat(bin_y)
    dataset = TensorDataset(x, bin_y)
    bin_loader = DataLoader(dataset, batch_size=loader.batch_size)
    return bin_loader

def convert_class_to_bin(y):
    ans    = torch.rand(y.size(0))
    ans[0] = 0
    for i in range(1, y.size(0)):
        if y[i] == y[i-1]:
            ans[i] = 0
        else:
            ans[i] = 1
    return ans

def make_seq_loader(loader, seq_len, stride):
    # input : loader of size [#n, 1, #dim], [#n]
    # return: loader of size [#n, seq_len, #dim], [#n]

    x, y   = loader.dataset.tensors[0], loader.dataset.tensors[1]
    idx    = gen_seq(x.shape[0], seq_len, stride)
    xx, yy = [x[i:i+seq_len, :, :] for i in idx], [y[i:i+seq_len] for i in idx]
    xx     = [x.reshape(-1, x.shape[0]*x.shape[2]) for x in xx]
    xx, yy = [x.unsqueeze(0) for x in xx], [y.unsqueeze(0) for y in yy]
    xx, yy = torch.cat(xx), torch.cat(yy)
    dataset = TensorDataset(xx, yy)
    loader  = DataLoader(dataset, batch_size=loader.batch_size)
    return loader

def gen_seq(n, seq_len, stride):
    res = []
    for i in range(0, n, stride):
        if i + seq_len <= n:
            res.append(i)
    return res

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
