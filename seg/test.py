import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import math
import random
import argparse

from model import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

test_loader = torch.load('/media/jinzhuo/wjz/Data/loader/mass/ss_3.pt')
seq_test_loader = make_seq_loader(test_loader, seq_len=128, stride=64)
bin_test_loader = make_bin_loader(seq_test_loader)

step1_bnet, step2_bnet, snet, pnet = Bnet(), Bnet(), Snet(), Pnet()
step1_bnet, step2_bnet, snet, pnet = step1_bnet.to(device), step2_bnet.to(device), snet.to(device), pnet.to(device)

if device == "cuda":
    step1_bnet, step2_bnet, snet, pnet = nn.DataParallel(step1_bnet), nn.DataParallel(step2_bnet), nn.DataParallel(snet), nn.DataParallel(pnet),
    cudnn.benchmark = True

# load checkpoint
print("resuming from best checkpoint")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
checkpoint = torch.load("./checkpoint/step1_bnet.pth")
step1_bnet.load_state_dict(checkpoint["net"])

print(checkpoint['acc'])

checkpoint = torch.load("./checkpoint/step2_bnet.pth")
step2_bnet.load_state_dict(checkpoint["net"])

print(checkpoint['acc'])

checkpoint = torch.load("./checkpoint/snet.pth")
snet.load_state_dict(checkpoint["net"])

checkpoint = torch.load("./checkpoint/pnet.pth")
pnet.load_state_dict(checkpoint["net"])

def step1_test():
    print('step1 test ...')
    step1_bnet.eval()
    snet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(bin_test_loader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
            if inputs.size(2) != 128*3000:
                idx = list(range(0, 128*6000, 2))
                inputs = inputs[:, :, idx]
            bout  = step1_bnet(inputs)
            sout  = snet(bout)
            loss, corr_batch, total_batch = gdl(sout, targets, sout.size(2))
            total    += total_batch
            correct  += corr_batch.item()
    test_acc = correct/total
    print(correct, '/', total, ': ', test_acc)

def step2_test():
    print('step2 test ...')
    step2_bnet.eval()
    pnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(seq_test_loader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
            if inputs.size(2) != 128*3000:
                idx = list(range(0, 128*6000, 2))
                inputs = inputs[:, :, idx]
            bout  = step2_bnet(inputs)
            sout  = snet(bout)
            bsout = torch.max(sout, dim=2)[1]
            pin   = seg_pool(bout, bsout)
            pout  = pnet(pin)
            loss, corr_batch, total_batch = gdl(pout, targets, pout.size(2))
            total    += total_batch
            correct  += corr_batch.item()
    test_acc = correct/total
    print(correct, '/', total, ': ', test_acc)

step1_test()
step2_test()
