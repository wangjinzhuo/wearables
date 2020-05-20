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

device = "cuda" if torch.cuda.is_available() else "cpu"


test_loader = torch.load('../data/mass/ss5_loader.pt')

bnet, snet, pnet = Bnet(), Snet(), Pnet()
bnet, snet, pnet = bnet.to(device), snet.to(device), pnet.to(device)
if device == "cuda":
    bnet, snet, pnet = nn.DataParallel(bnet), nn.DataParallel(snet), nn.DataParallel(pnet),
    cudnn.benchmark = True

# load checkpoint
print("resuming from best checkpoint")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
checkpoint = torch.load("./checkpoint/step2_bnet.pth")
bnet.load_state_dict(checkpoint["net"])

checkpoint = torch.load("./checkpoint/snet.pth")
snet.load_state_dict(checkpoint["net"])

checkpoint = torch.load("./checkpoint/pnet.pth")
pnet.load_state_dict(checkpoint["net"])

best_acc = checkpoint["acc"]
print("best acc: ", best_acc)

def test():
    bnet.eval()
    snet.eval()
    pnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            print(batch_idx)
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
            bout  = bnet(inputs)
            sout  = snet(bout)
            bsout = torch.max(sout, dim=2)[1]
            pin   = seg_pool(bout, bsout)
            pout  = pnet(pin)
            predicted = pout.max(1)[1]
            total    += torch.numel(targets)
            correct  += predicted.eq(targets).sum().item()
    test_acc = correct/total
    print(correct, '/', total, ': ', test_acc)

test()
