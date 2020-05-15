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
from utime import *

from plot_cm import *

device = "cuda" if torch.cuda.is_available() else "cpu"

test_loader = torch.load('../data/mass/ss5_loader.pt')
test_loader = make_seq_loader(test_loader, seq_len=35, stride=35)

net = Utime()
net = net.to(device)
if device == "cuda":
    net = nn.DataParallel(net)
    cudnn.benchmark = True

# load checkpoint
print("resuming from best checkpoint")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
checkpoint = torch.load("./checkpoint/ckpt.pth")
net.load_state_dict(checkpoint["net"])
print("best acc: ", checkpoint["acc"])

def test():
    net.eval()
    correct = 0
    total = 0
    pred, gt = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            print(batch_idx)
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
            outputs = net(inputs)
            predicted = outputs.max(1)[1]
            total += torch.numel(targets)
            correct += predicted.eq(targets).sum().item()
            pred.append(predicted.cpu())
            gt.append(targets.cpu())
    test_acc = correct/total
    print("test acc: ", test_acc)
    # draw cm
    pred, gt = np.concatenate(pred), np.concatenate(gt)
    pred, gt = pred.reshape(-1), gt.reshape(-1)
    plot_confusion_matrix_from_data(gt, pred, [], True, 'Oranges', '.2f', 0.5, False, 2, 'y')

test()
