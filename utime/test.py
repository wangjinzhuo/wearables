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

from utime import *
from plot_cm import *

device = "cuda" if torch.cuda.is_available() else "cpu"

print("-------start data preparation----------")
ss_num = 2
dataset_dir='/media/jinzhuo/wjz/Data/MASS/ss'+str(ss_num)+'/'
start_time = time.time()

if os.path.isfile('../data/ss'+str(ss_num)+'_test_loader.pt'):
    print('file exist')
    test_loader = torch.load('../data/ss'+str(ss_num)+'_test_loader.pt')
else:
    print('file dont exist')
    files = os.listdir(dataset_dir)
    test_loader = make_seq_dataloader(dataset_dir, files, seq_len=35, stride=15, batch_size=32, shuffle=True, num_workers=0)
    torch.save(test_loader, '../data/ss'+str(ss_num)+'_test_loader.pt')

print("-------%s seconds for data preparation----------" % (time.time() - start_time))

print("building model...")
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
best_acc = checkpoint["acc"]
print("best acc: ", best_acc)

def test():
    net.eval()
    correct, total = 0, 0
    pred, gt = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            print(batch_idx)
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
            outputs = net(inputs)
            predicted = outputs.max(1)[1]
            total += torch.numel(targets)
            correct += predicted.eq(targets).sum().item()

            pred.append(outputs.cpu()), gt.append(targets.cpu())
    test_acc = correct/total
    print(correct, '/', total, ': ', test_acc)

    # draw cm
    pred, gt = np.concatenate(pred), np.concatenate(gt)
    pred, gt = pred.reshape(-1), gt.reshape(-1)
    plot_confusion_matrix_from_data(gt, pred, [], True, 'Oranages', '.2f', 0.5, False, 2, 'y')

test()
