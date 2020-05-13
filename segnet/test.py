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
from utils import *
from plot_cm import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_cmd_args():
    parser = argparse.ArgumentParser(description="test set")
    parser.add_argument("--test_set", default='5', help="choose a set to test, from 1-5")
    args = parser.parse_args()
    return args

args = parse_cmd_args()
test_set = args.test_set

print("building model...")
net = Utime()
net = net.to(device)
'''
if device == "cuda":
    net = nn.DataParallel(net)
    cudnn.benchmark = True
'''

# load checkpoint
print("resuming from best checkpoint")
assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
checkpoint = torch.load("checkpoint/ckpt.pth")
net.load_state_dict(checkpoint['net'])
best_acc = checkpoint['acc']
print("best acc: ", best_acc)

print("-------start data preparation----------")
start_time = time.time()

if os.path.isfile('../data/ss'+str(test_set)+'_loader.pt'):
    print('test_set ss'+str(test_set)+' loader file exist')
    test_loader = torch.load('../data/ss'+str(test_set)+'_loader.pt')
else:
    print('test_set ss'+str(test_set)+' loader file dont exist')
    test_loader = make_feat_seq_loader(dataset_dir+'ss'+str(args.test_set), 32, 35, 3000)
    torch.save(test_loader, '../data/ss'+str(test_set)+'_loader.pt')

print("-------%s seconds for data preparation----------" % (time.time() - start_time))

def test():
    net.eval()
    correct = 0
    total = 0
    all_pred = []
    all_gt = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
            outputs = net(inputs) # bs, 5, 35 ; targets bs, 35
            predicted = outputs.max(1)[1]
            correct += predicted.eq(targets).sum().item()
            total += torch.numel(targets)

            all_pred.append(predicted.cpu())
            all_gt.append(targets.cpu())
    test_acc = correct/(total+0.0)
    print(test_acc)
    # draw cm
    pred, gt = np.concatenate(all_pred), np.concatenate(all_gt)
    pred, gt = pred.reshape(-1), gt.reshape(-1)
    plot_confusion_matrix_from_data(gt, pred, [], True, 'Organges', '.2f', 0.5, False, 2, 'y')

test()
