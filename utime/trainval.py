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

from utils import *
from utime import *

def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    args = parser.parse_args()
    return args

args = parse_cmd_args()

device      = "cuda" if torch.cuda.is_available() else "cpu"
best_acc    = 0 # best val accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

net = Utime(ch=4)
net = net.to(device)

if args.resume:
    # load checkpoint
    print("resuming from checkpoint")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print("best acc: ", best_acc)

optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-8)

tr_path   = '/media/jinzhuo/wjz/Data/Navin_RBD/tr'
val_path  = '/media/jinzhuo/wjz/Data/Navin_RBD/val'

# Training
def train(epoch):
    print('Train - epoch: %d' % epoch)
    net.train()
    fs = os.listdir(tr_path)
    for f in fs:
        print(f)
        mat = loadmat(os.path.join(tr_path, f))
        x, y = mat['data'], mat['labels']
        x    = x.transpose(0,2,1)
        x    = x[:,:,0:-1:2]
        x    = x[:,0:4,:]
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        y    = y.max(1)[1]
        idx  = gen_seq(x.shape[0], 35, 1)
        batch_idx = list(range(0, len(idx), 32))
        for b in batch_idx[:-1]:
            correct, total = 0, 0
            xlist, ylist = [], []
            for i in range(b, b+32):
                tmpx, tmpy = x[i:i+35, :, :], y[i:i+35]
                tmpx       = tmpx.transpose(0,1)
                tmpx       = tmpx.reshape(4, 3000*35)
                tmpx, tmpy = tmpx.unsqueeze(0), tmpy.unsqueeze(0)
                xlist.append(tmpx), ylist.append(tmpy)
            inputs, targets = torch.cat(xlist), torch.cat(ylist)
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
            outputs = net(inputs)

            optimizer.zero_grad()
            loss, correct_batch, total_batch = gdl(outputs, targets, outputs.size(1))
            loss.backward()
            optimizer.step()
            assert correct_batch <= total_batch, print('error ... c: %d, t: %d' % (correct_batch, total_batch))
            correct    += correct_batch
            total      += total_batch
            print(int(b/32), ' | ', '{:.4f}'.format(loss.item()), ' | ', '{:.2f}'.format(100.*correct/total), ' | ', correct, ' | ', total)

# Validataion
def val(epoch):
    print('Val - Epoch: %d' % epoch)
    global best_acc
    net.eval()
    fs = os.listdir(val_path)
    with torch.no_grad():
        for f in fs:
            print(f)
            mat = loadmat(os.path.join(val_path, f))
            x, y = mat['data'], mat['labels']
            x    = x.transpose(0,2,1)
            x    = x[:,:,0:-1:2]
            x    = x[:,0:4,:]
            x, y = torch.from_numpy(x), torch.from_numpy(y)
            y    = y.max(1)[1]
            idx  = gen_seq(x.shape[0], 35, 1)
            batch_idx = list(range(0, len(idx), 32))
            for b in batch_idx[:-1]:
                correct, total = 0, 0
                xlist, ylist = [], []
                for i in range(b, b+32):
                    tmpx, tmpy = x[i:i+35, :, :], y[i:i+35]
                    tmpx       = tmpx.transpose(0,1)
                    tmpx       = tmpx.reshape(4, 3000*35)
                    tmpx, tmpy = tmpx.unsqueeze(0), tmpy.unsqueeze(0)
                    xlist.append(tmpx), ylist.append(tmpy)
                inputs, targets = torch.cat(xlist), torch.cat(ylist)
                inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
                outputs = net(inputs)

                optimizer.zero_grad()
                loss, correct_batch, total_batch = gdl(outputs, targets, outputs.size(1))
                correct  += correct_batch
                total    += total_batch
                assert correct_batch <= total_batch, print('error ... c: %d, t: %d' % (correct_batch, total_batch))
                print(int(b/32), ' | ', '{:.4f}'.format(loss.item()), ' | ', '{:.2f}'.format(100.*correct/total), ' | ', correct, ' | ', total)
                #progress_bar(idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #         % (val_loss/(idx+1), 100.*correct/total, correct, total))

                # Save checkpoint.
                acc = 100.*correct/total
                if acc > best_acc:
                    print('Saving..')
                    print(acc)
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/stft_ckpt.pth')
                best_acc = acc

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
for epoch in range(start_epoch, start_epoch+400):
    train(epoch)
    val(epoch)
    lr_scheduler.step()

print("best acc: ", best_acc)
