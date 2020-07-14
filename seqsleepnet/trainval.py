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
from seqsleepnet import *

parser = argparse.ArgumentParser(description='prepare args')
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--loss", default='gdl')
args = parser.parse_args()

device      = "cuda" if torch.cuda.is_available() else "cpu"
best_acc    = 0 # best val accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

loss_criterion = seq_cel if args.loss == 'ce' else gdl

filterbanks  = torch.from_numpy(lin_tri_filter_shape(32, 256, 100, 0, 50)).to(torch.float).cuda()
net = SeqSleepNet(filterbanks=filterbanks, seq_len=128, class_num=5)
net = net.to(device)

if args.resume:
    # load checkpoint
    print("resuming from checkpoint")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
    checkpoint = torch.load("./checkpoint/stft_ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print("best acc: ", best_acc)

optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)

tr_path   = '/media/jinzhuo/wjz/Data/Navin_RBD/tr'
val_path  = '/media/jinzhuo/wjz/Data/Navin_RBD/val'

# Training
def train(epoch):
    print('Train - Epoch: %d' % epoch)
    net.train()
    fs = os.listdir(tr_path)
    for f in fs:
        print(f)
        mat = loadmat(os.path.join(tr_path, f))
        x, y = mat['data'], mat['labels']
        x    = x.transpose(2, 0, 1)
        x    = x[:,:,0:-1:2]
        x    = preprocessing(x)
        x    = x[0,:,:]
        y    = torch.from_numpy(y)
        y    = y.max(1)[1]
        idx  = gen_seq(x.shape[0], 128, 1)
        batch_idx = list(range(0, len(idx), 32))
        for b in batch_idx[:-1]:
            correct, total = 0, 0
            xlist, ylist = [], []
            for i in range(b, b+32):
                tmpx, tmpy = x[i:i+128, :, :], y[i:i+128]
                tmpx, tmpy = tmpx.unsqueeze(0), tmpy.unsqueeze(0)
                xlist.append(tmpx), ylist.append(tmpy)
            inputs, targets = torch.cat(xlist), torch.cat(ylist)
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
            outputs = net(inputs)

            optimizer.zero_grad()
            outputs = outputs.transpose(2, 1)
            loss, correct_batch, total_batch = loss_criterion(outputs, targets, outputs.size(1))
            loss.backward()
            optimizer.step()
            assert correct_batch < total_batch, print('error ...')
            correct    += correct_batch
            total      += total_batch
            print(b, ' | ', '{:.4f}'.format(loss.item()), ' | ', '{:.2f}'.format(100.*correct/total), ' | ', correct, ' | ', total)
            #progress_bar(i, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #             % (train_loss/(i+1), 100.*correct/total, correct, total))

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
            x    = x.transpose(0, 2, 1)
            x    = x[:,:,0:-1:2]
            x    = preprocessing(x)
            x    = x[0,:,:]
            y    = torch.from_numpy(y)
            y    = y.max(1)[1]
            idx  = gen_seq(x.shape[0], 128, 1)
            batch_idx = list(range(0, len(idx), 32))
            for b in batch_idx[:-1]:
                correct, total = 0, 0
                xlist, ylist = [], []
                for i in range(b, b+32):
                    tmpx, tmpy = x[i:i+128, :, :], y[i:i+128]
                    tmpx, tmpy = tmpx.unsqueeze(0), tmpy.unsqueeze(0)
                    xlist.append(tmpx), ylist.append(tmpy)
                inputs, targets = torch.cat(xlist), torch.cat(ylist)
                inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
                outputs = net(inputs)

                optimizer.zero_grad()
                outputs = outputs.transpose(2, 1)
                loss, correct_batch, total_batch = loss_criterion(outputs, targets, outputs.size(1))
                correct  += correct_batch
                total    += total_batch
                assert correct_batch < total_batch, print('error ...')
                print(b, ' | ', '{:.4f}'.format(loss.item()), ' | ', '{:.2f}'.format(100.*correct/total), ' | ', correct, ' | ', total)
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
for epoch in range(start_epoch, start_epoch+1000):
    train(epoch)
    val(epoch)
    lr_scheduler.step()

print("best acc: ", best_acc)
