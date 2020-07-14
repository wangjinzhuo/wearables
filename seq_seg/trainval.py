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
from model import *

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='prepare args')
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
parser.add_argument("--loss", default='ce')
args = parser.parse_args()

device      = "cuda" if torch.cuda.is_available() else "cpu"
best_acc    = 0 # best val accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch


print("preparing loader ...")
path = '/media/jinzhuo/wjz/Data/Navin (RBD)/rbd_loader/seq_loader/stft/'
loader_list  = os.listdir(path)
#train_loader = torch.load(path + loader_list[0])
#val_loader   = torch.load(path + loader_list[1])
filterbanks  = torch.from_numpy(lin_tri_filter_shape(32, 256, 100, 0, 50)).to(torch.float).cuda()
net = Bnet(filterbanks=filterbanks, ch_num=4, seq_len=128, class_num=5)
net = net.to(device)
print("finish ...")

loss_criterion = seq_cel if args.loss == 'ce' else gdl
optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)

if args.resume:
    # load checkpoint
    print("resuming from checkpoint")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
    checkpoint = torch.load("./checkpoint/stft_ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print("best acc: ", best_acc)


'''
# Training
def train(epoch):
    print('Train - Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs  = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = net(inputs)


        outputs = outputs.transpose(2, 1)
        loss, correct_batch, total_batch = loss_criterion(outputs, targets, outputs.size(1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        correct    += correct_batch
        total      += total_batch
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Validataion
def val(epoch):
    print('Val - Epoch: %d' % epoch)
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            outputs = net(inputs)
            outputs = outputs.transpose(2, 1)
            loss, correct_batch, total_batch = loss_criterion(outputs, targets, outputs.size(1))
            correct  += correct_batch
            total    += total_batch
            val_loss += loss.item()
            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

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


'''

def train(epoch):
    print('Train - Epoch: %d' % epoch)
    net.train()
    train_loss, correct, total = 0, 0, 0

    for i in range(10):
        loader = torch.load(path+loader_list[i])
        for batch_idx, data in enumerate(loader, 0):
            inputs, targets = data
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = net(inputs)
            outputs = outputs.transpose(2, 1)
            loss, correct_batch, total_batch = loss_criterion(outputs, targets, outputs.size(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct    += correct_batch
            total      += total_batch
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def val(epoch):
    print('Val - Epoch: %d' % epoch)
    global best_acc
    net.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for i in range(5):
            loader  = torch.load(path+loader_list[i+10])
            for batch_idx, data in enumerate(loader, 0):
                inputs, targets = data
                inputs  = inputs.to(device, dtype=torch.float)
                targets = targets.to(device, dtype=torch.long)
                outputs = net(inputs)
                outputs = outputs.transpose(2, 1)
                loss, correct_batch, total_batch = loss_criterion(outputs, targets, outputs.size(1))
                correct  += correct_batch
                total    += total_batch
                val_loss += loss.item()
                progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

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
