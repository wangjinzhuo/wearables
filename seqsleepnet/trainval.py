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

def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    args = parser.parse_args()
    return args

args = parse_cmd_args()

device      = "cuda" if torch.cuda.is_available() else "cpu"
best_acc    = 0 # best val accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

print("preparing loader ...")
train_loader = torch.load('/media/jinzhuo/wjz/Data/loader/mass/ch_0/ss_5.pt')
val_loader   = torch.load('/media/jinzhuo/wjz/Data/loader/mass/ch_0/ss_2.pt')
train_loader = make_seq_loader(train_loader, seq_len=20, stride=40)
val_loader   = make_seq_loader(val_loader, seq_len=20, stride=40)

tr_y, val_y  = train_loader.dataset.tensors[1], val_loader.dataset.tensors[1]
print('training sample: ', tr_y.size(0))
print('valid sample: ', val_y.size(0))

print("finish ...")

net = SeqSleepNet(seq_len=20, class_num=5)
net = net.to(device)
if device == "cuda":
    net = nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # load checkpoint
    print("resuming from checkpoint")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print("best acc: ", best_acc)

optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

# Training
def train(epoch):
    print('Train - Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs  = preprocessing(inputs)
        inputs  = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)
        if inputs.size(2) != 20*3000:
            idx    = list(range(0,6000*20,2))
            inputs = inputs[:,:,idx]
        optimizer.zero_grad()
        outputs = net(inputs)

        print(inputs.size(), '... ', outputs.size(), '... ', targets.size())
        outputs = outputs.transpose(0, 2, 1)
        loss, correct_batch, total_batch = gdl(outputs, targets, outputs.size(1))
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
            inputs  = preprocessing(inputs)
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            if inputs.size(2) != 20*3000:
                idx    = list(range(0,6000*20,2))
                inputs = inputs[:,:,idx]
            outputs = net(inputs)
            outputs = outputs.transpose(0, 2, 1)
            loss, correct_batch, total_batch = gdl(outputs, targets, outputs.size(1))
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
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
for epoch in range(start_epoch, start_epoch+400):
    train(epoch)
    val(epoch)
    lr_scheduler.step()

print("best acc: ", best_acc)
