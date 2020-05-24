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
import itertools

from utils import *
from segnet import *

def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--resume",      action="store_true", help="resume from checkpoint")
    parser.add_argument("--iter_num",    default=2,   help="step1 epoch number")
    parser.add_argument("--step1_num",   default=500, help="step1 epoch number")
    parser.add_argument("--step2_num",   default=500, help="step2 epoch number")
    parser.add_argument("--train_stage", default=12,  help="select train stage")
    args = parser.parse_args()
    return args

args = parse_cmd_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

step1_best_acc    = 0 # best step1 val accuracy
step2_best_acc    = 0 # best step2 val accuracy
step1_start_epoch = 0 # start from epoch 0 or last checkpoint epoch
step2_start_epoch = 0 # start from epoch 0 or last checkpoint epoch

print("preparing train and validation dataloader ...")
train_loader     = torch.load('/media/jinzhuo/wjz/Data/loader/mass/ch_0/ss_1.pt')
val_loader       = torch.load('/media/jinzhuo/wjz/Data/loader/mass/ch_0/ss_2.pt')
train_loader     = make_seq_loader(train_loader, seq_len=128, stride=64)
val_loader       = make_seq_loader(val_loader, seq_len=128, stride=64)
bin_train_loader = make_bin_loader(train_loader)
bin_val_loader   = make_bin_loader(val_loader)

print("finish preparing train and validation dataloader ...")

step1_bnet, step2_bnet, snet, pnet = Bnet(ch=1), Bnet(ch=1), Snet(), Pnet()
step1_bnet, step2_bnet, snet, pnet = step1_bnet.to(device), step2_bnet.to(device), snet.to(device), pnet.to(device)
if device == "cuda":
    step1_bnet, step2_bnet, snet, pnet = nn.DataParallel(step1_bnet), nn.DataParallel(step2_bnet), nn.DataParallel(snet), nn.DataParallel(pnet)
    cudnn.benchmark = True

if args.resume:
    # load checkpoint
    print("resuming from checkpoint")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
    checkpoint = torch.load("./checkpoint/step1_bnet.pth")
    step1_bnet.load_state_dict(checkpoint["net"])
    step1_start_epoch = checkpoint["epoch"]
    step1_best_acc    = checkpoint["acc"]

    checkpoint = torch.load("./checkpoint/snet.pth")
    snet.load_state_dict(checkpoint["net"])

    '''
    checkpoint = torch.load("./checkpoint/step2_bnet.pth")
    step2_bnet.load_state_dict(checkpoint["net"])
    '''

    checkpoint = torch.load("./checkpoint/pnet.pth")
    pnet.load_state_dict(checkpoint["net"])
    step2_start_epoch = checkpoint["epoch"]
    step2_best_acc    = checkpoint["acc"]

# step1 - training
def step1_train(epoch):
    print('Step 1: Train - Epoch: {}'.format(epoch))
    step1_bnet.train()
    snet.train()

    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(bin_train_loader):
        inputs  = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)
        #print(inputs.size(), targets.size()) # bs, 1, dim*128;  bs, 128
        if inputs.size(2) != 128*3000:
            idx    = list(range(0,6000*128,2))
            inputs = inputs[:,:,idx]
        step1_optimizer.zero_grad()
        bout = step1_bnet(inputs) # bs, seq_len, class_num
        sout = snet(bout) # bs, seq_len, class_num

        loss, correct_batch, total_batch = gdl(sout, targets, sout.size(2))
        loss.backward()
        step1_optimizer.step()

        train_loss += loss.item()
        correct    += correct_batch
        total      += total_batch
        progress_bar(batch_idx, len(bin_train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# step1 - validataion
def step1_val(epoch):
    print('Step 1: Valid - Epoch: {}'.format(epoch))
    global step1_best_acc
    step1_bnet.eval()
    snet.eval()

    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(bin_val_loader):
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            if inputs.size(2) != 128*3000:
                idx    = list(range(0,6000*128,2))
                inputs = inputs[:,:,idx]
            bout = step1_bnet(inputs)
            sout = snet(bout)

            loss, correct_batch, total_batch = gdl(sout, targets, sout.size(2))

            correct  += correct_batch
            total    += total_batch
            val_loss += loss.item()
            progress_bar(batch_idx, len(bin_val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > step1_best_acc:
        print('Saving step1_bnet, snet ...')
        print(acc)
        b_state = {
            'net': step1_bnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(b_state, './checkpoint/step1_bnet.pth')
        s_state = {
            'net': snet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(s_state, './checkpoint/snet.pth')
        step1_best_acc = acc

# step2 - training
def step2_train(epoch):
    print('Step 2: Train - Epoch: {}'.format(epoch))
    step2_bnet.train()
    snet.eval()
    pnet.train()

    train_loss, correct, total = 0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs  = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)
        #print(inputs.size(), targets.size()) # bs, 1, dim*128;  bs, 128
        if inputs.size(2) != 128*3000:
            idx    = list(range(0,6000*128,2))
            inputs = inputs[:,:,idx]
        step2_optimizer.zero_grad()

        bout  = step2_bnet(inputs) # bs, ch, dim
        sout  = snet(bout)   # bs, seq_len, 2

        # evaluate snet
        slist = []
        for i in range(targets.size(0)):
            slist.append(convert_class_to_bin(targets[i]))
        starget = torch.cat([x.unsqueeze(0) for x in slist])
        starget = starget.to(device, dtype=torch.long)
        sloss, scorr, stotal = gdl(sout, starget, sout.size(2))
        print('snet acc: {}'.format(scorr/stotal))
        # finish

        bsout = torch.max(sout, dim=2)[1] # binary segment vector: bs, seq_len
        pin   = seg_pool(bout, bsout) # bs, ch, dim(1/16)
        pout  = pnet(pin) # bs, seq_len, 5

        loss, correct_batch, total_batch = gdl(pout, targets, pout.size(2))
        loss.backward()
        step2_optimizer.step()

        train_loss += loss.item()
        correct    += correct_batch
        total      += total_batch
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# step2 - validataion
def step2_val(epoch):
    print('Step 2: Valid - Epoch: {}'.format(epoch))
    global step2_best_acc
    step2_bnet.eval()
    snet.eval()
    pnet.eval()

    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)
            if inputs.size(2) != 128*3000:
                idx    = list(range(0,6000*128,2))
                inputs = inputs[:,:,idx]
            bout  = step2_bnet(inputs)
            sout  = snet(bout)

            # evaluate snet
            slist = []
            for i in range(targets.size(0)):
                slist.append(convert_class_to_bin(targets[i]))
            starget = torch.cat([x.unsqueeze(0) for x in slist])
            starget = starget.to(device, dtype=torch.long)
            sloss, scorr, stotal = gdl(sout, starget, sout.size(2))
            print('snet acc: {}'.format(scorr/stotal))
            # finish

            bsout = torch.max(sout, dim=2)[1]
            pin   = seg_pool(bout, bsout)
            pout  = pnet(pin)
            loss, correct_batch, total_batch = gdl(pout, targets, pout.size(2))

            correct  += correct_batch
            total    += total_batch
            val_loss += loss.item()
            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > step2_best_acc:
        print('Saving step2_bnet, pnet ...')
        print(acc)
        '''
        b_state = {
            'net': step2_bnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(b_state, './checkpoint/step2_bnet.pth')
        '''
        p_state = {
            'net': pnet.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(p_state, './checkpoint/pnet.pth')
        step2_best_acc = acc

#optimizer = optim.Adam(net.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-8)
iter_num    = args.iter_num
step1_num   = args.step1_num
step2_num   = args.step2_num
train_stage = args.train_stage

for iter_ in range(iter_num):
    print(iter_)
    # step1: bnet + snet
    if '1' in str(train_stage):
        print('stage 1 ...')
        step1_optimizer = optim.SGD(itertools.chain(step1_bnet.parameters(), snet.parameters()), lr=1e-4, momentum=0.9, weight_decay=5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(step1_optimizer, step_size=50, gamma=0.5)
        for epoch in range(step1_start_epoch, step1_start_epoch + step1_num):
            step1_train(epoch)
            step1_val(epoch)
            lr_scheduler.step()

    if '2' in str(train_stage):
        # step2: bnet + pnet -- keep snet's params fixed
        print('stage 2 ...')
        step2_bnet = step1_bnet

        #step2_optimizer = optim.SGD(itertools.chain(step2_bnet.parameters(), pnet.parameters()), lr=1e-5, momentum=0.9, weight_decay=5e-4)
        step2_optimizer = optim.SGD(pnet.parameters(), lr=1e-5, momentum=0.9, weight_decay=5e-4)
        lr_scheduler    = optim.lr_scheduler.StepLR(step2_optimizer, step_size=50, gamma=0.5)
        for epoch in range(step2_start_epoch, step2_start_epoch + step2_num):
            step2_train(epoch)
            step2_val(epoch)
            lr_scheduler.step()

    iter_ += 1
