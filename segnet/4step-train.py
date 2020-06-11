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
    args = parser.parse_args()
    return args

args = parse_cmd_args()
device = "cuda" if torch.cuda.is_available() else "cpu"

print("preparing dataloader ...")
train_loader     = torch.load('/media/jinzhuo/wjz/Data/loader/mass/ch_0/ss_1.pt')
#val_loader       = torch.load('/media/jinzhuo/wjz/Data/loader/mass/ch_0/ss_2.pt')
train_loader     = make_seq_loader(train_loader, seq_len=128, stride=64)
#val_loader       = make_seq_loader(val_loader, seq_len=128, stride=64)
bin_train_loader = make_bin_loader(train_loader)
#bin_val_loader   = make_bin_loader(val_loader)

print("finish ...")

bnet0, fcnet, bnet1, bnet2, snet1, snet2, pnet1, pnet2 = Bnet(), Fc(), Bnet(), Bnet(), Snet(), Snet(), Pnet(), Pnet()
bnet0, fcnet, bnet1, bnet2, snet1, snet2, pnet1, pnet2 = bnet0.to(device), fcnet.to(device), bnet1.to(device), bnet2.to(device), snet1.to(device), snet2.to(device), pnet1.to(device), pnet2.to(device)

if device == "cuda":
    cudnn.benchmark = True
    nn.DataParallel(bnet0), nn.DataParallel(bnet1), nn.DataParallel(bnet2), nn.DataParallel(snet1), nn.DataParallel(snet2), nn.DataParallel(pnet1), nn.DataParallel(pnet2)


def step0(epoch):
    global step0_acc
    print('\n')
    print('step 0: epoch {}'.format(epoch))
    print('loss | corr | total | acc')
    bnet0.train()
    fcnet.train()
    loss, corr, total = 0, 0, 0
    for idx, (x, gt) in enumerate(train_loader):
        x, gt = x.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        if x.size(2) != 128*3000:
            idx = list(range(0, 6000*128, 2))
            x   = x[:, :, idx]
        bout = bnet0(x)
        fcout = fcnet(bout)
        batch_loss, batch_corr, batch_total = gdl(fcout, gt, fcout.size(2))
        optimizer_0.zero_grad()
        batch_loss.backward()
        optimizer_0.step()

        loss += batch_loss
        corr += batch_corr
        total += batch_total

        print('%.4f' % batch_loss.item(), batch_corr, batch_total, '%.4f' % (batch_corr/batch_total))
        #progress_bar(idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (loss/(idx+1), 100.*corr/total, corr, total))
    acc = corr/total
    if acc > step0_acc:
        step0_acc = acc
        bnet0_state = {'net': bnet0.state_dict(), 'acc': acc}
        torch.save(bnet0_state, './4step_ckp/bnet0.pt')
        fcnet_state = {'net': fcnet.state_dict(), 'acc': acc}
        torch.save(fcnet_state, './4step_ckp/fcnet.pt')
        record.write(str(acc) + '; ')

def step1(epoch):
    global step1_acc
    print('\n')
    print('step 1: epoch {}'.format(epoch))
    print('loss | corr | total | acc')
    bnet1.train()
    snet1.train()
    loss, corr, total = 0, 0, 0
    for idx, (x, gt) in enumerate(bin_train_loader):
        x, gt = x.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        if x.size(2) != 128*3000:
            idx = list(range(0, 6000*128, 2))
            x   = x[:, :, idx]
        bout = bnet2(x)
        sout = snet1(bout)
        batch_loss, batch_corr, batch_total = gdl(sout, gt, fcout.size(2))
        optimizer_1.zero_grad()
        batch_loss.backward()
        optimizer_1.step()

        loss += batch_loss
        corr += batch_corr
        total += batch_total
        print('%.4f' % batch_loss.item(), batch_corr, batch_total, '%.4f' % (batch_corr/batch_total))
        #progress_bar(idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (loss/(idx+1), 100.*corr/total, corr, total))
    acc = corr/total
    if acc > step1_acc:
        step1_acc = acc
        bnet1_state = {'net': bnet1.state_dict(), 'acc': acc}
        torch.save(bnet1_state, './4step_ckp/bnet1.pt')
        snet1_state = {'net': snet1.state_dict(), 'acc': acc}
        torch.save(snet1_state, './4step_ckp/snet1.pt')
        record.write(str(acc) + '; ')

def step2(epoch):
    global step2_acc
    print('\n')
    print('step 2: epoch {}'.format(epoch))
    print('loss | corr | total | acc')
    bnet2.train()
    snet1.eval()
    pnet1.train()
    loss, corr, total = 0, 0, 0
    for idx, (x, gt) in enumerate(train_loader):
        x, gt = x.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        if x.size(2) != 128*3000:
            idx = list(range(0, 6000*128, 2))
            x   = x[:, :, idx]
        bout = bnet2(x)
        sout = snet1(bout)
        bsout = torch.max(sout, dim=2)[1]
        pin   = seg_pool(bout, bsout)
        pout  = pnet1(pin)
        batch_loss, batch_corr, batch_total = gdl(pout, gt, pout.size(2))
        optimizer_2.zero_grad()
        batch_loss.backward()
        optimizer_2.step()

        loss += batch_loss
        corr += batch_corr
        total += batch_total
        print('%.4f' % batch_loss.item(), batch_corr, batch_total, '%.4f' % (batch_corr/batch_total))
        #progress_bar(idx, len(train_loader), 'loss: %.3f | acc: %.3f%% (%d/%d)'
        #             % (loss/(idx+1), 100.*corr/total, corr, total))
    acc = corr/total
    if acc > step2_acc:
        step2_acc = acc
        bnet2_state = {'net': bnet2.state_dict(), 'acc': acc}
        torch.save(bnet2_state, './4step_ckp/bnet2.pt')
        pnet1_state = {'net': pnet1.state_dict(), 'acc': acc}
        torch.save(pnet1_state, './4step_ckp/pnet1.pt')
        record.write(str(acc) + '; ')

def step3(epoch):
    global step3_acc
    print('\n')
    print('step 3: epoch {}'.format(epoch))
    print('loss | corr | total | acc')
    bnet2.eval()
    snet2.train()
    loss, corr, total = 0, 0, 0
    for idx, (x, gt) in enumerate(bin_train_loader):
        x, gt = x.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        if x.size(2) != 128*3000:
            idx = list(range(0, 6000*128, 2))
            x   = x[:, :, idx]
        bout = bnet2(x)
        sout = snet2(bout)
        batch_loss, batch_corr, batch_total = gdl(sout, gt, sout.size(2))
        optimizer_3.zero_grad()
        batch_loss.backward()
        optimizer_3.step()

        loss += batch_loss
        corr += batch_corr
        total += batch_total
        print('%.4f' % batch_loss.item(), batch_corr, batch_total, '%.4f' % (batch_corr/batch_total))
        #progress_bar(idx, len(bin_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (loss/(idx+1), 100.*corr/total, corr, total))

    acc = corr/total
    if acc > step3_acc:
        step3_acc = acc
        snet2_state = {'net': snet2.state_dict(), 'acc': acc}
        torch.save(snet2_state, './4step_ckp/snet2.pt')
        record.write(str(acc) + '; ')

def step4(epoch):
    global step4_acc
    print('\n')
    print('step 4: epoch {}'.format(epoch))
    print('loss | corr | total | acc')
    bnet2.eval()
    snet2.eval()
    pnet2.train()
    loss, corr, total = 0, 0, 0
    for idx, (x, gt) in enumerate(train_loader):
        x, gt = x.to(device, dtype=torch.float), gt.to(device, dtype=torch.long)
        if x.size(2) != 128*3000:
            idx = list(range(0, 6000*128, 2))
            x   = x[:, :, idx]
        bout = bnet2(x)
        sout = snet2(bout)
        bsout = torch.max(sout, dim=2)[1]
        pin   = seg_pool(bout, bsout)
        pout  = pnet2(pin)
        batch_loss, batch_corr, batch_total = gdl(pout, gt, pout.size(2))

        optimizer_4.zero_grad()
        batch_loss.backward()
        optimizer_4.step()

        loss += batch_loss
        corr += batch_corr
        total += batch_total
        print('%.4f' % batch_loss.item(), batch_corr, batch_total, '%.4f' % (batch_corr/batch_total))
        #progress_bar(idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (loss/(idx+1), 100.*corr/total, corr, total))
    acc = corr/total
    if acc > step4_acc:
        step4_acc = acc
        pnet2_state = {'net': pnet2.state_dict(), 'acc': acc}
        torch.save(pnet2_state, './4step_ckp/pnet2.pt')
        record.write(str(acc) + '; ')

step0_num, step1_num, step2_num, step3_num, step4_num = 1000, 5000, 5000, 5000, 5000
step0_acc, step1_acc, step2_acc, step3_acc, step4_acc = 0, 0, 0, 0, 0

optimizer_0 = optim.SGD(itertools.chain(bnet0.parameters(), fcnet.parameters()), lr=1e-3, momentum=0.9, weight_decay=5e-4)

optimizer_1 = optim.SGD(itertools.chain(bnet1.parameters(), snet1.parameters()), lr=1e-4, momentum=0.9, weight_decay=5e-4)

optimizer_2 = optim.SGD(itertools.chain(bnet2.parameters(), pnet1.parameters()), lr=1e-4, momentum=0.9, weight_decay=5e-4)

optimizer_3 = optim.SGD(snet2.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

optimizer_4 = optim.SGD(pnet2.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)

if __name__ == '__main__':

    record = open('acc.txt','a+')

    record.write('step0: ')
    for i in range(step0_num):
        step0(i) # (bnet0) + fcnet

    record.write('step1: ')

    bnet1 = bnet0
    for i in range(step1_num):
        step1(i) # (bnet1) + (snet1)

    record.write('step2: ')

    bnet2 = bnet0
    for i in range(step2_num):
        step2(i) # (bnet2) + snet1 + (pnet1)

    record.write('step3: ')

    snet2 = snet1
    for i in range(step3_num):
        step3(i) # bnet2 + (snet2)

    record.write('step4: ')

    pnet2 = pnet1
    for i in range(step4_num):
        step4(i) # bnet2 + snet2 + (pnet2)

    record.close()
