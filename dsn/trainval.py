import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import os
import time
import math
import random
import argparse

from dsn import *
from utils import *
#from utils import progress_bar

parser = argparse.ArgumentParser(description="Feature Mearusement")
parser.add_argument("--lr",default=0.05, type=float, help="learning rate")
parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0 # best validation accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

print("-------start data preparation----------")
ss_num = 4

dataset_dir = '/media/jinzhuo/wjz/Data/MASS/ss4/'

start_time = time.time()
if os.path.isfile('../dsn_data/ss'+str(ss_num)+'_train.pt') and os.path.isfile('../dsn_data/ss'+str(ss_num)+'_test.pt'):
    print('loader exist')
    trainloader = torch.load('../dsn_data/ss'+str(ss_num)+'_train.pt')
    valloader = torch.load('../dsn_data/ss'+str(ss_num)+'_test.pt')
else:
    print('loader dont exist')
    files = os.listdir(dataset_dir)
    l = list(range(len(files)))
    random.shuffle(l)
    s = math.ceil(len(files)*0.8)
    train_files = [files[i] for i in l[:s]]
    val_files = [files[i] for i in l[s:]]
    trainloader = make_dataloader(dataset_dir, train_files, batch_size=128, shuffle=True, num_workers=0)
    valloader = make_dataloader(dataset_dir, val_files, batch_size=128, shuffle=True, num_workers=0)
    torch.save(trainloader, '../dsn_data/ss'+str(ss_num)+'_train.pt')
    torch.save(valloader, '../dsn_data/ss'+str(ss_num)+'_test.pt')
'''
trainloader = torch.load('../../dsn_data/fpz_cz_tr_loader.pt')
valloader = torch.load('../../dsn_data/fpz_cz_val_loader.pt')
'''
print("-------%s seconds for data preparation----------" % (time.time() - start_time))

print("building model...")
net = DeepSleepNet()
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
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
else:
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()
lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

# Training
def train(epoch):
    print('train epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
        print(inputs.size(), targets.size())
        optimizer.zero_grad()
        outputs = net(inputs)
        targets = torch.max(targets, 1)[1]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

# Validation
def val(epoch):
    print('val epoch: %d' % epoch)
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long) # RuntimeError: Expected object of scalar type Long but got scalar type Byte for argument #2 'target' in call to _thnn_nll_loss_forward
            outputs = net(inputs)
            targets = torch.max(targets, 1)[1] # solve issue "RuntimeError multi target not supported at /pytorch/.../ClassNLLCrit.."
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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
        torch.save(state, './checkpoint/ckpt'+str(acc)+'.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    val(epoch)
    lr_scheduler.step()

print("best acc: ", best_acc)
