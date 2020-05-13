import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import importlib
import os
import time
import math
import random
import argparse

from utils import make_feat_seq_loader

def get_backbone(netname, num_classes, feat_dim, seq_len, sample_dim=3000):
    netmodule = importlib.import_module(netname)
    backbone_net_class = eval('netmodule.%s' % netname)
    backbone = backbone_net_class(num_classes=num_classes, feat_dim=feat_dim, seq_len=seq_len, sample_dim=sample_dim)
    return backbone


def parse_cmd_args():
    parser = argparse.ArgumentParser(description="Feature Mearusement")
    parser.add_argument("--opt", type=str, default='sgd', help="optimizer used for training")
    parser.add_argument("--lr",default=0.01, type=float, help="learning rate")
    parser.add_argument("--resume", "-r", action="store_true", help="resume from checkpoint")
    parser.add_argument("--dataset_dir", type=str, default="/media/jinzhuo/wjz/Data/MASS", help="dataset dir")
    parser.add_argument("--network", type=str, default="SegTempNet", help="The backbone used")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size for train")
    parser.add_argument("--seq_len", type=int, default=35, help="how many spleep epoch in a segment")
    parser.add_argument("--feat-dim", type=int, default=175, help="feat dim from backbone")
    parser.add_argument("--num-classes", type=int, default=5, help="class number")
    parser.add_argument("--data-feat-dim", type=int, default=3000, help="how many spleep epoch in a segment")
    parser.add_argument("--loss", type=str, default='dice_loss', help="loss function")

    args = parser.parse_args()
    return args

args = parse_cmd_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0 # best val accuracy
start_epoch = 0 # start from epoch 0 or last checkpoint epoch

batch_size = args.batch_size
seq_len = args.seq_len
dataset_dir=args.dataset_dir
print("-------start data preparation with batch_size {} seq_len {} net input sample_dim {}----------".format(batch_size, seq_len, args.data_feat_dim))
start_time = time.time()

class ParallBitLinear(nn.Module):
    def __init__(self, seq_len=35, feat_dim=175, num_class=5):
        super(ParallBitLinear, self).__init__()
        self.bit_list = []
        for i in range(seq_len):
            bit_name = 'seq_at_%d' % i
            bit = nn.Linear(feat_dim, num_class)
            setattr(self, bit_name, bit)
            self.bit_list.append(getattr(self, bit_name))

    def forward(self, x):
        bit_fcs = [bit(x) for bit in self.bit_list]
        return bit_fcs


def get_data_loader(dataset_dir, mod, batch_size, seq_len, data_feat_dim):
    train_loader = None
    val_loader = None
    mass_set_num= 1
    val_mass_set_num= 2
    if mod == 'bin':
        assert False
        train_bin = os.path.join(dataset_dir, '../data/bin_train_loader.pt')
        val_bin = os.path.join(dataset_dir, '../data/bin_val_loader.pt')
        test_bin = os.path.join(dataset_dir, '../data/bin_test_loader.pt')
        assert (os.path.isfile(train_bin) and os.path.isfile(val_bin))
        print('prepare bin files exist')
        train_loader = torch.load(train_bin)
        val_loader = torch.load(val_bin)
    elif mod == 'feat':
        if os.path.isfile('../data/ss'+str(mass_set_num)+'_loader.pt'):
            print('train_loader file exist')
            train_loader = torch.load('../data/ss'+str(mass_set_num)+'_loader.pt')
        else:
            print('loader file dont exist')
            train_loader = make_feat_seq_loader(dataset_dir+'/ss'+str(mass_set_num), batch_size, seq_len, data_feat_dim)
            torch.save(train_loader,'../data/ss'+str(mass_set_num)+'_loader.pt')
        if os.path.isfile('../data/ss'+str(val_mass_set_num)+'_loader.pt'):
            print('val_loader file exist')
            val_loader = torch.load('../data/ss'+str(val_mass_set_num)+'_loader.pt')
            torch.save(val_loader,'../data/ss'+str(val_mass_set_num)+'_loader.pt')
        else:
            print('val_loader file dont exist')
            val_loader = make_feat_seq_loader(dataset_dir+'/ss'+str(val_mass_set_num), batch_size, seq_len, data_feat_dim)
    return train_loader, val_loader

train_loader, val_loader = get_data_loader(dataset_dir=dataset_dir, batch_size=batch_size, seq_len=seq_len, mod='feat', data_feat_dim=args.data_feat_dim)
print("-------%s seconds for data preparation----------" % (time.time() - start_time))

def dice_loss(outputs, targets, seq_len, class_num):
    onehot_y = F.one_hot(targets.long(), num_classes=class_num)
    pred = outputs.permute(0, 2, 1)

    intersection = torch.sum(onehot_y * pred)
    union = torch.sum(onehot_y + pred)
    loss = 1 - 2 * intersection / (class_num*union)

    pred_label = torch.argmax(outputs, dim=1)
    correct_num = torch.sum(torch.eq(pred_label.long(), targets.long()))
    total_size = torch.numel(targets)

    acc = correct_num/(total_size+0.0)

    return loss, acc, correct_num


def ctc_loss(outputs, targets, seq_len, ce_loss_fun):
    bit_labels = torch.split(targets, 1, dim=1)
    outputs = torch.split(outputs, 1, dim=2)
    outputs = [torch.squeeze(bit_pred) for bit_pred in outputs]
    assert len(bit_labels) == seq_len
    assert len(outputs) == seq_len
    bit_total_loss = 0
    bit_correct_total = 0
    true_seq_label_prob = 1.0
    seq_pred_correct = 1

    for predict_bit, bit_label in zip(outputs, bit_labels):
        bit_loss = ce_loss_fun(predict_bit, torch.squeeze(bit_label).long())
        pred = torch.argmax(predict_bit, dim=1, keepdim=False)
        bit_correct = torch.eq(pred.long(), torch.squeeze(bit_label).long())
        seq_pred_correct = seq_pred_correct * bit_correct
        bit_correct_total += torch.sum(bit_correct)

        true_seq_label_prob = true_seq_label_prob * torch.gather(predict_bit, 1, bit_label.long())

        bit_total_loss += bit_loss

    seq_real_prob = torch.cat([true_seq_label_prob, true_seq_label_prob], dim=1)
    seq_loss = ce_loss_fun(seq_real_prob, torch.ones_like(seq_pred_correct).long())

    total_loss = bit_total_loss + seq_loss

    total_size = torch.numel(targets)

    bit_acc_mean = (bit_correct_total+0.0)/total_size
    seq_acc = torch.sum(seq_pred_correct)/len(seq_pred_correct)
    return total_loss, bit_acc_mean, seq_acc

print("-------prepare network and loss function ----------")
feat_dim=args.feat_dim
num_classes = args.num_classes
loss_name = args.loss

net = get_backbone(args.network, num_classes, feat_dim, seq_len)
net = net.to(device)
if device == "cuda":
    net = nn.DataParallel(net)
    cudnn.benchmark = True
print("-------prepare network done ----------")

if loss_name == 'dice_loss':
    print("==== choose dice loss ===========")
    loss_param = num_classes
elif loss_name == 'ctc_loss':
    print("==== choose ctc loss ===========")
    loss_param = nn.CrossEntropyLoss()
else:
    assert False
loss_fun = eval(loss_name)


if args.resume:
    # load checkpoint
    print("resuming from checkpoint")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found"
    checkpoint = torch.load("./checkpoint/ckpt.pth")
    net.load_state_dict(checkpoint["net"])
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
    print("best acc: ", best_acc)


if args.opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.opt == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
else:
    assert False


# Training
def train(epoch):
    print('\nTrain epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs  = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        optimizer.zero_grad()

        outputs = net(inputs)

        #print(outputs.size()) # torch.tensor - 16 * 5 * 35
        loss, bit_acc, correct_n = loss_fun(outputs, targets, seq_len, loss_param)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(net.parameters(), 15)
        optimizer.step()

        assert not math.isnan(loss.item())

        if batch_idx % 20 == 0:
            print("Epoch-", epoch, " Batch-", batch_idx, \
                    " Loss: %.3f"% loss.item(), "Bit Acc : %.3f"%bit_acc, " Correc n : %.0f" % correct_n)


# Validataion
def val(epoch):
    print('\nVal epoch: %d' % epoch)
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs  = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            outputs = net(inputs)

            loss, bit_acc, correct_n = loss_fun(outputs, targets, seq_len, loss_param)

            correct += correct_n.item()
            total += torch.numel(targets)

            if batch_idx % 20 == 0:
                print("Epoch-", epoch, " Batch-", batch_idx, \
                    " Loss: %.3f"% loss.item(), "Bit Acc : %.3f"%bit_acc, " Correc n : %.0f" % correct_n)

    # Save checkpoint.
    acc = correct/total
    print("val overall acc: ", acc)
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

lr_scheduler = StepLR(optimizer, step_size=50, gamma=0.1)
for epoch in range(start_epoch, start_epoch+600):
    train(epoch)
    val(epoch)
    lr_scheduler.step()

print("best val acc: ", best_acc)
