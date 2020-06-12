import os
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn

from dsn import *
from plot_cm import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_loader = torch.load('/media/jinzhuo/wjz/Data/loader/mass/ch_0/ss_4.pt')
net = DeepSleepNet()
net = net.to(device)
if device == 'cuda':
    net = nn.DataParallel(net)
    cudnn.benchmark = True

# load checkpoint
assert os.path.isfile('checkpoint/ckpt.pth'), 'no checkpoint file found, please download it at: and save it at checkpoint directory'
checkpoint = torch.load('checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
print('best acc: ', checkpoint['acc'])

def test():
    net.eval()
    pred, gt = [], []
    for idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.long)
        outputs = net(inputs) # outputs bs, 5; targets bs, 5
        outputs, targets = outputs.max(1)[1], targets.max(1)[1]
        pred.append(outputs.cpu())
        gt.append(targets.cpu())

    pred, gt = np.concatenate(pred), np.concatenate(gt)
    pred, gt = pred.reshape(-1), gt.reshape(-1)
    print('acc: ', (pred == gt).sum() / len(gt))
    # draw cm
    plot_confusion_matrix_from_data(gt, pred, [], True, 'Oranges', '.2f', 0.5, False, 2, 'y')

test()
