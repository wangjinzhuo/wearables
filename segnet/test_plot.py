import torch
import numpy as np
import time

from plot_psg import *

t = time.time()

data = torch.load('../../data/test_loader.pt')

print('{}s'.format(time.time() - t))

x = data.dataset.tensors[0]
y = data.dataset.tensors[1]

x = x.numpy()
y = y.tolist()

x = np.expand_dims(x, -1)

#plot_period(x[0][0], str(int(y[0][0].index(1.0))), ['EEG'], 0, 200, return_fig=False)
#'''
plot_periods([x[3][i] for i in range(20,30)],
             [str(int(y[3][i].index(1.0))) for i in range(20,30)],
             channel_names=['EEG'],
             init_second=0,
             sample_rate=200,
             highlight_periods=True,
             return_fig=False)
#'''
