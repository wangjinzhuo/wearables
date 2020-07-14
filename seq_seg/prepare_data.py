import torch
import os
path = '/media/jinzhuo/wjz/Data/Navin (RBD)/rbd_loader/seq_loader/stft'
fs = os.listdir(path)
for f in fs:
    print(f)
    l = torch.load(path + '/' + f)
    x, y = l.dataset.tensors[0], l.dataset.tensors[1]
    ds = torch.utils.data.TensorDataset(x,y)
    ll = torch.utils.data.DataLoader(ds,batch_size=32)
    torch.save(ll, os.path.join(path,f))
