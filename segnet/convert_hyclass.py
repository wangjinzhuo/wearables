import torch
import os

from torch.utils.data import Dataset, TensorDataset, DataLoader

dataset_dir = '../../data'

files = os.listdir(dataset_dir)

for ff in files:
    print(ff)
    if 'ss4' in ff:
        continue
    df = torch.load(os.path.join(dataset_dir, ff))
    x = df.dataset.feat
    y = df.dataset.label
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=64)
    torch.save(loader, 'tmp/'+ff)
