import torch
import torch.nn as nn

class Bnet(nn.Module):
    def __init__(self):
        super(Bnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, 16, 8),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(16, 8),

            nn.Conv1d(32, 64, 8, 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, 4, 2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 256, 4, 2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(256),
        )
    def forward(self, x):
        out = self.features(x) # -1, 256, 256 - #bs, #ch, #dim
        return out

class Snet(nn.Module):
    def __init__(self):
        super(Snet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(256, 64, 16, 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3904, 512),
            nn.Linear(512, 128)
        )
        self.cls = Parabit(128, 2)

    def forward(self, x):
        features = self.features(x)
        out      = self.cls(features)
        return out

class Pnet(nn.Module):
    def __init__(self):
        super(Pnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(256, 128, 4),
            nn.Flatten(),
            nn.Linear(1664, 512),
            nn.Linear(512, 128),
            #nn.Linear(4096, 1024),
            #nn.Linear(1024, 128)
        )
        self.cls = Parabit(128, 5)
    def forward(self, x):
        features  = self.features(x)
        out       = self.cls(features)
        return out


class Parabit(nn.Module):
    def __init__(self, seq_len, class_num):
        super(Parabit, self).__init__()
        self.bits = []
        for i in range(seq_len):
            bit = nn.Linear(seq_len, class_num)
            bit_name = 'seq_at_%d' % i
            setattr(self, bit_name, bit)
            self.bits.append(getattr(self, bit_name))

    def forward(self, x):
        bit_fcs    = [bit(x) for bit in self.bits]
        torch_bits = [bits.unsqueeze(1) for bits in bit_fcs]
        torch_bits = torch.cat(torch_bits, 1) # bs, seq_len, class_num
        return torch_bits

def seg_pool(x, segment):
    # x       [bs, ch, dim=256]
    # segment [bs, 128]
    # return  [bs, ch, dim=16] downsample 16 times
    seg = torch.cat([segment, segment], dim=1)
    out = x
    for i in range(x.size(0)):
        for j in range(x.size(1)):
            out[i,j,:] = x[i,j,:]*seg[i,:]
    p   = nn.MaxPool1d(16)
    out = p(out)
    return out

if __name__ == '__main__':
    x    = torch.rand(2,1,128*3000)

    bnet = Bnet()
    bout = bnet(x)

    snet = Snet()
    sout = snet(bout)

    pnet = Pnet()
    bseg = torch.max(sout, dim=2)[1]
    pin  = seg_pool(bout, bseg) # bs, 256, 16
    pout = pnet(pin)

    print('bout: {}\nsout: {}\npout: {}'.format(bout.shape, sout.shape, pout.shape))
    print('params of pnet: {}'.format(sum(torch.numel(p) for p in pnet.parameters())))
