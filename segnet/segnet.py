import torch
import torch.nn as nn

class Segnet(nn.Module):
    def __init__(self):
        super(Segnet, self).__init__()
        self.bnet = nn.Sequential(
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
        self.snet = nn.Sequential(
            nn.Conv1d(256, 64, 16, 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(3904, 512),
            nn.Linear(512, 128),
        )
        self.pnet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.Linear(1024, 128),
        )

    def forward(self, x):
        bout         = self.bnet(x) # bs, 256, 256
        sout         = self.snet(bout) # bs, 128
        seg_parabit  = Parabit(128, 128, 2)
        seg_output   = seg_parabit(sout)
        pin          = seg_pool(bout, seg_output)
        pout         = self.pnet(pin)
        pred_parabit = Parabit(128, 128, 5)
        pred_output  = pred_parabit(pout)
        return pred_output

def seg_pool(x, segment):
    # x     : [bs, 256, 256], segment: 128
    # return: [bs, 256, 16]   downsample 16 times
    return torch.rand(x.size(0), 256, 16)

class Parabit(nn.Module):
    def __init__(self, seq_len, feat_dim, class_num):
        super(Parabit, self).__init__()
        self.bits = []
        for i in range(seq_len):
            bit_name = 'seq_at_%d' % i
            bit      = nn.Linear(feat_dim, class_num)
            setattr(self, bit_name, bit)
            self.bits.append(getattr(self, bit_name))

    def forward(self, x):
        bit_fcs    = [bit(x) for bit in self.bits]
        torch_bits = [bits.unsqueeze(0) for bits in bit_fcs]
        torch_bits = torch.cat(torch_bits)
        return torch_bits

if __name__ == '__main__':
    net = Segnet()
    x = torch.rand(2, 1, 128*3000)
    y = net(x)
    print(y.size())
    print('total param num: {}'.format(
        sum(torch.numel(p) for p in net.parameters())
        )
    )
