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
        self.snet_output = Parabit(128, 2)
        self.pnet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 1024),
            nn.Linear(1024, 128),
        )
        self.pnet_output = Parabit(128, 5)

    def forward(self, x):
        bout         = self.bnet(x) # bs, 256(ch), 256
        sout         = self.snet(bout) # bs, 128
        snet_output  = self.snet_output(sout) # bs, 128, 2
        pin          = seg_pool(bout, snet_output)
        pout         = self.pnet(pin) # bs, 128
        pnet_output  = self.pnet_output(pout) # bs, 128, 5
        return snet_output, pnet_output

def seg_pool(x, segment):
    # x     : [bs, 256, 256], segment: 128
    # return: [bs, 256, 16]   downsample 16 times
    return torch.rand(x.size(0), 256, 16)

class Parabit(nn.Module):
    def __init__(self, seq_len, class_num):
        super(Parabit, self).__init__()
        self.bits = []
        for i in range(seq_len):
            bit       = nn.Linear(seq_len, class_num)
            bit_name = 'seq_at_%d' % i
            setattr(self, bit_name, bit)
            self.bits.append(getattr(self, bit_name))
            #self.bits.append(bit)

    def forward(self, x):
        bit_fcs    = [bit(x) for bit in self.bits]
        torch_bits = [bits.unsqueeze(1) for bits in bit_fcs]
        torch_bits = torch.cat(torch_bits, 1) # bs, seq_len, class_num
        return torch_bits

if __name__ == '__main__':
    net = Segnet()
    x = torch.rand(2, 1, 128*3000)
    seg_y, pred_y = net(x)
    print(seg_y.size(), pred_y.size())
    for n, p in net.named_parameters():
        print(n, torch.numel(p))
    print('total param num: {}'.format(
        sum(torch.numel(p) for p in net.parameters())
        )
    )
