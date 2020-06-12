import torch
import torch.nn as nn
import torch.nn.functional as F

Fs = 200

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)

    def forward(self, x):
        # set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda() #RuntimeError: Input and hidden tensors are not at the same device
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()

        # forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        return out

class DeepSleepNet(nn.Module):

    def __init__(self, ch=1):
        super(DeepSleepNet, self).__init__()
        self.features_s = nn.Sequential(
            nn.Conv1d(ch, 64, 50, 6),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(),
            nn.Conv1d(64, 128, 6),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 6),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_l = nn.Sequential(
            nn.Conv1d(ch, 64, 400, 50),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(),
            nn.Conv1d(64, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_seq = nn.Sequential(
            BiLSTM(7296, 512, 2),
        )
        self.res = nn.Linear(7296, 1024)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 5),
        )

    def forward(self, x):
        x_s = self.features_s(x)
        x_l = self.features_l(x)
        x_s = x_s.flatten(1,2)
        x_l = x_l.flatten(1,2)
        x = torch.cat((x_s, x_l),1) # [bs, 7296]
        x_seq = x.unsqueeze(1)
        x_blstm = self.features_seq(x_seq) # [bs, 1, 1024]
        x_blstm = torch.squeeze(x_blstm, 1)
        x_res = self.res(x)
        x = torch.mul(x_res, x_blstm)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    ch_num = 1
    batch_size = 3
    net = DeepSleepNet(ch=ch_num)
    net = net.cuda()
    inputs = torch.rand(batch_size, ch_num, int(Fs*30))
    inputs = inputs.cuda()
    outputs = net(inputs)
    print(outputs.size())
    params = list(net.parameters())
    print("total param num is: {}".format(
        sum(torch.numel(p) for p in params)
        )
    )
    a, b, c, d, e  = 0, 0, 0, 0, 0
    for name, param in net.named_parameters():
        if '_l' in name:
            a += torch.numel(param)
        if '_s' in name:
            b += torch.numel(param)
        if 'seq' in name:
            c += torch.numel(param)
        if 'res' in name:
            d += torch.numel(param)
        if 'class' in name:
            e += torch.numel(param)
    print('l: ', a, 's: ', b, 'seq: ', c, 'res: ', d, 'cls: ', e)
