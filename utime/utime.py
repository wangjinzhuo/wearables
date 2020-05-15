import torch
import torch.nn as nn

class Utime(nn.Module):
    def __init__(self):
        super(Utime, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), # padding = (kernel_size - 1)/2, stride=1
            nn.Conv1d(16, 16, 5, padding=2),
            nn.MaxPool1d(10),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.MaxPool1d(8),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.MaxPool1d(6),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.Conv1d(256, 256, 5, padding=2),
        )
        self.decoder_b1 = nn.Sequential(
            nn.Upsample(216),
            nn.Conv1d(256, 128, 5, padding=2),
        )
        self.decoder_b2 = nn.Sequential(
            nn.Conv1d(256, 128, 5, padding=2),
            nn.Conv1d(128, 128, 5, padding=2),
            nn.Upsample(1296),
            nn.Conv1d(128, 64, 6, padding=2), # careful when filter size is even
        )
        self.decoder_b3 = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.Upsample(10368),
            nn.Conv1d(64, 32, 8, padding=2),
        )
        self.decoder_b4 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2),
            nn.Conv1d(32, 32, 5, padding=2),
            nn.Upsample(103680),
            nn.Conv1d(32, 16, 10, padding=2),
        )
        self.decoder_b5 = nn.Sequential(
            nn.Conv1d(32, 16, 5, padding=2),
            nn.Conv1d(16, 16, 5, padding=2),
        )
        self.segment_classifier = nn.Sequential(
            nn.Conv1d(16, 5, 1),
            nn.ConstantPad1d(660, 0), # 103680 + 660*2 = 105000
        )
        self.final_conv = nn.Conv1d(5,5,1)

    def forward(self, x):
        res = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            if i in {1, 4, 7, 10}:
                res.append(x)
        x = self.decoder_b1(x)
        x = crop_conc(x, res[3]) # the second one is larger - 1296

        x = self.decoder_b2(x) # output 1295,  expected BS, 64, 1296
        x = manually_pad(x, 1296)
        x = crop_conc(x, res[2])

        x = self.decoder_b3(x) # output 10365, expected BS, 64, 10368
        x = manually_pad(x, 10368)
        x = crop_conc(x, res[1])

        x = self.decoder_b4(x) # 103675, expected BS, 64, 103680
        x = manually_pad(x, 103680)
        x = crop_conc(x, res[0])

        x = self.decoder_b5(x) # expected BS, 16, 103680

        x = self.segment_classifier(x)

        x = x.view(-1, 5, 3000, 35)

        x = torch.mean(x, dim=2)
        x = self.final_conv(x)

        return x

def crop_conc(x1, x2):
    # crop x1 to comply with x2's dim and concatenate them
    # skip-connection implementation
    crop_x2 = x2[:,:,:x1.size()[2]]
    return torch.cat((x1, crop_x2), 1)


def manually_pad(x, dim):
    # e.g. x's dim is [bs, 128, 1295] required dim = 1296
    tmp = torch.zeros(x.size()[0], x.size()[1], dim, device=x.device)
    tmp[:, :, :x.size()[2]] = x[:, :, :x.size()[2]]
    return tmp

if __name__ == '__main__':
    model = Utime()
    x = torch.rand(2, 1, 3000*35)
    y = model(x)
    print(y.size())
    print("total param num is: {}".format(
        sum(torch.numel(p) for p in model.parameters())
        )
    )

