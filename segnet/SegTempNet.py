import torch
import torch.nn as nn
import torch.nn.functional as F

class SegTempNet(nn.Module):
    '''
    share backbone module for feature extraction
    '''
    def __init__(self, feat_dim=128, seq_len=128, num_class=5, **kwargs):
        super(SegTempNet, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, 16, 8),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 8, 4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 8, 4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 4, 2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(256)),
            nn.Conv1d(256, 64, 16, 4),
            nn.BatchNorm1d(64),
            nn.Flatten(),
            nn.Linear(3904, 512),
            nn.Linear(512, feat_dim),
        )

    def forward(self, x):
        feature_map = self.backbone(x)
        return feature_map

