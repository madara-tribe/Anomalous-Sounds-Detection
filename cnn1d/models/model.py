from torch import nn
from .layers import GeM
import torch

class CNN1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=16),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=8),
            GeM(kernel_size=8),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=4),
            GeM(kernel_size=6),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=4),
            GeM(kernel_size=4),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.liner = nn.Linear(16, 512)
        self.lstm1 = nn.GRU(16, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.GRU(256*2, 256, bidirectional=True, batch_first=True)
        self.residual = nn.Sequential(
                nn.LayerNorm([256, 512]),
                nn.Dropout(0.25),
                nn.SiLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(131072, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.LayerNorm([256]),
            nn.Dropout(0.25),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 256),
            nn.LayerNorm([256]),
            nn.Dropout(0.25),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, x, pos=None):
        #xs_ = x[:, 1, :].unsqueeze(1)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        res = self.cnn6(x)
        x, _ = self.lstm1(res)
        x, _ = self.lstm2(x)
        x = self.residual(x)
        x = x + self.liner(res)
        #print("8", x.shape)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

