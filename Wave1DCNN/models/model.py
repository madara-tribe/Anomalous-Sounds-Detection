from torch import nn
from .layers import GeM
from .layers import MultiLayerPerceptron as MLP
        
class OneDimensionalCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=16),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=8),
            GeM(kernel_size=8),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=4),
            GeM(kernel_size=6),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.conv_block5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4),
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.conv_block6 = nn.Sequential(
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
        self.mlp_head1 = MLP(in_dim=131072, emb_dim=64, out_dim=256)
        self.mlp_head2 = MLP(in_dim=256, emb_dim=64, out_dim=256)
        self.mlp_head3 = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        res = self.conv_block6(x)
        x, _ = self.lstm1(res)
        x, _ = self.lstm2(x)
        x = self.residual(x)
        x = x + self.liner(res)
        x = x.flatten(start_dim=1)
        x = self.mlp_head1(x)
        x = self.mlp_head2(x)
        x = self.mlp_head3(x)
        return x


