from torch import nn

class UNET1D(nn.Module):
    def __init__(self, in_channels):
        super(UNET1D, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=16, stride=2, padding=7),  # Reducción: /2
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),  # Reducción: /4
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, stride=2, padding=3),  # Reducción: /8
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )

        self.enc4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),  # Reducción: /16
            nn.BatchNorm1d(512),
            nn.SiLU(),
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1, output_padding=0),  # Ampliación: *8
            nn.BatchNorm1d(256),
            nn.SiLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3, output_padding=0),  # Ampliación: *4
            nn.BatchNorm1d(128),
            nn.SiLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3, output_padding=0),  # Ampliación: *2
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(64, in_channels, kernel_size=16, stride=2, padding=7, output_padding=0),  # Ampliación: *1 (original size)
            nn.SiLU()
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        # Decoder
        x5 = self.dec1(x4)
        x5 = x5 + x3
        x6 = self.dec2(x5)
        x6 = x6 + x2
        x7 = self.dec3(x6)
        x7 = x7 + x1
        x8 = self.dec4(x7)
        return x8

