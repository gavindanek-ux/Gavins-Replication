import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    """DeepLOB (Zhang et al. 2019), per the layer spec in Mejia Becerra's
    dissertation Table F.2: 3 composite conv blocks -> inception -> LSTM -> FC.

    Expects input already shaped (N, 1, 100, 40) -- matches
    DeepLOBDataset's per-sample (1, 100, 40) from gavin_ddb_pipeline.py, so
    no channel dim gets added here (deeplob_rep_inference's version does
    that itself in forward(), because its own Dataset hands out plain
    (100, 40) samples -- the two conventions aren't interchangeable).
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()

        # CConv(2,1)_2: (N,1,100,40) -> (N,32,94,20). The (1,2) stride-(1,2)
        # conv groups price+volume pairs first (40 -> 20 columns).
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )

        # CConv(2,2)_2: (N,32,94,20) -> (N,32,88,10). Same (1,2) stride-(1,2)
        # shape again, now grouping the ask/bid pairs (20 -> 10 levels).
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )

        # CConv(10)_1: (N,32,88,10) -> (N,32,82,1). Collapses the remaining
        # 10 price levels down to 1 with a single (1,10) filter.
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )

        # Inception layer IL: 3 branches, 64 channels each, concatenated to
        # 192 (the dissertation's own summary table says 194, but that
        # contradicts its own formulas -- 3 branches x 64 -- and the code
        # below; see the shape check at the bottom of this file).
        self.inception1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inception2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inception3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )

        # Perm(3,2) swaps channel and time axes so the LSTM sees (N, time,
        # features) instead of (N, features, time, 1).
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1,
                             batch_first=True)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x1 = self.inception1(x)
        x2 = self.inception2(x)
        x3 = self.inception3(x)
        x = torch.cat([x1, x2, x3], dim=1)

        x = x.permute(0, 2, 1, 3)
        x = x.squeeze(3)

        x, _ = self.lstm(x)
        x = x[:, -1, :]  # last timestep only, matches LSTM64's a_<H>

        return self.fc(x)


if __name__ == '__main__':
    # Shape-only smoke test against dissertation Table F.2 (batch=32,
    # K=3 classes here instead of the table's K=1 scalar example).
    model = DeepLOB(n_classes=3)
    x = torch.randn(32, 1, 100, 40)

    with torch.no_grad():
        h = model.conv1(x); print('conv1  ', tuple(h.shape), 'expected (32, 32, 94, 20)')
        h = model.conv2(h); print('conv2  ', tuple(h.shape), 'expected (32, 32, 88, 10)')
        h = model.conv3(h); print('conv3  ', tuple(h.shape), 'expected (32, 32, 82, 1)')
        h1, h2, h3 = model.inception1(h), model.inception2(h), model.inception3(h)
        h = torch.cat([h1, h2, h3], dim=1)
        print('inception', tuple(h.shape), 'expected (32, 192, 82, 1) -- settles the 192 vs 194 question')

        out = model(x)
        print('output ', tuple(out.shape), 'expected (32, 3)')

    n_params = sum(p.numel() for p in model.parameters())
    print('total trainable params:', n_params, '(dissertation Table F.2: 143,777 for K=1)')
