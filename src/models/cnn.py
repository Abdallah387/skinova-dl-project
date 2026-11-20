"""Custom residual CNN written from scratch (no torchvision/timm backbone)."""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or c_in != c_out:
            self.short = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + (x if self.short is None else self.short(x)))


class CNNEncoder(nn.Module):
    """Stem (stride 2) + 4 residual stages (stride 2 each) + global average pooling -> widths[-1] features."""

    def __init__(self, widths=(32, 64, 128, 256)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        stages, c = [], widths[0]
        for w in widths:
            stages.append(ResBlock(c, w, 2))
            c = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = widths[-1]

    def forward(self, x):
        return self.pool(self.stages(self.stem(x))).flatten(1)


class CNNClassifier(nn.Module):
    """Image-only classifier. The `meta` argument is accepted and ignored so all models share one signature."""

    def __init__(self, num_classes=7, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(widths)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, img, meta=None):
        return self.head(self.encoder(img))
