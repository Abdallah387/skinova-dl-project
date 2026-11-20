"""Multimodal fusion: CNN image encoder + MLP metadata encoder, concatenated before the classifier."""
import torch
import torch.nn as nn

from src.models.cnn import CNNEncoder


class FusionNet(nn.Module):
    def __init__(self, meta_dim, num_classes=7, widths=(32, 64, 128, 256), meta_hidden=64, dropout=0.3):
        super().__init__()
        self.encoder = CNNEncoder(widths)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden), nn.BatchNorm1d(meta_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(meta_hidden, meta_hidden), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.encoder.out_dim + meta_hidden, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, img, meta):
        return self.head(torch.cat([self.encoder(img), self.meta_mlp(meta)], dim=1))
