"""Loss components written from scratch: class weighting and focal loss."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def class_weights(labels, num_classes=7, power=1.0):
    """Inverse-frequency weights (counts ** -power), rescaled to have mean 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = counts ** -power
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(nn.Module):
    """FL = -alpha_t (1 - p_t)^gamma log(p_t); alpha is an optional per-class weight vector."""

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        loss = -((1 - logp.exp()) ** self.gamma) * logp
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()
