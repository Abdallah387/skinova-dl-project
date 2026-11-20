"""Shared evaluation used by every model."""
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

from src.datasets.data import CLASSES


def compute_metrics(y_true, probs):
    K = len(CLASSES)
    y_pred = probs.argmax(1)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(K), average=None, zero_division=0)
    aucs = []
    for k in range(K):
        t = y_true == k
        aucs.append(roc_auc_score(t, probs[:, k]) if 0 < t.sum() < len(t) else float("nan"))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "macro_auc": float(np.nanmean(aucs)),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                          "auc": float(aucs[i]), "support": int(sup[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(K)).tolist(),
    }


def bootstrap_ci(y_true, probs, n=1000, seed=0):
    """95% bootstrap interval of macro-F1 over test images."""
    rng = np.random.default_rng(seed)
    y_pred = probs.argmax(1)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        vals.append(f1_score(y_true[i], y_pred[i], labels=range(len(CLASSES)), average="macro", zero_division=0))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for img, meta, y in loader:
        logits = model(img.to(device), meta.to(device))
        probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)
