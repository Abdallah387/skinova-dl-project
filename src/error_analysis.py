"""Confusion matrix, top confusion pairs and misclassified examples for one experiment.

    python -m src.error_analysis --name fusion --config configs/fusion.yaml
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.data import CLASSES
from src.train import load_config, prepare_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default="experiments")
    ap.add_argument("--n_examples", type=int, default=12)
    args = ap.parse_args()

    out = Path(args.root) / "error_analysis"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    df, images, _, _ = prepare_data(cfg)

    seeds = sorted((Path(args.root) / args.name).glob("seed*"))
    cms = np.array([json.loads((s / "results.json").read_text())["test"]["confusion_matrix"] for s in seeds])
    cm = cms.mean(0)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), CLASSES, rotation=45)
    ax.set_yticks(range(7), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{args.name}: row-normalised confusion (mean of {len(seeds)} seeds)")
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_confusion.png", dpi=150)
    plt.close(fig)

    # first seed's per-image predictions for concrete examples
    probs = np.load(seeds[0] / "test_probs.npy")
    test_idx = np.load(seeds[0] / "test_idx.npy")
    y, pred, conf = df["label"].values[test_idx], probs.argmax(1), probs.max(1)
    wrong = np.where(y != pred)[0]
    rows = []
    for (t, p), g in pd.DataFrame({"t": y[wrong], "p": pred[wrong], "k": wrong}).groupby(["t", "p"]):
        k = g["k"].values
        rows.append({"error_type": f"{CLASSES[t]} -> {CLASSES[p]}", "count": len(k),
                     "mean_confidence": float(conf[k].mean()), "example_image": df["image_id"].values[test_idx[k[np.argmax(conf[k])]]]})
    table = pd.DataFrame(rows).sort_values("count", ascending=False)
    table.to_csv(out / f"{args.name}_error_types.csv", index=False)
    print(table.head(10).to_string(index=False))

    # most confident mistakes, plus missed melanomas
    order = wrong[np.argsort(-conf[wrong])][: args.n_examples // 2]
    mel = wrong[(y[wrong] == CLASSES.index("mel"))]
    order = np.concatenate([order, mel[np.argsort(-conf[mel])][: args.n_examples - len(order)]]).astype(int)
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(order) / cols)), cols, figsize=(3 * cols, 3.2 * np.ceil(len(order) / cols)))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, k in zip(np.ravel(axes), order):
        ax.imshow(images[test_idx[k]])
        ax.set_title(f"true {CLASSES[y[k]]} / pred {CLASSES[pred[k]]} ({conf[k]:.2f})", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_examples.png", dpi=150)


if __name__ == "__main__":
    main()
