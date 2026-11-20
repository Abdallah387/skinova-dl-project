"""Build the leakage-free split and the resized image cache once (train.py also does this on demand).

    python data/prepare.py --config configs/base.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import load_config, prepare_data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    df, images, idx, enc = prepare_data(cfg)
    print({k: len(v) for k, v in idx.items()}, "images", images.shape, "meta dim", enc.dim)
    print(df.groupby(["split", "dx"]).size().unstack(0))
