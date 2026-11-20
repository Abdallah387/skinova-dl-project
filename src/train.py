"""Train / evaluate one configuration with one seed.

    python -m src.train --config configs/fusion.yaml --seed 0 [--set train.lr=1e-3 name=fusion_lr1e-3]
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.datasets.data import (CLASSES, MetaEncoder, SkinDataset, build_image_cache, load_metadata, make_split)
from src.evaluate import bootstrap_ci, compute_metrics, predict
from src.losses import FocalLoss, class_weights
from src.models import build_model, count_params


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "_base_" in cfg:
        cfg = deep_merge(load_config(path.parent / cfg.pop("_base_")), cfg)
    for item in overrides:
        key, val = item.split("=", 1)
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_data(cfg):
    d = cfg["data"]
    cache = Path(d["cache_dir"])
    df = load_metadata(d["root"])
    df = make_split(df, d["split_seed"], cache / "split.csv")
    images = build_image_cache(df, d["img_size"], cache / f"images_{d['img_size']}.npy")
    idx = {s: np.where(df["split"] == s)[0] for s in ("train", "val", "test")}
    enc = MetaEncoder().fit(df.iloc[idx["train"]])
    return df, images, idx, enc


def make_loaders(cfg, df, images, idx, enc):
    t, d = cfg["train"], cfg["data"]
    meta, labels = enc.transform(df), df["label"].values
    pin = torch.cuda.is_available()
    kw = dict(num_workers=d["num_workers"], pin_memory=pin, persistent_workers=d["num_workers"] > 0)
    train_ds = SkinDataset(images, meta, labels, idx["train"], augment=t["augment"], size=d["img_size"])
    sampler = None
    if t["sampler"]:
        counts = np.bincount(labels[idx["train"]], minlength=len(CLASSES))
        sampler = WeightedRandomSampler(1.0 / counts[labels[idx["train"]]], len(idx["train"]), replacement=True)
    train_dl = DataLoader(train_ds, t["batch_size"], shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    mk = lambda s: DataLoader(SkinDataset(images, meta, labels, idx[s], size=d["img_size"]), 128, shuffle=False, **kw)
    return train_dl, mk("val"), mk("test"), labels


def hardware():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"


def inference_ms(model, loader, device, n=50):
    """Median single-image latency (batch size 1) in milliseconds."""
    model.eval()
    img, meta, _ = next(iter(loader))
    img, meta = img[:1].to(device), meta[:1].to(device)
    times = []
    with torch.no_grad():
        for i in range(n + 5):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(img, meta)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 5:
                times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run_deep(cfg, seed, df, images, idx, enc, out_dir):
    t = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dl, val_dl, test_dl, labels = make_loaders(cfg, df, images, idx, enc)
    model = build_model(cfg, enc.dim).to(device)

    weight = class_weights(labels[idx["train"]], power=t["class_weight_power"]).to(device) if t["class_weights"] else None
    criterion = FocalLoss(t["focal_gamma"], weight) if t["loss"] == "focal" else nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_f1, best_state, bad, history = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(t["epochs"]):
        model.train()
        loss_sum, n = 0.0, 0
        for img, meta, y in train_dl:
            img, meta, y = img.to(device, non_blocking=True), meta.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(img, meta)
            loss = criterion(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum, n = loss_sum + loss.item() * len(y), n + len(y)
        sched.step()
        vp, vy = predict(model, val_dl, device)
        val_f1 = f1_score(vy, vp.argmax(1), labels=range(len(CLASSES)), average="macro", zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / n, "val_macro_f1": float(val_f1)})
        print(f"epoch {epoch + 1:02d} loss {loss_sum / n:.4f} val_macro_f1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, bad = val_f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= t["patience"]:
                break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    probs, y_test = predict(model, test_dl, device)
    ckpt = Path("checkpoints") / f"{cfg['name']}_seed{seed}.pt"
    ckpt.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "cfg": cfg, "meta_encoder": enc.state, "classes": CLASSES}, ckpt)
    cost = {"params": count_params(model), "model_size_mb": ckpt.stat().st_size / 1e6, "train_time_s": train_time,
            "epochs_run": len(history), "inference_ms_per_image": inference_ms(model, test_dl, device),
            "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0}
    return probs, y_test, best_f1, history, cost


def run_svm(cfg, seed, df, images, idx):
    from src.models.svm_baseline import SVMBaseline

    labels = df["label"].values
    t0 = time.time()
    svm = SVMBaseline(C=cfg["model"]["C"]).fit(images[idx["train"]], labels[idx["train"]])
    train_time = time.time() - t0
    val_p = svm.probs(images[idx["val"]])
    val_f1 = f1_score(labels[idx["val"]], val_p.argmax(1), labels=range(len(CLASSES)), average="macro", zero_division=0)
    probs, y_test = svm.probs(images[idx["test"]]), labels[idx["test"]]
    t1 = time.perf_counter()
    for i in idx["test"][:30]:
        svm.probs(images[i:i + 1])
    cost = {"params": int(svm.model[-1].support_vectors_.size), "model_size_mb": 0.0, "train_time_s": train_time,
            "epochs_run": 0, "inference_ms_per_image": (time.perf_counter() - t1) / 30 * 1000, "peak_gpu_mem_mb": 0.0}
    return probs, y_test, val_f1, [], cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="overrides such as train.lr=1e-3 name=run1")
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(args.seed)
    out_dir = Path(args.out) / cfg["name"] / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, images, idx, enc = prepare_data(cfg)
    print(f"{cfg['name']} seed {args.seed}: train {len(idx['train'])} val {len(idx['val'])} test {len(idx['test'])}")
    if cfg["model"]["type"] == "svm":
        probs, y_test, val_f1, history, cost = run_svm(cfg, args.seed, df, images, idx)
    else:
        probs, y_test, val_f1, history, cost = run_deep(cfg, args.seed, df, images, idx, enc, out_dir)

    metrics = compute_metrics(y_test, probs)
    metrics["macro_f1_ci95"] = bootstrap_ci(y_test, probs)
    result = {"name": cfg["name"], "seed": args.seed, "best_val_macro_f1": float(val_f1), "test": metrics,
              "cost": {**cost, "hardware": hardware()}, "history": history, "cfg": cfg}
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    np.save(out_dir / "test_probs.npy", probs)
    np.save(out_dir / "test_idx.npy", idx["test"])
    print(f"TEST macro_f1 {metrics['macro_f1']:.4f} bal_acc {metrics['balanced_accuracy']:.4f} "
          f"acc {metrics['accuracy']:.4f} auc {metrics['macro_auc']:.4f}")


if __name__ == "__main__":
    main()
