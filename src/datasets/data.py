"""HAM10000 loading, leakage-free splitting, metadata encoding and the torch Dataset."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.transforms import v2

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SEXES = ["male", "female", "unknown"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_metadata(root):
    """Read HAM10000_metadata.csv and attach the image path and integer label."""
    root = Path(root)
    csv = next(root.rglob("HAM10000_metadata.csv"))
    df = pd.read_csv(csv)
    paths = {p.stem: str(p) for p in root.rglob("*.jpg")}
    df["path"] = df["image_id"].map(paths)
    if df["path"].isna().any():
        raise FileNotFoundError(f"{df['path'].isna().sum()} images listed in the csv were not found under {root}")
    df["label"] = df["dx"].map(CLASS_TO_IDX)
    return df.sort_values("image_id").reset_index(drop=True)


def make_split(df, seed=42, save_path=None):
    """70/10/20 train/val/test split; every lesion_id lives in exactly one split."""
    save_path = Path(save_path) if save_path else None
    if save_path is not None and save_path.exists():
        saved = pd.read_csv(save_path)
        return df.merge(saved[["image_id", "split"]], on="image_id", how="left")

    y, g = df["label"].values, df["lesion_id"].values
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval, test = next(outer.split(df, y, g))
    inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed)
    tr, va = next(inner.split(trainval, y[trainval], g[trainval]))

    split = np.empty(len(df), dtype=object)
    split[trainval[tr]], split[trainval[va]], split[test] = "train", "val", "test"
    df = df.assign(split=split)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        shared = set(df.loc[df.split == a, "lesion_id"]) & set(df.loc[df.split == b, "lesion_id"])
        assert not shared, f"lesion leakage between {a} and {b}"
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df[["image_id", "lesion_id", "dx", "split"]].to_csv(save_path, index=False)
    return df


class MetaEncoder:
    """Age (z-scored + missing flag), sex and anatomical site (one-hot). Fitted on the training split only."""

    def __init__(self, state=None):
        self.state = state

    def fit(self, df):
        age = df["age"].astype(float)
        median = float(age.median())
        filled = age.fillna(median)
        sites = sorted(df["localization"].dropna().unique().tolist()) + ["unknown"]
        self.state = dict(age_median=median, age_mean=float(filled.mean()), age_std=float(filled.std()), sites=sites)
        return self

    @property
    def dim(self):
        return 2 + len(SEXES) + len(self.state["sites"])

    def transform(self, df):
        s = self.state
        age = df["age"].astype(float)
        age_z = (age.fillna(s["age_median"]) - s["age_mean"]) / s["age_std"]
        sex = df["sex"].fillna("unknown")
        sex = sex.where(sex.isin(SEXES), "unknown")
        site = df["localization"].fillna("unknown")
        site = site.where(site.isin(s["sites"]), "unknown")
        cols = [age_z.values, age.isna().values]
        cols += [(sex == x).values for x in SEXES] + [(site == x).values for x in s["sites"]]
        return np.stack(cols, 1).astype(np.float32)


def build_image_cache(df, size, path):
    """Decode + resize every image once into a uint8 array (N, size, size, 3) in df order."""
    path = Path(path)
    if path.exists():
        return np.load(path, mmap_mode="r")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty((len(df), size, size, 3), dtype=np.uint8)
    for i, p in enumerate(df["path"]):
        arr[i] = np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))
    np.save(path, arr)
    return np.load(path, mmap_mode="r")


def train_transform(size):
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(30),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.02),
    ])


class SkinDataset(Dataset):
    """Returns (normalised image, metadata vector, label). Augmentation is only applied when augment=True."""

    def __init__(self, images, meta, labels, idx, augment=False, size=224):
        self.images, self.meta, self.labels, self.idx = images, meta, labels, np.asarray(idx)
        self.aug = train_transform(size) if augment else None
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        img = torch.from_numpy(np.array(self.images[j])).permute(2, 0, 1)
        if self.aug is not None:
            img = self.aug(img)
        img = (img.float() / 255.0 - self.mean) / self.std
        return img, torch.from_numpy(self.meta[j]), int(self.labels[j])


def preprocess_pil(img, size=224):
    """Same resize + normalisation as the cache, for the demo. Returns (1, 3, size, size)."""
    arr = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR))
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)
