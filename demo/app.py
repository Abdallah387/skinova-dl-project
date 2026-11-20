"""Gradio demo: dermoscopic image + age/sex/site -> class probabilities.

    python demo/app.py --ckpt checkpoints/fusion_seed0.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import numpy as np
import pandas as pd
import torch

from src.datasets.data import MetaEncoder, SEXES, preprocess_pil
from src.models import build_model

DISCLAIMER = ("Educational prototype from a university course project. It is NOT a medical device and must not "
              "be used for diagnosis. Consult a dermatologist.")
NAMES = {"akiec": "Actinic keratosis / intraepithelial carcinoma", "bcc": "Basal cell carcinoma",
         "bkl": "Benign keratosis-like lesion", "df": "Dermatofibroma", "mel": "Melanoma",
         "nv": "Melanocytic nevus", "vasc": "Vascular lesion"}

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/fusion_seed0.pt")
ap.add_argument("--share", action="store_true")
args = ap.parse_args()

ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
enc = MetaEncoder(ckpt["meta_encoder"])
cfg = {**ckpt["cfg"], "model": {**ckpt["cfg"]["model"], "pretrained": False}}  # weights come from the checkpoint
model = build_model(cfg, enc.dim)
model.load_state_dict(ckpt["state_dict"])
model.eval()
uses_meta = cfg["model"]["type"] == "fusion"
size = cfg["data"]["img_size"]
sites = [s for s in enc.state["sites"] if s != "unknown"]


def predict(image, age, sex, site):
    if image is None:
        return {}
    x = preprocess_pil(image, size)
    row = pd.DataFrame([{"age": age if age else np.nan, "sex": sex, "localization": site}])
    meta = torch.from_numpy(enc.transform(row))
    with torch.no_grad():
        p = torch.softmax(model(x, meta), 1)[0].numpy()
    return {f"{c} - {NAMES[c]}": float(p[i]) for i, c in enumerate(ckpt["classes"])}


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Dermoscopic image"),
            gr.Number(label="Age (leave 0 if unknown)", value=0),
            gr.Dropdown(SEXES, value="unknown", label="Sex"),
            gr.Dropdown(sites + ["unknown"], value="unknown", label="Anatomical site")],
    outputs=gr.Label(num_top_classes=7, label="Predicted class probabilities"),
    title="Skinova - skin lesion classifier",
    description=DISCLAIMER + ("" if uses_meta else " (This checkpoint is image-only; metadata is ignored.)"),
)

if __name__ == "__main__":
    demo.launch(share=args.share)
