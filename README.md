# Skinova — Multimodal Skin-Lesion Diagnosis (HAM10000)

Deep Learning (AI 403) final project, Pharos University in Alexandria. **Team Skinova:** Aya Abdo, Rana Shehata,
Areej Alaa, Abdallah Moustafa, Amr Essam. **Idea chosen:** #1 Multimodal skin-lesion diagnosis (image + tabular → class).

Task: classify a dermoscopic image (plus age, sex, anatomical site) into 7 classes: akiec, bcc, bkl, df, mel, nv, vasc.
Academic decision-support prototype, not a medical device.

> Best result: multimodal EfficientNetB0 + LSTM (image + metadata), 85.1% test accuracy and 0.718 macro-F1 (single run, seed 42).
> The notebook that produced all reported results is `notebooks/skinova_final_project.ipynb` (Kaggle, GPU T4):
> data split, SVM, CNN, ViT and the multimodal CNN+GRU/LSTM models. Trained model files are on a shared drive: _add link here_.

## Models

| Config | Family | What it is |
|---|---|---|
| `configs/svm.yaml` | baseline | HOG + colour stats → RBF-SVM (class-balanced) |
| `configs/cnn.yaml` | classic DL | custom residual CNN written from scratch (`src/models/cnn.py`), image only |
| `configs/vit_pretrained.yaml` | modern DL | `timm` ViT-S/16 (ImageNet pretrained) fine-tuned |
| `configs/fusion.yaml` | advanced | same CNN encoder + MLP on metadata, concatenated before the classifier |
| `configs/vit_scratch.yaml` | study | ViT-S/16 without pretraining (pretrained-vs-scratch) |

Components written by us: residual CNN, fusion network, focal loss / class weights (`src/losses.py`), shared evaluation.
**Ablation (H1):** removing the metadata branch from `fusion` gives exactly the `cnn` run (same encoder, same head width).

## Environment (Kaggle)

1. New notebook → Add data → search **"Skin Cancer MNIST: HAM10000"** (`kmader/skin-cancer-mnist-ham10000`).
2. Settings: Accelerator **GPU T4 x1**, Internet **On** (needed once for the pretrained ViT weights).
3. In a cell:

```bash
!git clone https://github.com/Abdallah387/skinova-dl-project.git skinova && cd skinova && pip install -q timm scikit-image tabulate gradio
```

Data root is already `/kaggle/input/skin-cancer-mnist-ham10000` in `configs/base.yaml`. Locally, override with
`--set data.root=PATH`. Outputs (`experiments/`, `checkpoints/`, `cache/`) are written to the working directory;
on Kaggle run from `/kaggle/working/skinova` and download the results before the session ends (or "Save Version").

## Run

```bash
python data/prepare.py                                   # split + image cache (also done automatically)
python -m src.train --config configs/fusion.yaml --seed 0   # one model, one seed
bash scripts/run_all.sh sweep      # hyperparameter study (seed 0, pick by validation macro-F1)
bash scripts/run_all.sh final      # svm, cnn, vit_pretrained, fusion x 3 seeds
bash scripts/run_all.sh strategy   # data-strategy study (H3)
bash scripts/run_all.sh scratch    # ViT pretrained vs scratch
bash scripts/run_all.sh report     # tables (experiments/summary.md) + error analysis figures
python demo/app.py --ckpt checkpoints/fusion_seed0.pt    # Gradio demo (--share on Kaggle)
```

If you change the best learning rate after the sweep, pass it to the finals (`--set train.lr=...`) for **every** model
so the tuning budget stays equal.

## Protocol (what makes the comparison fair)

* One split for all models: `StratifiedGroupKFold` grouped by `lesion_id` → 70 / 10 / 20 train / val / test, saved in
  `cache/split.csv` (seed 42). The assertion in `make_split` fails if a lesion appears in two splits.
* Same preprocessing (224×224, ImageNet normalisation), metadata encoder fitted on the training split only.
* Augmentation and class weights touch the training split only; test is used once, after model selection on val macro-F1.
* Final configs run with seeds 0, 1, 2; tables report mean ± SD. Test macro-F1 also gets a 95 % bootstrap interval.
* Cost per run: parameters, checkpoint size, training time, single-image latency, peak GPU memory, hardware.

## Layout

```
configs/   one YAML per experiment (base.yaml holds shared defaults)
data/      prepare.py (no raw data)
src/       datasets/data.py, models/, losses.py, train.py, evaluate.py, aggregate.py, error_analysis.py
scripts/   run_all.sh
demo/      app.py (Gradio)
experiments/  results.json, test_probs.npy per run; summary.md, error_analysis/
checkpoints/  .pt files (upload to a shared drive and link here)
report/  contributions/
```

## Credits

* HAM10000: Tschandl, Rosendahl, Kittler, *Sci. Data* 5, 180161 (2018); licence CC BY-NC 4.0 (non-commercial, academic use).
  Kaggle mirror by K. Mader.
* ViT: Dosovitskiy et al. 2021; weights and models via `timm` (Ross Wightman).
* Residual blocks follow He et al. 2016; focal loss follows Lin et al. 2017 (re-implemented in `src/losses.py`).
