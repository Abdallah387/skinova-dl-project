#!/usr/bin/env bash
# Full experiment protocol. Run a stage at a time on Kaggle:  bash scripts/run_all.sh <stage>
# Stages: sweep | final | strategy | scratch | report
set -e
SEEDS="0 1 2"
run() { python -m src.train "$@"; }

case "$1" in
sweep)   # hyperparameter study, seed 0 only, selection by VALIDATION macro-F1 (see experiments/*/seed0/results.json)
  for lr in 1e-3 3e-4 1e-4; do
    run --config configs/cnn.yaml    --seed 0 --set name=cnn_lr$lr    train.lr=$lr
    run --config configs/fusion.yaml --seed 0 --set name=fusion_lr$lr train.lr=$lr
  done
  for lr in 1e-4 5e-5 2e-5; do
    run --config configs/vit_pretrained.yaml --seed 0 --set name=vit_lr$lr train.lr=$lr
  done
  for bs in 32 128; do
    run --config configs/fusion.yaml --seed 0 --set name=fusion_bs$bs train.batch_size=$bs
  done
  for wd in 1e-5 1e-2; do
    run --config configs/fusion.yaml --seed 0 --set name=fusion_wd$wd train.weight_decay=$wd
  done ;;
final)   # baseline + the three deep models (+ the metadata ablation = cnn), 3 seeds each
  for s in $SEEDS; do
    for c in svm cnn vit_pretrained fusion; do run --config configs/$c.yaml --seed $s; done
  done ;;
strategy) # data-strategy study on the fusion model (H3); "fusion" itself is the augment+weights setting
  for s in $SEEDS; do
    run --config configs/fusion.yaml --seed $s --set name=fusion_plain       train.augment=false train.class_weights=false
    run --config configs/fusion.yaml --seed $s --set name=fusion_aug_only    train.class_weights=false
    run --config configs/fusion.yaml --seed $s --set name=fusion_weights_only train.augment=false
    run --config configs/fusion.yaml --seed $s --set name=fusion_sampler     train.class_weights=false train.sampler=true
    run --config configs/fusion.yaml --seed $s --set name=fusion_focal       train.loss=focal
  done ;;
scratch) # pretrained vs scratch ViT
  for s in $SEEDS; do run --config configs/vit_scratch.yaml --seed $s; done ;;
report)
  python -m src.aggregate
  for m in cnn fusion vit_pretrained; do python -m src.error_analysis --name $m --config configs/$m.yaml; done ;;
*) echo "usage: $0 sweep|final|strategy|scratch|report"; exit 1 ;;
esac
