"""Collect experiments/*/seed*/results.json into mean +- SD tables (markdown + csv).

    python -m src.aggregate
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.data import CLASSES

METRICS = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc"]
COST = ["params", "model_size_mb", "train_time_s", "inference_ms_per_image", "peak_gpu_mem_mb"]


def fmt(x):
    return f"{np.mean(x):.4f} ± {np.std(x):.4f}"


def main(root="experiments"):
    runs = {}
    for f in sorted(Path(root).glob("*/seed*/results.json")):
        r = json.loads(f.read_text())
        runs.setdefault(r["name"], []).append(r)

    main_rows, cost_rows, class_rows = [], [], []
    for name, rs in runs.items():
        row = {"model": name, "seeds": len(rs), "val_macro_f1": fmt([r["best_val_macro_f1"] for r in rs])}
        row.update({m: fmt([r["test"][m] for r in rs]) for m in METRICS})
        main_rows.append(row)
        cost_rows.append({"model": name, **{c: f"{np.mean([r['cost'][c] for r in rs]):.2f}" for c in COST},
                          "epochs_run": f"{np.mean([r['cost']['epochs_run'] for r in rs]):.1f}",
                          "hardware": rs[0]["cost"]["hardware"]})
        for c in CLASSES:
            class_rows.append({"model": name, "class": c,
                               "recall": fmt([r["test"]["per_class"][c]["recall"] for r in rs]),
                               "f1": fmt([r["test"]["per_class"][c]["f1"] for r in rs])})

    tables = {"main_results": pd.DataFrame(main_rows), "computational_cost": pd.DataFrame(cost_rows),
              "per_class": pd.DataFrame(class_rows)}
    md = []
    for name, t in tables.items():
        t.to_csv(Path(root) / f"{name}.csv", index=False)
        md += [f"## {name}", "", t.to_markdown(index=False), ""]
    (Path(root) / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
