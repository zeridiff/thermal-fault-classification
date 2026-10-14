"""
noise_robustness.py  --  THE EXPERIMENT THAT MAKES THIS WORK YOUR OWN.
======================================================================

The reference paper measures what happens when Gaussian and speckle noise are
added to the IR images at TEST time. Its accuracy falls from 100% to roughly
88% at best, because the models were only ever trained on clean thermograms.

This script asks a different question:

    If noise is injected during TRAINING as well, how much of that degradation
    disappears - and does it cost anything on clean images?

To answer it you need two trained runs, evaluated on exactly the same corrupted
test images:

    python train.py --name noiseaug                 (noise injection ON)
    python train.py --name geomonly --no-noise-aug  (geometry only = baseline)
    python noise_robustness.py --runs noiseaug geomonly

Output
------
    results/noise_robustness/noise_sweep.csv    full accuracy table
    results/noise_robustness/noise_sweep.png    accuracy vs severity, per noise
    results/noise_robustness/summary.txt        the numbers to quote in the report

Reading the result: the gap between the two curves at each severity is the
measured benefit of noise-augmented training. The value at severity 0 tells you
whether that robustness cost you anything on clean data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import data as D
import metrics as M
from engine import evaluate, get_device
from model import build_model


def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(num_classes=len(ckpt["class_names"]),
                        backbone=ckpt.get("backbone", config.BACKBONE),
                        pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def sweep_run(run_dir: Path, paths, labels, class_names, device, sweeps):
    """Evaluate every saved fold model over the noise grid, on its own test split."""
    ckpts = sorted(run_dir.glob("model_fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No model_fold*.pt inside {run_dir}. Run train.py with --name {run_dir.name} first."
        )

    rows = []
    for kind, severities in sweeps.items():
        for sev in severities:
            y_true_all, y_pred_all, y_prob_all = [], [], []
            for ck in ckpts:
                model, meta = load_checkpoint(ck, device)
                te_idx = np.asarray(meta["test_idx"])
                te_paths, te_labels = D.subset(paths, labels, te_idx)
                loader = D.make_loader(
                    te_paths, te_labels,
                    D.eval_transform(noise_kind=kind, severity=sev)
                )
                _, _, yt, yp, pb = evaluate(model, loader, None, device)
                y_true_all.append(yt); y_pred_all.append(yp); y_prob_all.append(pb)

            yt = np.concatenate(y_true_all)
            yp = np.concatenate(y_pred_all)
            pb = np.concatenate(y_prob_all)
            res = M.compute_metrics(yt, yp, pb, class_names)
            o = res["overall"]
            rows.append({
                "model": run_dir.name,
                "noise": kind,
                "severity": sev,
                "accuracy": o["accuracy"],
                "balanced_accuracy": o["balanced_accuracy"],
                "macro_f1": o["macro_f1"],
                "macro_auc": o["macro_auc"],
                "mcc": o["mcc"],
                "n": o["n_samples"],
            })
            print(f"  {run_dir.name:<18} {kind:<12} sev={sev:<6} "
                  f"acc={o['accuracy']:.4f}  macroF1={o['macro_f1']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Test-time noise robustness sweep")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run folder names under results/, e.g. --runs noiseaug geomonly")
    ap.add_argument("--out", default="noise_robustness")
    args = ap.parse_args()

    D.set_seed(config.SEED)
    device = get_device()
    paths, labels, class_names = D.build_index()

    sweeps = {
        "gaussian": config.TEST_GAUSSIAN_SIGMAS,
        "salt_pepper": config.TEST_SALTPEPPER_DENSITIES,
        "speckle": config.TEST_SPECKLE_VARS,
    }

    out_dir = Path(config.OUT_DIR) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("NOISE ROBUSTNESS SWEEP")
    print("=" * 66)
    print("Corruption is applied to TEST images only, after resizing and before")
    print("normalisation, so severities are directly comparable to the paper.\n")

    rows = []
    for name in args.runs:
        rows += sweep_run(Path(config.OUT_DIR) / name, paths, labels,
                          class_names, device, sweeps)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "noise_sweep.csv", index=False)
    M.plot_noise_sweep(df, out_dir / "noise_sweep.png")

    # ---- summary table: accuracy by model x noise x severity -------------- #
    pivot = df.pivot_table(index=["noise", "severity"], columns="model",
                           values="accuracy")
    lines = ["TEST ACCURACY UNDER NOISE", "=" * 66,
             pivot.to_string(float_format=lambda v: f"{v:.4f}"), ""]

    if len(args.runs) == 2:
        a, b = args.runs
        if a in pivot.columns and b in pivot.columns:
            delta = (pivot[a] - pivot[b])
            lines += ["", f"ACCURACY GAIN: {a} minus {b}", "-" * 66,
                      delta.to_string(float_format=lambda v: f"{v:+.4f}"), "",
                      f"mean gain across all corrupted settings: "
                      f"{delta[delta.index.get_level_values('severity') > 0].mean():+.4f}",
                      f"difference on clean images (severity 0)   : "
                      f"{delta[delta.index.get_level_values('severity') == 0].mean():+.4f}"]

    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text)
    print("\n" + text)
    print(f"\nWritten to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
