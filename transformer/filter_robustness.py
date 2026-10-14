"""
filter_robustness.py
====================
Benchmark image filters AFTER synthetic noise and BEFORE the CNN.

This script is designed to sit next to your existing config.py, data.py,
metrics.py, model.py, engine.py and trained results/ folders.

Why this version is important
-----------------------------
Your existing noise_robustness.py is useful, but the random corruption is drawn
inside the transform. Therefore two different model runs can receive different
random realisations of "the same" Gaussian/speckle/salt-and-pepper condition.
This script fixes that by making noise deterministic per original image,
noise type, severity and repeat. Every model and every filter therefore sees
exactly the same corrupted image for a fair paired comparison.

Pipeline
--------
clean held-out image
    -> resize to 224x224
    -> deterministic synthetic noise
    -> optional denoising filter
    -> ImageNet normalisation
    -> already-trained CNN

Outputs
-------
results/filter_robustness/
    filter_sweep_raw.csv
    filter_sweep_summary.csv
    filter_improvement.csv
    filter_ranking.csv
    summary.txt
    accuracy_<model>_<noise>.png

The CSV records both diagnosis metrics (accuracy, balanced accuracy, macro-F1,
AUC, MCC) and restoration metrics (PSNR, SSIM) plus filter latency.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import config
import data as D
import metrics as M
from engine import get_device
from filters import FILTER_NAMES, apply_filter
from model import build_model


# ---------------------------------------------------------------------------
# Deterministic corruption: same image + condition => same noise realisation.
# ---------------------------------------------------------------------------
def _stable_seed(image_index: int, noise_kind: str, severity: float, repeat: int) -> int:
    token = f"{config.SEED}|{image_index}|{noise_kind}|{severity:.12g}|{repeat}".encode("utf-8")
    digest = hashlib.blake2b(token, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def add_noise_np(img: np.ndarray, kind: str, severity: float, seed: int) -> np.ndarray:
    """Match the semantics of noise.py, but with a local deterministic RNG."""
    x = np.asarray(img, dtype=np.float32)
    if severity <= 0:
        return x.copy()

    rng = np.random.default_rng(seed)

    if kind == "gaussian":
        n = rng.normal(0.0, severity, size=x.shape).astype(np.float32)
        return np.clip(x + n, 0.0, 1.0)

    if kind == "salt_pepper":
        out = x.copy()
        h, w, _ = out.shape
        mask = rng.random((h, w), dtype=np.float32)
        pepper = mask < (severity / 2.0)
        salt = mask > (1.0 - severity / 2.0)
        out[pepper, :] = 0.0
        out[salt, :] = 1.0
        return out

    if kind == "speckle":
        # Existing noise.py interprets severity as variance.
        std = math.sqrt(severity)
        n = rng.normal(0.0, std, size=x.shape).astype(np.float32)
        return np.clip(x + x * n, 0.0, 1.0)

    raise ValueError(f"Unknown noise kind: {kind}")


# ---------------------------------------------------------------------------
def load_fold_model(ckpt_path: Path, device):
    meta = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = build_model(
        num_classes=len(meta["class_names"]),
        backbone=meta.get("backbone", config.BACKBONE),
        pretrained=False,
    )
    net.load_state_dict(meta["state_dict"])
    net.to(device).eval()
    return net, meta


@torch.no_grad()
def predict_tensor_batch(model, x: torch.Tensor, device, batch_size: int):
    probs, preds = [], []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        logits = model(xb)
        pb = torch.softmax(logits, dim=1)
        probs.append(pb.cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def prepare_fold_batch(paths, labels, indices, noise_kind, severity, filter_name, repeat):
    """Return normalized tensor batch + labels + image-quality measurements."""
    mean = np.asarray(config.MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config.STD, dtype=np.float32).reshape(1, 1, 3)

    xs, ys = [], []
    psnr_noisy, psnr_filt = [], []
    ssim_noisy, ssim_filt = [], []
    filter_times_ms = []

    # Pillow's resize is deterministic. BILINEAR is close to torchvision Resize
    # for ordinary RGB images and keeps this script independent of transform RNG.
    size_wh = (config.IMG_SIZE, config.IMG_SIZE)

    for global_idx in indices:
        with Image.open(paths[int(global_idx)]) as im:
            clean = np.asarray(
                im.convert("RGB").resize(size_wh, resample=Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0

        seed = _stable_seed(int(global_idx), noise_kind, float(severity), int(repeat))
        noisy = add_noise_np(clean, noise_kind, float(severity), seed)

        t0 = time.perf_counter()
        filtered = apply_filter(noisy, filter_name)
        filter_times_ms.append((time.perf_counter() - t0) * 1000.0)

        # Synthetic-noise experiment gives us a known clean reference, so image
        # restoration quality can be measured as well as CNN accuracy.
        psnr_noisy.append(peak_signal_noise_ratio(clean, noisy, data_range=1.0))
        psnr_filt.append(peak_signal_noise_ratio(clean, filtered, data_range=1.0))
        ssim_noisy.append(structural_similarity(clean, noisy, data_range=1.0, channel_axis=2))
        ssim_filt.append(structural_similarity(clean, filtered, data_range=1.0, channel_axis=2))

        norm = (filtered - mean) / std
        xs.append(torch.from_numpy(norm.transpose(2, 0, 1).copy()).float())
        ys.append(int(labels[int(global_idx)]))

    return (
        torch.stack(xs, dim=0),
        np.asarray(ys, dtype=np.int64),
        {
            "psnr_noisy": float(np.mean(psnr_noisy)),
            "psnr_filtered": float(np.mean(psnr_filt)),
            "ssim_noisy": float(np.mean(ssim_noisy)),
            "ssim_filtered": float(np.mean(ssim_filt)),
            "filter_ms_per_image": float(np.mean(filter_times_ms)),
        },
    )


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)


def plot_accuracy(summary: pd.DataFrame, out_dir: Path):
    for (model_name, noise), sub in summary.groupby(["model", "noise"]):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for filt, grp in sub.groupby("filter"):
            grp = grp.sort_values("severity")
            ax.plot(grp["severity"], grp["accuracy"], marker="o", label=filt)
        ax.set_title(f"{model_name}: {noise} noise")
        ax.set_xlabel("noise severity")
        ax.set_ylabel("pooled held-out accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"accuracy_{_safe_name(model_name)}_{_safe_name(noise)}.png", dpi=190)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Noise + filtering robustness benchmark")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="trained run folders under results/, e.g. noiseaug geomonly")
    ap.add_argument("--filters", nargs="+",
                    default=["none", "median3", "gaussian3", "bilateral", "nlm"],
                    choices=FILTER_NAMES)
    ap.add_argument("--repeats", type=int, default=1,
                    help="independent deterministic noise realisations per condition")
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--out", default="filter_robustness")
    args = ap.parse_args()

    if args.repeats < 1:
        ap.error("--repeats must be >= 1")

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

    print("=" * 78)
    print("FILTER ROBUSTNESS BENCHMARK")
    print("=" * 78)
    print(f"device  : {device}")
    print(f"runs    : {args.runs}")
    print(f"filters : {args.filters}")
    print(f"repeats : {args.repeats}")
    print("Every run/filter receives the SAME deterministic corrupted image.")
    print()

    raw_rows = []

    for run_name in args.runs:
        run_dir = Path(config.OUT_DIR) / run_name
        ckpts = sorted(run_dir.glob("model_fold*.pt"))
        if not ckpts:
            raise FileNotFoundError(
                f"No model_fold*.pt files in {run_dir}. Check the exact folder name under results/."
            )

        print("\n" + "#" * 78)
        print(f"RUN: {run_name}  ({len(ckpts)} fold checkpoint(s))")
        print("#" * 78)

        # Accumulate out-of-fold predictions for each experimental condition.
        acc = defaultdict(lambda: {
            "yt": [], "yp": [], "pb": [],
            "psnr_noisy": [], "psnr_filtered": [],
            "ssim_noisy": [], "ssim_filtered": [],
            "filter_ms_per_image": [], "n": [],
        })

        for fold_no, ckpt in enumerate(ckpts, start=1):
            model, meta = load_fold_model(ckpt, device)
            te_idx = np.asarray(meta.get("test_idx", []), dtype=np.int64)
            if len(te_idx) == 0:
                raise RuntimeError(f"{ckpt} has no test_idx metadata")

            for repeat in range(args.repeats):
                for noise_kind, severities in sweeps.items():
                    for severity in severities:
                        for filter_name in args.filters:
                            x, y, iq = prepare_fold_batch(
                                paths, labels, te_idx, noise_kind, severity,
                                filter_name, repeat,
                            )
                            yp, pb = predict_tensor_batch(model, x, device, args.batch_size)
                            key = (noise_kind, float(severity), filter_name, int(repeat))
                            slot = acc[key]
                            slot["yt"].append(y)
                            slot["yp"].append(yp)
                            slot["pb"].append(pb)
                            for k in ("psnr_noisy", "psnr_filtered", "ssim_noisy",
                                      "ssim_filtered", "filter_ms_per_image"):
                                slot[k].append(iq[k] * len(y))
                            slot["n"].append(len(y))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  finished fold {fold_no}/{len(ckpts)}")

        # Convert pooled OOF predictions into metrics.
        for (noise_kind, severity, filter_name, repeat), slot in acc.items():
            yt = np.concatenate(slot["yt"])
            yp = np.concatenate(slot["yp"])
            pb = np.concatenate(slot["pb"])
            res = M.compute_metrics(yt, yp, pb, class_names)
            o = res["overall"]
            n_total = int(sum(slot["n"]))

            def wmean(name):
                return float(sum(slot[name]) / max(n_total, 1))

            row = {
                "model": run_name,
                "noise": noise_kind,
                "severity": severity,
                "filter": filter_name,
                "repeat": repeat,
                "accuracy": o["accuracy"],
                "balanced_accuracy": o["balanced_accuracy"],
                "macro_f1": o["macro_f1"],
                "macro_auc": o["macro_auc"],
                "mcc": o["mcc"],
                "psnr_noisy": wmean("psnr_noisy"),
                "psnr_filtered": wmean("psnr_filtered"),
                "ssim_noisy": wmean("ssim_noisy"),
                "ssim_filtered": wmean("ssim_filtered"),
                "filter_ms_per_image": wmean("filter_ms_per_image"),
                "n": n_total,
            }
            row["delta_psnr_db"] = row["psnr_filtered"] - row["psnr_noisy"]
            row["delta_ssim"] = row["ssim_filtered"] - row["ssim_noisy"]
            raw_rows.append(row)
            print(
                f"  {noise_kind:<12} sev={severity:<6g} {filter_name:<10} "
                f"acc={row['accuracy']:.4f} F1={row['macro_f1']:.4f} "
                f"dPSNR={row['delta_psnr_db']:+.2f}dB"
            )

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out_dir / "filter_sweep_raw.csv", index=False)

    # Average across independent noise realisations.
    group_cols = ["model", "noise", "severity", "filter"]
    metric_cols = [
        "accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "mcc",
        "psnr_noisy", "psnr_filtered", "ssim_noisy", "ssim_filtered",
        "delta_psnr_db", "delta_ssim", "filter_ms_per_image", "n",
    ]
    summary = raw.groupby(group_cols, as_index=False)[metric_cols].mean()
    summary.to_csv(out_dir / "filter_sweep_summary.csv", index=False)

    # Classification gain relative to the unfiltered image under identical noise.
    base = summary[summary["filter"] == "none"][
        ["model", "noise", "severity", "accuracy", "macro_f1", "mcc"]
    ].rename(columns={
        "accuracy": "accuracy_unfiltered",
        "macro_f1": "macro_f1_unfiltered",
        "mcc": "mcc_unfiltered",
    })
    imp = summary.merge(base, on=["model", "noise", "severity"], how="left")
    imp["delta_accuracy"] = imp["accuracy"] - imp["accuracy_unfiltered"]
    imp["delta_macro_f1"] = imp["macro_f1"] - imp["macro_f1_unfiltered"]
    imp["delta_mcc"] = imp["mcc"] - imp["mcc_unfiltered"]
    imp.to_csv(out_dir / "filter_improvement.csv", index=False)

    # Exploratory universal-filter ranking over corrupted conditions only.
    corrupted = imp[imp["severity"] > 0].copy()
    ranking = corrupted.groupby(["model", "filter"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        mean_macro_f1=("macro_f1", "mean"),
        mean_mcc=("mcc", "mean"),
        mean_delta_accuracy=("delta_accuracy", "mean"),
        mean_delta_macro_f1=("delta_macro_f1", "mean"),
        mean_delta_psnr_db=("delta_psnr_db", "mean"),
        mean_delta_ssim=("delta_ssim", "mean"),
        mean_filter_ms=("filter_ms_per_image", "mean"),
    ).sort_values(["model", "mean_macro_f1"], ascending=[True, False])
    ranking.to_csv(out_dir / "filter_ranking.csv", index=False)

    plot_accuracy(summary, out_dir)

    lines = [
        "FILTER ROBUSTNESS SUMMARY",
        "=" * 78,
        "Ranking is exploratory: report all filters, or select parameters on validation data",
        "before making a final 'best filter' claim on the held-out test folds.",
        "",
        ranking.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(ranking.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nEverything written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
