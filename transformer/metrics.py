"""
metrics.py
==========
Every number and figure the report needs.

Metric definitions match Eqs. (1)-(5) of the reference paper, computed per class
in a one-vs-rest manner from the 9x9 confusion matrix:

    Accuracy    = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (TP + FN)          (= recall)
    Specificity = TN / (TN + FP)
    Precision   = TP / (TP + FP)
    F1          = 2TP / (2TP + FP + FN)

Beyond those, three extra numbers are reported because they are much harder to
fool on a small imbalanced dataset:

    Balanced accuracy - mean of per-class recall; immune to class-size effects
    Cohen's kappa     - agreement corrected for chance
    MCC               - correlation between prediction and truth, robust to imbalance

Why that matters here: the largest class (600 SC, 40 images) is 16% of the data,
so a model that only ever learned that class would already score 16% "accuracy".
Balanced accuracy and MCC do not reward that.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless backend; PyCharm saves files instead of popping windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred, y_prob, class_names):
    """
    Returns a dict with `overall` (scalars) and `per_class` (DataFrame).

    y_prob : (N, C) softmax probabilities. Used for AUC and top-k only.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n_classes = len(class_names)
    labels = np.arange(n_classes)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # one-vs-rest TP / FP / FN / TN from the confusion matrix
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = np.where((tn + fp) > 0, tn / (tn + fp), np.nan)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    # per-class AUC, guarding against classes absent from this split
    aucs = []
    y_bin = label_binarize(y_true, classes=labels)
    if n_classes == 2:                       # label_binarize collapses binary
        y_bin = np.hstack([1 - y_bin, y_bin])
    for c in labels:
        if y_bin[:, c].sum() in (0, len(y_true)):
            aucs.append(np.nan)
        else:
            aucs.append(roc_auc_score(y_bin[:, c], y_prob[:, c]))
    aucs = np.asarray(aucs, dtype=float)

    per_class = pd.DataFrame({
        "class": class_names,
        "support": support,
        "TP": tp.astype(int), "FP": fp.astype(int),
        "FN": fn.astype(int), "TN": tn.astype(int),
        "precision": precision,
        "sensitivity(recall)": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": aucs,
    })

    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        weighted_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except ValueError:
        macro_auc = weighted_auc = float("nan")

    try:
        top3 = top_k_accuracy_score(y_true, y_prob, k=3, labels=labels)
    except ValueError:
        top3 = float("nan")

    overall = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "top3_accuracy": top3,
        "macro_precision": float(np.mean(precision)),
        "macro_sensitivity": float(np.mean(recall)),
        "macro_specificity": float(np.nanmean(specificity)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0)[2]),
        "macro_auc": macro_auc,
        "weighted_auc": weighted_auc,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred, labels=labels),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "n_samples": int(len(y_true)),
    }

    return {"overall": overall, "per_class": per_class, "confusion_matrix": cm}


# --------------------------------------------------------------------------- #
# Pretty printing
# --------------------------------------------------------------------------- #
def format_report(res, title="RESULTS"):
    o = res["overall"]
    lines = [
        "", "=" * 66, title, "=" * 66,
        f"  Accuracy            : {o['accuracy']:.4f}  ({o['accuracy']*100:.2f} %)",
        f"  Balanced accuracy   : {o['balanced_accuracy']:.4f}",
        f"  Top-3 accuracy      : {o['top3_accuracy']:.4f}",
        f"  Macro precision     : {o['macro_precision']:.4f}",
        f"  Macro sensitivity   : {o['macro_sensitivity']:.4f}",
        f"  Macro specificity   : {o['macro_specificity']:.4f}",
        f"  Macro F1            : {o['macro_f1']:.4f}",
        f"  Weighted F1         : {o['weighted_f1']:.4f}",
        f"  Macro AUC (OvR)     : {o['macro_auc']:.4f}",
        f"  Cohen's kappa       : {o['cohen_kappa']:.4f}",
        f"  MCC                 : {o['mcc']:.4f}",
        f"  Test samples        : {o['n_samples']}",
        "-" * 66, "Per-class:", "-" * 66,
        res["per_class"].to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "=" * 66, "",
    ]
    return "\n".join(lines)


def save_results(res, out_dir: Path, tag: str = "test"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["per_class"].to_csv(out_dir / f"{tag}_per_class_metrics.csv", index=False)
    pd.DataFrame([res["overall"]]).to_csv(out_dir / f"{tag}_overall_metrics.csv", index=False)
    with open(out_dir / f"{tag}_overall_metrics.json", "w") as f:
        json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v)
                   for k, v in res["overall"].items()}, f, indent=2)
    np.savetxt(out_dir / f"{tag}_confusion_matrix.csv",
               res["confusion_matrix"], fmt="%d", delimiter=",")
    with open(out_dir / f"{tag}_report.txt", "w") as f:
        f.write(format_report(res, title=f"{tag.upper()} RESULTS"))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm, class_names, out_path, normalize=False, title=None):
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        row[row == 0] = 1
        cm = cm / row

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title or ("Confusion matrix (row-normalised)" if normalize
                           else "Confusion matrix (counts)"))

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    fmt = "{:.2f}" if normalize else "{:.0f}"
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt.format(cm[i, j]), ha="center", va="center",
                    fontsize=8, color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_roc_curves(y_true, y_prob, class_names, out_path):
    """One-vs-rest ROC per class + micro average."""
    labels = np.arange(len(class_names))
    y_bin = label_binarize(y_true, classes=labels)
    if len(class_names) == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for c, name in enumerate(class_names):
        if y_bin[:, c].sum() in (0, len(y_true)):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_prob[:, c])
        ax.plot(fpr, tpr, lw=1.4,
                label=f"{name} (AUC={roc_auc_score(y_bin[:, c], y_prob[:, c]):.3f})")

    fpr_m, tpr_m, _ = roc_curve(y_bin.ravel(), np.asarray(y_prob).ravel())
    ax.plot(fpr_m, tpr_m, "k--", lw=2,
            label=f"micro-average (AUC={roc_auc_score(y_bin.ravel(), np.asarray(y_prob).ravel()):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")

    ax.set_xlabel("False positive rate  (1 - specificity)")
    ax.set_ylabel("True positive rate  (sensitivity)")
    ax.set_title("ROC curves, one-vs-rest")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pr_curves(y_true, y_prob, class_names, out_path):
    """Precision-recall curves; more informative than ROC under class imbalance."""
    labels = np.arange(len(class_names))
    y_bin = label_binarize(y_true, classes=labels)
    if len(class_names) == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for c, name in enumerate(class_names):
        if y_bin[:, c].sum() in (0, len(y_true)):
            continue
        pr, rc, _ = precision_recall_curve(y_bin[:, c], y_prob[:, c])
        ap = average_precision_score(y_bin[:, c], y_prob[:, c])
        ax.plot(rc, pr, lw=1.4, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves, one-vs-rest")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_curves(history, out_path):
    """Loss and accuracy against epoch, train vs validation. Read this for overfitting."""
    ep = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].plot(ep, history["train_loss"], label="train")
    axes[0].plot(ep, history["val_loss"], label="validation")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()

    axes[1].plot(ep, history["train_acc"], label="train")
    axes[1].plot(ep, history["val_acc"], label="validation")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()

    axes[2].plot(ep, history["val_f1"], color="tab:green")
    axes[2].set_title("Validation macro-F1"); axes[2].set_xlabel("epoch")

    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_noise_sweep(df, out_path):
    """Accuracy vs noise severity, one panel per noise type."""
    kinds = df["noise"].unique()
    fig, axes = plt.subplots(1, len(kinds), figsize=(5.2 * len(kinds), 4.4), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        sub = df[df["noise"] == kind]
        for model_tag, grp in sub.groupby("model"):
            grp = grp.sort_values("severity")
            ax.plot(grp["severity"], grp["accuracy"], marker="o", label=model_tag)
        ax.set_title(f"{kind} noise")
        ax.set_xlabel("severity")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def aggregate_folds(fold_overalls):
    """mean +/- std across cross-validation folds."""
    df = pd.DataFrame(fold_overalls)
    summary = pd.DataFrame({"mean": df.mean(numeric_only=True),
                            "std": df.std(numeric_only=True, ddof=1)})
    return df, summary
