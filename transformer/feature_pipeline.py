"""
feature_pipeline.py  --  the paper's own recipe, for a fair comparison table.
============================================================================

The reference paper does NOT classify with the CNN. It uses the CNN only as a
feature extractor, then:

    deep features (2 layers) -> concatenate -> chi-square selection (50 features)
    -> LDA / kNN / linear-SVM / quadratic-SVM / cubic-SVM / medium-Gaussian-SVM
    -> 10-fold cross-validation

This script reproduces that, so the report can put "Trans-Light-style pipeline"
and "end-to-end multi-scale CNN" side by side on identical splits.

Two deliberate deviations, both of which make the numbers more trustworthy:

  1. Scaling and feature selection run INSIDE each CV fold (sklearn Pipeline).
     Selecting features on all 255 images first, then cross-validating, lets
     information from the test fold influence which features exist. That is a
     subtle but real leak and it inflates accuracy.

  2. Feature extraction can optionally be done under noise (--noise), so the
     robustness question can be asked of this pipeline too.

Runs on CPU in well under a minute - useful when you want a result fast.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import config
import data as D
import metrics as M
from engine import get_device
from model import build_model


# --------------------------------------------------------------------------- #
def build_classifiers():
    """The same six models the paper compares, named as in MATLAB's toolbox."""
    return {
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "kNN (k=1)": KNeighborsClassifier(n_neighbors=1, metric="euclidean"),
        "Linear SVM": SVC(kernel="linear", C=1.0, probability=True, random_state=config.SEED),
        "Quadratic SVM": SVC(kernel="poly", degree=2, C=1.0, probability=True, random_state=config.SEED),
        "Cubic SVM": SVC(kernel="poly", degree=3, C=1.0, probability=True, random_state=config.SEED),
        "Medium Gaussian SVM": SVC(kernel="rbf", gamma="scale", C=1.0, probability=True,
                                   random_state=config.SEED),
    }


@torch.no_grad()
def extract_all_features(paths, labels, device, checkpoint=None,
                         noise_kind=None, severity=0.0):
    """
    Run every image through the multi-scale network and keep the fused vector.

    checkpoint=None -> plain ImageNet weights (pure transfer learning, no fine-tune)
    checkpoint=path -> weights from a train.py run (fine-tuned features)
    """
    if checkpoint:
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        model = build_model(num_classes=len(ck["class_names"]),
                            backbone=ck.get("backbone", config.BACKBONE),
                            pretrained=False)
        model.load_state_dict(ck["state_dict"])
    else:
        model = build_model(num_classes=len(np.unique(labels)),
                            pretrained=config.PRETRAINED)

    model.to(device).eval()
    loader = D.make_loader(paths, labels,
                           D.eval_transform(noise_kind=noise_kind, severity=severity),
                           batch_size=32)

    feats, ys = [], []
    for x, y in loader:
        feats.append(model.extract_features(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(feats), np.concatenate(ys)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Deep features + classical classifiers")
    ap.add_argument("--k-features", type=int, nargs="+", default=[50, 100, 150, 0],
                    help="Feature counts for chi-square selection; 0 means use all.")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--checkpoint", default=None,
                    help="Optional path to a model_fold*.pt from train.py.")
    ap.add_argument("--noise", default=None,
                    choices=["gaussian", "salt_pepper", "speckle"])
    ap.add_argument("--severity", type=float, default=0.0)
    ap.add_argument("--name", default="feature_pipeline")
    args = ap.parse_args()

    D.set_seed(config.SEED)
    device = get_device()
    out_dir = Path(config.OUT_DIR) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, class_names = D.build_index()
    print("=" * 66)
    print("DEEP FEATURES + CLASSICAL CLASSIFIERS  (paper-style pipeline)")
    print("=" * 66)
    print(f"images : {len(paths)}   classes: {len(class_names)}   device: {device}")
    print(f"weights: {'fine-tuned ' + args.checkpoint if args.checkpoint else 'ImageNet (not fine-tuned)'}")
    if args.checkpoint:
        print("\n  !! LEAKAGE WARNING -------------------------------------------")
        print("  These features come from a network that was fine-tuned on this")
        print("  same dataset, so images in the CV test folds helped shape the")
        print("  feature extractor. Accuracy will be optimistic, often 100%.")
        print("  The reference paper does exactly this, which is one reason its")
        print("  headline number should be read with caution. Report the run")
        print("  WITHOUT --checkpoint as your clean result, and this one only as")
        print("  a like-for-like reproduction of the paper's protocol.")
        print("  --------------------------------------------------------------\n")
    if args.noise:
        print(f"noise  : {args.noise} at severity {args.severity}")

    X, y = extract_all_features(paths, labels, device, args.checkpoint,
                                args.noise, args.severity)
    print(f"feature matrix: {X.shape}\n")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)
    classifiers = build_classifiers()

    rows, best = [], None
    for k in args.k_features:
        k_eff = "all" if k in (0, None) else min(k, X.shape[1])
        for clf_name, clf in classifiers.items():
            # MinMax first: chi2 requires non-negative inputs, and the fused
            # features contain negatives because of the BatchNorm branches.
            pipe = Pipeline([
                ("scale", MinMaxScaler()),
                ("select", SelectKBest(chi2, k=k_eff)),
                ("clf", clf),
            ])
            # One CV pass only: predictions are the argmax of the probabilities,
            # which keeps the two perfectly consistent and halves the runtime.
            y_prob = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba", n_jobs=-1)
            y_pred = y_prob.argmax(axis=1)
            res = M.compute_metrics(y, y_pred, y_prob, class_names)
            o = res["overall"]
            rows.append({
                "n_features": k_eff, "classifier": clf_name,
                "accuracy": o["accuracy"], "balanced_accuracy": o["balanced_accuracy"],
                "macro_precision": o["macro_precision"],
                "macro_sensitivity": o["macro_sensitivity"],
                "macro_specificity": o["macro_specificity"],
                "macro_f1": o["macro_f1"], "macro_auc": o["macro_auc"],
                "mcc": o["mcc"],
            })
            print(f"  {str(k_eff):>4} features | {clf_name:<22} "
                  f"acc={o['accuracy']:.4f}  macroF1={o['macro_f1']:.4f}")

            if best is None or o["accuracy"] > best[0]:
                best = (o["accuracy"], k_eff, clf_name, res, y_pred, y_prob)

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df.to_csv(out_dir / "classifier_comparison.csv", index=False)

    acc, k_eff, clf_name, res, y_pred, y_prob = best
    print("\n" + "=" * 66)
    print(f"BEST: {clf_name} with {k_eff} features -> accuracy {acc:.4f}")
    print("=" * 66)
    print(M.format_report(res, title=f"BEST CONFIGURATION ({clf_name}, {k_eff} features)"))

    M.save_results(res, out_dir, tag="best")
    M.plot_confusion_matrix(res["confusion_matrix"], class_names,
                            out_dir / "best_confusion_matrix.png")
    M.plot_confusion_matrix(res["confusion_matrix"], class_names,
                            out_dir / "best_confusion_matrix_normalised.png", normalize=True)
    M.plot_roc_curves(y, y_prob, class_names, out_dir / "best_roc_curves.png")

    # accuracy vs feature-count chart, the paper's Tables 7 and 8
    fig_df = df.pivot_table(index="n_features", columns="classifier", values="accuracy")
    fig_df.to_csv(out_dir / "accuracy_vs_feature_count.csv")

    print(f"\nWritten to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
