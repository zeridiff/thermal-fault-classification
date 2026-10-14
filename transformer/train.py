"""
train.py  --  MAIN SCRIPT.  Right-click in PyCharm -> Run 'train'.
=================================================================

What it does
------------
1. Scans DATA_ROOT and prints the class distribution.
2. Splits the data (stratified; see PROTOCOL in config.py).
3. Trains the multi-scale CNN with noise-injection augmentation.
4. Evaluates on the held-out fold/test set and writes every metric and figure
   into results/.

Outputs (per run, under results/<run_name>/)
--------------------------------------------
    model_fold{k}.pt                 trained weights
    test_report.txt                  human-readable metric dump
    test_overall_metrics.csv/json    accuracy, F1, AUC, kappa, MCC ...
    test_per_class_metrics.csv       TP/FP/FN/TN, precision, sensitivity,
                                     specificity, F1, AUC per class
    test_confusion_matrix.csv/.png   counts + row-normalised
    roc_curves.png, pr_curves.png
    training_curves_fold{k}.png
    cv_fold_metrics.csv              per-fold table (k-fold protocol)
    cv_summary.csv                   mean +/- std across folds
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

import config
import data as D
import metrics as M
from engine import fit, evaluate, get_device
from model import build_model, count_parameters


# --------------------------------------------------------------------------- #
def run_one_split(paths, labels, class_names, train_idx, test_idx, device,
                  out_dir, fold_tag="", noise_aug=None, verbose=True):
    """Train on train_idx (minus an inner validation slice) and score on test_idx."""
    # Inner validation slice, carved out of the training portion only.
    tr_idx, va_idx = D.stratified_split(
        labels, test_size=config.VAL_FRACTION, seed=config.SEED, indices=train_idx
    )

    tr_paths, tr_labels = D.subset(paths, labels, tr_idx)
    va_paths, va_labels = D.subset(paths, labels, va_idx)
    te_paths, te_labels = D.subset(paths, labels, test_idx)

    train_loader = D.make_loader(tr_paths, tr_labels,
                                 D.train_transform(with_noise=noise_aug), shuffle=True)
    val_loader = D.make_loader(va_paths, va_labels, D.eval_transform())
    test_loader = D.make_loader(te_paths, te_labels, D.eval_transform())

    if verbose:
        print(f"  split sizes -> train {len(tr_idx)} | val {len(va_idx)} | test {len(test_idx)}")

    model = build_model(num_classes=len(class_names)).to(device)
    if verbose and fold_tag in ("", "fold1 "):
        total, trainable = count_parameters(model)
        print(f"  model: {config.BACKBONE}, fused feature dim {model.feature_dim}, "
              f"{total:,} params ({trainable:,} trainable)")

    cw = D.class_weights(tr_labels, len(class_names)) if config.USE_CLASS_WEIGHTS else None
    model, history = fit(model, train_loader, val_loader, device,
                         class_weight=cw, verbose=verbose, tag=fold_tag)

    _, _, y_true, y_pred, y_prob = evaluate(model, test_loader, None, device)
    res = M.compute_metrics(y_true, y_pred, y_prob, class_names)
    res["overall"]["train_time_s"] = round(history["train_time_s"], 2)
    res["overall"]["epochs_run"] = len(history["train_loss"])

    return model, history, res, (y_true, y_pred, y_prob)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Train IR transformer fault classifier")
    ap.add_argument("--protocol", choices=["kfold", "single"], default=config.PROTOCOL)
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--backbone", default=config.BACKBONE,
                    choices=["resnet18", "mobilenet_v2"])
    ap.add_argument("--no-noise-aug", action="store_true",
                    help="Disable noise injection. Use this to produce the "
                         "geometry-only baseline that the paper corresponds to.")
    ap.add_argument("--name", default=None, help="Sub-folder name under results/")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # Command-line overrides win over config.py
    config.EPOCHS = args.epochs
    config.BACKBONE = args.backbone
    noise_aug = not args.no_noise_aug
    config.NOISE_AUG_ENABLED = noise_aug

    run_name = args.name or (
        f"{args.backbone}_{'noiseaug' if noise_aug else 'geomonly'}_{args.protocol}"
    )
    out_dir = Path(config.OUT_DIR) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    D.set_seed(config.SEED)
    device = get_device()

    paths, labels, class_names = D.build_index()
    print("\n" + "=" * 66)
    print("DATASET")
    print("=" * 66)
    print(f"root   : {config.DATA_ROOT}")
    print(f"device : {device}")
    print(f"run    : {run_name}   (noise augmentation: {'ON' if noise_aug else 'OFF'})")
    print()
    print(D.describe(labels, class_names))

    verbose = not args.quiet
    fold_overalls, all_true, all_pred, all_prob = [], [], [], []

    # ---------------------------------------------------------------- #
    if args.protocol == "single":
        print("\n" + "=" * 66)
        print(f"TRAINING  --  single stratified split "
              f"({1-config.TEST_FRACTION:.0%} train / {config.TEST_FRACTION:.0%} test)")
        print("=" * 66)
        tr_idx, te_idx = D.stratified_split(labels, test_size=config.TEST_FRACTION)
        model, history, res, (yt, yp, pr) = run_one_split(
            paths, labels, class_names, tr_idx, te_idx, device, out_dir,
            noise_aug=noise_aug, verbose=verbose)

        torch.save({"state_dict": model.state_dict(),
                    "class_names": class_names,
                    "backbone": config.BACKBONE,
                    "test_idx": te_idx.tolist()},
                   out_dir / "model_fold1.pt")
        M.plot_training_curves(history, out_dir / "training_curves_fold1.png")
        fold_overalls.append(res["overall"])
        all_true, all_pred, all_prob = yt, yp, pr

    # ---------------------------------------------------------------- #
    else:
        print("\n" + "=" * 66)
        print(f"TRAINING  --  stratified {args.folds}-fold cross-validation")
        print("Every image is used for testing exactly once. Reported numbers are")
        print("the mean +/- std over folds, which is the defensible way to report")
        print("results on 255 images.")
        print("=" * 66)

        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)
        for k, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), start=1):
            print(f"\n--- Fold {k}/{args.folds} " + "-" * 44)
            model, history, res, (yt, yp, pr) = run_one_split(
                paths, labels, class_names, tr_idx, te_idx, device, out_dir,
                fold_tag=f"fold{k} ", noise_aug=noise_aug, verbose=verbose)

            torch.save({"state_dict": model.state_dict(),
                        "class_names": class_names,
                        "backbone": config.BACKBONE,
                        "test_idx": te_idx.tolist()},
                       out_dir / f"model_fold{k}.pt")
            M.plot_training_curves(history, out_dir / f"training_curves_fold{k}.png")

            o = dict(res["overall"]); o["fold"] = k
            fold_overalls.append(o)
            all_true.append(yt); all_pred.append(yp); all_prob.append(pr)
            print(f"  fold {k}: accuracy {res['overall']['accuracy']:.4f} | "
                  f"macro-F1 {res['overall']['macro_f1']:.4f} | "
                  f"{res['overall']['train_time_s']:.0f}s")

        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        all_prob = np.concatenate(all_prob)

        fold_df, summary = M.aggregate_folds(fold_overalls)
        fold_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
        summary.to_csv(out_dir / "cv_summary.csv")
        print("\n" + "=" * 66)
        print("CROSS-VALIDATION SUMMARY (mean +/- std over folds)")
        print("=" * 66)
        for key in ["accuracy", "balanced_accuracy", "macro_precision",
                    "macro_sensitivity", "macro_specificity", "macro_f1",
                    "macro_auc", "cohen_kappa", "mcc"]:
            if key in summary.index:
                print(f"  {key:<20}: {summary.loc[key,'mean']:.4f} +/- {summary.loc[key,'std']:.4f}")

    # ---------------------------------------------------------------- #
    # Pooled out-of-fold results: every image predicted exactly once.
    pooled = M.compute_metrics(all_true, all_pred, all_prob, class_names)
    print(M.format_report(pooled, title="POOLED OUT-OF-FOLD TEST RESULTS"
                          if args.protocol == "kfold" else "HELD-OUT TEST RESULTS"))

    M.save_results(pooled, out_dir, tag="test")
    M.plot_confusion_matrix(pooled["confusion_matrix"], class_names,
                            out_dir / "confusion_matrix_counts.png", normalize=False)
    M.plot_confusion_matrix(pooled["confusion_matrix"], class_names,
                            out_dir / "confusion_matrix_normalised.png", normalize=True)
    M.plot_roc_curves(all_true, all_prob, class_names, out_dir / "roc_curves.png")
    M.plot_pr_curves(all_true, all_prob, class_names, out_dir / "pr_curves.png")

    np.savez(out_dir / "predictions.npz",
             y_true=all_true, y_pred=all_pred, y_prob=all_prob,
             class_names=np.array(class_names))
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(config).items()
                   if k.isupper()}, f, indent=2)

    print(f"Everything written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
