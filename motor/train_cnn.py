"""
train_cnn.py
------------
Fault classification of induction-motor thermal (IRT) images using a
transfer-learned CNN, following the general approach used in the IRT +
deep-learning condition-monitoring literature (e.g. Calderon-Uribe et al.
2024; Piechocki et al. 2023; Choudhary et al. 2021): a pretrained CNN
backbone fine-tuned on thermal images, evaluated with stratified k-fold
cross-validation because the dataset is small (a few hundred images spread
across many classes).

WHAT THIS SCRIPT ASSUMES ABOUT YOUR DATA
-----------------------------------------
DATA_DIR should point at a folder that contains one sub-folder per class,
e.g. (matches the "thermal-images-equip" motor set):

    Motor motor rrrrrrr/
        Noload/
        Rotor-0/
        Fan/
        A50/
        A30/
        A10/
        A&C30/
        A&C10/
        A&C&B30/
        A&C&B10/
        A&B50/
        ... (any other class folders)

Each class folder holds the thermal images for that condition (any nesting
inside a class folder is fine -- images are found recursively). The folder
NAME becomes the class label, so name them however you like.

OUTPUTS (written to OUTPUT_DIR)
--------------------------------
- cv_classification_report.txt / .csv   -> per-class precision/recall/F1
- cv_summary.json                       -> accuracy, macro/weighted F1, kappa, ROC-AUC (mean +/- std across folds)
- confusion_matrix.png                  -> aggregated out-of-fold confusion matrix
- training_curves.png                   -> loss/accuracy curves for the final deployment model
- motor_fault_model.pth                 -> final model weights (trained on ALL data) for predict.py / camera_inference.py
- class_names.json                      -> index -> class name mapping + run config, needed by predict.py

HOW TO RUN
----------
1. pip install -r requirements.txt
2. Edit the CONFIG block below (DATA_DIR especially).
3. python train_cnn.py
"""

import os
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # so this also works with no display, e.g. over SSH
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
)

try:
    import cv2  # only needed if ENABLE_CLAHE = True
except ImportError:
    cv2 = None

from PIL import Image


# =====================================================================
# CONFIG -- edit this block for your setup
# =====================================================================
CONFIG = {
    # Folder that contains one sub-folder per fault class.
    # This is the path shown in your screenshot -- change it if you moved
    # or renamed the folder.
    "DATA_DIR": r"C:\Users\dukeb\Downloads\Motor motor rrrrrrrr",

    "OUTPUT_DIR": "outputs",

    # mobilenet_v3_small (fast, light, good default for a small dataset on CPU)
    # resnet18        (classic baseline used in a lot of the IRT papers)
    # squeezenet1_1    (very lightweight, mirrors Siraj et al. 2024's approach)
    "MODEL_NAME": "mobilenet_v3_small",
    "PRETRAINED": True,          # use ImageNet weights (needs internet the first time)
    "IMG_SIZE": 224,

    "N_FOLDS": 5,                 # stratified k-fold cross-validation
    "BATCH_SIZE": 16,
    "EPOCHS": 40,                 # per fold; early stopping usually cuts this short
    "FINAL_EPOCHS": 40,           # for the deployment model trained on 100% of the data
    "LEARNING_RATE": 1e-4,
    "WEIGHT_DECAY": 1e-4,
    "EARLY_STOP_PATIENCE": 10,    # epochs with no val macro-F1 improvement

    # CLAHE (contrast-limited adaptive histogram equalization) is a
    # preprocessing step used in some of the IRT papers (e.g. Xu et al.
    # 2024) to boost contrast in the temperature gradient before the CNN
    # sees it. Off by default -- try turning it on and compare results.
    "ENABLE_CLAHE": False,

    "SEED": 42,
    "NUM_WORKERS": 2,             # set to 0 on Windows if you hit multiprocessing errors
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =====================================================================
# Reproducibility
# =====================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =====================================================================
# Dataset
# =====================================================================
class CLAHETransform:
    """Applies CLAHE to the L channel of the image (LAB colour space).
    Kept separate from hue/saturation jitter on purpose: thermal images
    encode temperature via colour, so we never touch hue/saturation --
    only local contrast."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        if cv2 is None:
            raise ImportError("opencv-python is required for ENABLE_CLAHE=True. "
                               "pip install opencv-python")
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img.convert("RGB"))
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge((l, a, b))
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class ThermalImageDataset(Dataset):
    """Recursively collects images from `root/<class_name>/**` so it
    doesn't matter whether images sit directly inside the class folder or
    a level deeper."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"DATA_DIR does not exist: {self.root}\n"
                f"Update CONFIG['DATA_DIR'] in train_cnn.py to the correct path."
            )
        self.transform = transform

        # Skip folders that aren't fault classes -- important if your scripts
        # and outputs/ live in the same directory as your class folders.
        excluded = {"venv", "env", "__pycache__", "outputs", "output",
                    "node_modules", ".ipynb_checkpoints"}
        class_dirs = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and not d.name.startswith(".") and d.name.lower() not in excluded
        ])
        if not class_dirs:
            raise RuntimeError(f"No class sub-folders found inside {self.root}")

        self.classes_found = [d.name for d in class_dirs]  # every folder seen, before filtering
        class_to_idx_all = {c: i for i, c in enumerate(self.classes_found)}

        raw_samples = []  # (path, label_idx_into_classes_found)
        for class_dir in class_dirs:
            label = class_to_idx_all[class_dir.name]
            for path in class_dir.rglob("*"):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    raw_samples.append((str(path), label))

        if not raw_samples:
            raise RuntimeError(f"No images found under {self.root}")

        # Some folders may contain zero readable images (empty, or only
        # non-image files). Drop those as classes now -- with a clear
        # warning -- instead of letting sklearn crash on a labels/
        # target_names mismatch after a multi-hour training run.
        present_ids = sorted(set(label for _, label in raw_samples))
        empty_folders = [self.classes_found[i] for i in range(len(self.classes_found))
                          if i not in present_ids]
        if empty_folders:
            print(f"WARNING: {len(empty_folders)} folder(s) contain no readable image "
                  f"files and will be SKIPPED as classes: {empty_folders}")
            print("         (check these folders -- they may be empty, or the images "
                  "inside have an unsupported extension)")

        remap = {old: new for new, old in enumerate(present_ids)}
        self.classes = [self.classes_found[i] for i in present_ids]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(path, remap[old_label]) for path, old_label in raw_samples]

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(cfg, train: bool):
    ops = []
    if cfg["ENABLE_CLAHE"]:
        ops.append(CLAHETransform())

    if train:
        ops += [
            transforms.RandomResizedCrop(cfg["IMG_SIZE"], scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            # Deliberately NOT using hue/saturation jitter: thermal images
            # encode temperature as colour, so shifting hue would corrupt
            # the physical meaning of the image. Mild brightness/contrast
            # jitter only, to add some robustness to camera calibration.
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ]
    else:
        ops += [
            transforms.Resize((cfg["IMG_SIZE"], cfg["IMG_SIZE"])),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats -- required
                              std=[0.229, 0.224, 0.225]),  # to match pretrained weights
    ]
    return transforms.Compose(ops)


# =====================================================================
# Model
# =====================================================================
def build_model(model_name: str, num_classes: int, pretrained: bool = True):
    if model_name == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = torchvision.models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "squeezenet1_1":
        weights = torchvision.models.SqueezeNet1_1_Weights.DEFAULT if pretrained else None
        model = torchvision.models.squeezenet1_1(weights=weights)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        model.num_classes = num_classes

    else:
        raise ValueError(f"Unknown MODEL_NAME: {model_name}")

    return model


# =====================================================================
# Train / eval loops
# =====================================================================
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


def train_model(model, train_loader, val_loader, class_weights, device, cfg, epochs, verbose_prefix=""):
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["LEARNING_RATE"],
                                   weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                             factor=0.5, patience=4)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_macro_f1 = -1.0
    best_state = None
    best_val_labels, best_val_preds, best_val_probs = None, None, None
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)

        val_loss, val_acc, val_labels, val_preds = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        _, _, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average="macro", zero_division=0)

        scheduler.step(val_f1)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(f"{verbose_prefix}Epoch {epoch+1:3d}/{epochs} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.3f} macro_f1 {val_f1:.3f}")

        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_labels, best_val_preds = val_labels, val_preds
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["EARLY_STOP_PATIENCE"]:
                print(f"{verbose_prefix}Early stopping at epoch {epoch+1} "
                      f"(no val macro-F1 improvement for {cfg['EARLY_STOP_PATIENCE']} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_val_labels, best_val_preds, best_macro_f1


def get_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


# =====================================================================
# Main
# =====================================================================
def main(cfg=None):
    cfg = cfg or CONFIG
    set_seed(cfg["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(cfg["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = build_transforms(cfg, train=True)
    eval_tf = build_transforms(cfg, train=False)

    # Two dataset instances sharing the same file list / labels, but with
    # different transforms, so a fold's train indices can pull augmented
    # images while its val indices pull clean ones.
    base_ds = ThermalImageDataset(cfg["DATA_DIR"], transform=None)
    train_ds = ThermalImageDataset(cfg["DATA_DIR"], transform=train_tf)
    eval_ds = ThermalImageDataset(cfg["DATA_DIR"], transform=eval_tf)

    classes = base_ds.classes
    num_classes = len(classes)
    targets = np.array(base_ds.targets)

    print(f"Found {len(base_ds)} images across {num_classes} classes:")
    counts = Counter(targets.tolist())
    dist_lines = [f"Found {len(base_ds)} images across {num_classes} classes:"]
    for i, c in enumerate(classes):
        line = f"  [{i:2d}] {c:15s} {counts.get(i, 0)} images"
        print(line)
        dist_lines.append(line)
    with open(out_dir / "class_distribution.txt", "w") as f:
        f.write("\n".join(dist_lines))

    min_class_count = min(counts.values())
    n_folds = min(cfg["N_FOLDS"], min_class_count)
    if n_folds < cfg["N_FOLDS"]:
        print(f"WARNING: smallest class has only {min_class_count} images -- "
              f"reducing N_FOLDS from {cfg['N_FOLDS']} to {n_folds}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg["SEED"])

    oof_labels, oof_preds = [], []
    oof_probs = np.zeros((len(base_ds), num_classes))
    fold_metrics = []

    t0 = time.time()
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n===== Fold {fold+1}/{n_folds} "
              f"(train={len(train_idx)}, val={len(val_idx)}) =====")

        fold_train_ds = Subset(train_ds, train_idx)
        fold_val_ds = Subset(eval_ds, val_idx)

        train_loader = DataLoader(fold_train_ds, batch_size=cfg["BATCH_SIZE"],
                                   shuffle=True, num_workers=cfg["NUM_WORKERS"])
        val_loader = DataLoader(fold_val_ds, batch_size=cfg["BATCH_SIZE"],
                                 shuffle=False, num_workers=cfg["NUM_WORKERS"])

        # inverse-frequency class weights computed from this fold's TRAIN split only
        fold_train_labels = targets[train_idx]
        class_counts = np.array([np.sum(fold_train_labels == c) for c in range(num_classes)])
        class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * num_classes

        model = build_model(cfg["MODEL_NAME"], num_classes, cfg["PRETRAINED"]).to(device)
        model, history, val_labels, val_preds, best_f1 = train_model(
            model, train_loader, val_loader, class_weights, device, cfg,
            epochs=cfg["EPOCHS"], verbose_prefix=f"  [fold {fold+1}] "
        )

        probs, _ = get_probs(model, val_loader, device)
        oof_probs[val_idx] = probs

        acc = accuracy_score(val_labels, val_preds)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(val_labels, val_preds, average="macro", zero_division=0)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(val_labels, val_preds, average="weighted", zero_division=0)
        kappa = cohen_kappa_score(val_labels, val_preds)

        fold_metrics.append({
            "fold": fold + 1, "accuracy": acc,
            "precision_macro": prec_m, "recall_macro": rec_m, "f1_macro": f1_m,
            "precision_weighted": prec_w, "recall_weighted": rec_w, "f1_weighted": f1_w,
            "cohen_kappa": kappa,
        })
        print(f"  Fold {fold+1} results -> acc {acc:.3f} | macro-F1 {f1_m:.3f} | "
              f"weighted-F1 {f1_w:.3f} | kappa {kappa:.3f}")

        oof_labels.extend(val_labels)
        oof_preds.extend(val_preds)

    elapsed = time.time() - t0
    print(f"\nCross-validation finished in {elapsed/60:.1f} min")

    oof_labels = np.array(oof_labels)
    oof_preds = np.array(oof_preds)

    # Persist raw out-of-fold predictions immediately -- this is the
    # expensive part of the run, so it should survive even if a later
    # reporting/plotting step has a bug.
    np.savez(out_dir / "oof_predictions.npz", labels=oof_labels, preds=oof_preds, probs=oof_probs)

    # ---------------- Aggregate CV metrics ----------------
    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        "n_folds": n_folds,
        "per_fold": fold_metrics,
        "mean": metrics_df.drop(columns="fold").mean().to_dict(),
        "std": metrics_df.drop(columns="fold").std().to_dict(),
    }

    try:
        auc = roc_auc_score(oof_labels, oof_probs, multi_class="ovr", average="macro")
        summary["roc_auc_macro_ovr"] = auc
    except ValueError as e:
        summary["roc_auc_macro_ovr"] = None
        print(f"Could not compute ROC-AUC: {e}")

    print("\n===== Cross-validated performance (mean +/- std across folds) =====")
    for k, v in summary["mean"].items():
        print(f"  {k:20s} {v:.4f} +/- {summary['std'][k]:.4f}")
    if summary.get("roc_auc_macro_ovr") is not None:
        print(f"  {'roc_auc_macro_ovr':20s} {summary['roc_auc_macro_ovr']:.4f}")

    if abs(summary["mean"].get("f1_macro", 0) - 1.0) < 1e-9:
        print("\n*** NOTE: macro-F1 is exactly 1.0 across every fold. For a small IRT "
              "fault dataset this is unusual and worth double-checking before trusting "
              "it -- see the message in chat about why, and how to check for it. ***")

    with open(out_dir / "cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    metrics_df.to_csv(out_dir / "cv_per_fold_metrics.csv", index=False)

    # ---------------- Final deployment model on ALL data ----------------
    # Trained and saved BEFORE the report/plotting code below, so the model
    # you actually need for predict.py / camera_inference.py exists on disk
    # even if something downstream has a bug.
    print("\n===== Training final deployment model on 100% of the data =====")
    n_val = max(num_classes, int(0.1 * len(base_ds)))
    rng = np.random.RandomState(cfg["SEED"])
    all_idx = rng.permutation(len(base_ds))
    monitor_idx, deploy_train_idx = all_idx[:n_val], all_idx[n_val:]

    deploy_train_loader = DataLoader(Subset(train_ds, deploy_train_idx), batch_size=cfg["BATCH_SIZE"],
                                      shuffle=True, num_workers=cfg["NUM_WORKERS"])
    monitor_loader = DataLoader(Subset(eval_ds, monitor_idx), batch_size=cfg["BATCH_SIZE"],
                                 shuffle=False, num_workers=cfg["NUM_WORKERS"])

    deploy_train_labels = targets[deploy_train_idx]
    class_counts = np.array([np.sum(deploy_train_labels == c) for c in range(num_classes)])
    class_weights = torch.tensor(1.0 / np.maximum(class_counts, 1), dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * num_classes

    final_model = build_model(cfg["MODEL_NAME"], num_classes, cfg["PRETRAINED"]).to(device)
    final_model, final_history, _, _, _ = train_model(
        final_model, deploy_train_loader, monitor_loader, class_weights, device, cfg,
        epochs=cfg["FINAL_EPOCHS"], verbose_prefix="  [final] "
    )

    torch.save(final_model.state_dict(), out_dir / "motor_fault_model.pth")
    with open(out_dir / "class_names.json", "w") as f:
        json.dump({
            "classes": classes,
            "model_name": cfg["MODEL_NAME"],
            "img_size": cfg["IMG_SIZE"],
            "enable_clahe": cfg["ENABLE_CLAHE"],
        }, f, indent=2)
    print(f"Deployment model saved to {out_dir / 'motor_fault_model.pth'}")

    epochs_range = range(1, len(final_history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs_range, final_history["train_loss"], label="train")
    axes[0].plot(epochs_range, final_history["val_loss"], label="monitor")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(epochs_range, final_history["train_acc"], label="train")
    axes[1].plot(epochs_range, final_history["val_acc"], label="monitor")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    # ---------------- Classification report + confusion matrix ----------------
    # Built from only the classes actually present in the out-of-fold
    # predictions -- robust even if a class ended up entirely absent from
    # every fold's validation split.
    present_ids = sorted(set(oof_labels.tolist()) | set(oof_preds.tolist()))
    present_names = [classes[i] for i in present_ids]

    report_str = classification_report(oof_labels, oof_preds, labels=present_ids,
                                        target_names=present_names, zero_division=0)
    print("\n===== Out-of-fold classification report =====")
    print(report_str)
    with open(out_dir / "cv_classification_report.txt", "w") as f:
        f.write(report_str)

    report_dict = classification_report(oof_labels, oof_preds, labels=present_ids,
                                         target_names=present_names,
                                         output_dict=True, zero_division=0)
    pd.DataFrame(report_dict).transpose().to_csv(out_dir / "cv_classification_report.csv")

    cm = confusion_matrix(oof_labels, oof_preds, labels=present_ids)
    plt.figure(figsize=(max(8, len(present_ids) * 0.6), max(6, len(present_ids) * 0.55)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=present_names, yticklabels=present_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Out-of-fold confusion matrix (every image predicted once)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nDone. Reported metrics are from {n_folds}-fold cross-validation "
          f"(the honest estimate of real-world performance).")
    print(f"The deployment model in '{out_dir}/motor_fault_model.pth' was retrained on "
          f"~100% of the data afterwards and should generally be at least as good, "
          f"but its own accuracy on the tiny monitor split isn't the number to report.")
    print(f"All outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
