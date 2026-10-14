"""
data.py
=======
Reads the IR_trans_bmp folder, builds a stratified split, and defines the
train / eval transform pipelines.

Two rules are enforced here and they are the difference between a believable
result and a meaningless one:

  1. SPLIT BEFORE AUGMENT. Augmented copies of a training image must never end
     up in the test set. Because augmentation happens inside the Dataset's
     transform (on the fly), and the split is on file indices, this is
     structurally guaranteed.

  2. STRATIFY. Class sizes run from 22 (Healthy) to 40 (600 SC). A random split
     can easily leave a class with 2 test images, which makes per-class recall
     meaningless.
"""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config
from noise import RandomNoise, FixedNoise


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Scanning the dataset folder
# --------------------------------------------------------------------------- #
def build_index(root: Path = None):
    """
    Walk the dataset root and return (paths, labels, class_names).

    Expects one sub-folder per class:
        IR_trans_bmp/p1_Noload/*.bmp
        IR_trans_bmp/p2_80/*.bmp
        ...
    Folder names are sorted alphabetically, which yields the correct severity
    ordering p1 ... p9 for this dataset.
    """
    root = Path(root or config.DATA_ROOT)
    if not root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT does not exist:\n  {root}\n"
            "Open config.py and set DATA_ROOT to the folder that directly "
            "contains p1_Noload, p2_80, ... p9_600."
        )

    class_dirs = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda d: d.name)
    if not class_dirs:
        raise RuntimeError(f"No class sub-folders found inside {root}")

    paths, labels = [], []
    for label, d in enumerate(class_dirs):
        files = sorted(p for p in d.rglob("*") if p.suffix.lower() in config.IMG_EXTENSIONS)
        if not files:
            raise RuntimeError(f"No images with extensions {config.IMG_EXTENSIONS} inside {d}")
        paths.extend(files)
        labels.extend([label] * len(files))

    class_names = [config.PRETTY_NAMES.get(d.name, d.name) for d in class_dirs]
    return paths, np.asarray(labels, dtype=np.int64), class_names


def describe(labels: np.ndarray, class_names) -> str:
    lines = [f"{'class':<12}{'n':>5}{'share':>9}"]
    total = len(labels)
    for i, name in enumerate(class_names):
        n = int((labels == i).sum())
        lines.append(f"{name:<12}{n:>5}{n / total:>8.1%}")
    lines.append(f"{'TOTAL':<12}{total:>5}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class IRThermalDataset(Dataset):
    """Thermograms are stored as 24-bit BMP colour-palette images -> load as RGB."""

    def __init__(self, paths, labels, transform=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[idx])


def subset(paths, labels, idx):
    return [paths[i] for i in idx], labels[idx]


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def train_transform(with_noise: bool = None):
    """
    Geometry + photometric + (optionally) noise injection.

    Order matters: PIL-space geometry -> ToTensor -> noise in [0,1] -> Normalize.
    """
    with_noise = config.NOISE_AUG_ENABLED if with_noise is None else with_noise

    steps = [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=config.AUG_HFLIP_P),
        transforms.RandomAffine(
            degrees=config.AUG_ROTATION_DEG,
            translate=(config.AUG_TRANSLATE, config.AUG_TRANSLATE),
            scale=config.AUG_SCALE,
            shear=config.AUG_SHEAR_DEG,
        ),
        transforms.ColorJitter(
            brightness=config.AUG_BRIGHTNESS,
            contrast=config.AUG_CONTRAST,
        ),
        transforms.ToTensor(),
    ]

    if with_noise:
        steps.append(
            RandomNoise(
                p=config.NOISE_AUG_P,
                kinds=config.NOISE_AUG_KINDS,
                gaussian_sigma=config.AUG_GAUSSIAN_SIGMA,
                saltpepper_density=config.AUG_SALTPEPPER_DENSITY,
                speckle_var=config.AUG_SPECKLE_VAR,
            )
        )

    steps.append(transforms.Normalize(config.MEAN, config.STD))
    return transforms.Compose(steps)


def eval_transform(noise_kind: str = None, severity: float = 0.0):
    """
    Deterministic resize + optional fixed corruption. Used for validation, test,
    and the noise-robustness sweep.
    """
    steps = [
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ]
    if noise_kind is not None and severity > 0:
        steps.append(FixedNoise(noise_kind, severity))
    steps.append(transforms.Normalize(config.MEAN, config.STD))
    return transforms.Compose(steps)


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def stratified_split(labels, test_size, seed=config.SEED, indices=None):
    """Return (idx_a, idx_b) preserving class proportions."""
    idx = np.arange(len(labels)) if indices is None else np.asarray(indices)
    a, b = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels[idx], shuffle=True
    )
    return a, b


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def make_loader(paths, labels, transform, batch_size=None, shuffle=False):
    ds = IRThermalDataset(paths, labels, transform)
    return DataLoader(
        ds,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def class_weights(labels, n_classes):
    """Inverse-frequency weights, normalised to mean 1.0."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)
