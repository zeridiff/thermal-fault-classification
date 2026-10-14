"""
inspect_data.py  --  RUN THIS FIRST.
====================================

Before training anything, confirm the data is actually where you think it is and
looks like you think it does. This script:

  1. Prints the class distribution and checks it against the published counts
     (22, 26, 31, 26, 26, 25, 27, 32, 40 = 255 images).
  2. Reports image sizes and modes, and flags any file that will not open.
  3. Detects byte-identical duplicate files, which would leak across the split.
  4. Saves a one-sample-per-class contact sheet.
  5. Saves a noise demonstration figure - the same thermogram at every noise type
     and severity used in the experiments. This figure belongs in the report:
     it shows the reader exactly what the model is being asked to survive.

Outputs go to results/inspection/.
"""

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import config
import data as D
from noise import add_gaussian, add_salt_pepper, add_speckle

PUBLISHED_COUNTS = [22, 26, 31, 26, 26, 25, 27, 32, 40]   # from the dataset README


def file_hash(path, chunk=1 << 16):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out_dir = Path(config.OUT_DIR) / "inspection"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, class_names = D.build_index()

    print("=" * 66)
    print("DATASET INSPECTION")
    print("=" * 66)
    print(f"root: {config.DATA_ROOT}\n")
    print(D.describe(labels, class_names))

    counts = [int((labels == i).sum()) for i in range(len(class_names))]
    if len(counts) == len(PUBLISHED_COUNTS):
        match = counts == PUBLISHED_COUNTS
        print(f"\nMatches published counts {PUBLISHED_COUNTS}? {'YES' if match else 'NO'}")
        if not match:
            print("  found:", counts)
            print("  -> check for stray files, or a nested folder inside a class folder")

    # ---- sizes, modes, unreadable files ---------------------------------- #
    sizes, modes, bad = Counter(), Counter(), []
    hashes = defaultdict(list)
    for p in paths:
        try:
            with Image.open(p) as im:
                sizes[im.size] += 1
                modes[im.mode] += 1
        except Exception as e:
            bad.append((p, repr(e)))
        hashes[file_hash(p)].append(p)

    print("\nImage sizes :", dict(sizes))
    print("Image modes :", dict(modes))
    if bad:
        print(f"\n!! {len(bad)} unreadable file(s):")
        for p, e in bad[:10]:
            print("   ", p, e)
    else:
        print("All files opened successfully.")

    dupes = {h: ps for h, ps in hashes.items() if len(ps) > 1}
    if dupes:
        print(f"\n!! {len(dupes)} group(s) of byte-identical duplicates found.")
        print("   Duplicates split across train and test inflate accuracy. Consider removing:")
        for ps in list(dupes.values())[:5]:
            for p in ps[1:]:
                print("   ", p)
    else:
        print("No duplicate files detected.")

    # ---- contact sheet: one image per class ------------------------------ #
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    for i, ax in enumerate(axes.ravel()):
        if i < len(class_names):
            idx = int(np.where(labels == i)[0][0])
            ax.imshow(Image.open(paths[idx]).convert("RGB"))
            n = int((labels == i).sum())
            ax.set_title(f"{class_names[i]}  (n={n})", fontsize=10)
        ax.axis("off")
    fig.suptitle("One thermogram per class", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "class_samples.png", dpi=180)
    plt.close(fig)

    # ---- noise demonstration --------------------------------------------- #
    to_tensor = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])
    sample = to_tensor(Image.open(paths[int(np.where(labels == len(class_names) - 1)[0][0])]).convert("RGB"))

    grids = [
        ("Gaussian (sigma)", add_gaussian, config.TEST_GAUSSIAN_SIGMAS),
        ("Salt & pepper (density)", add_salt_pepper, config.TEST_SALTPEPPER_DENSITIES),
        ("Speckle (variance)", add_speckle, config.TEST_SPECKLE_VARS),
    ]
    ncol = max(len(g[2]) for g in grids)
    fig, axes = plt.subplots(len(grids), ncol, figsize=(2.3 * ncol, 2.7 * len(grids)))
    torch.manual_seed(config.SEED)
    for r, (title, fn, sevs) in enumerate(grids):
        for c in range(ncol):
            ax = axes[r, c]
            if c < len(sevs):
                img = fn(sample.clone(), sevs[c]).permute(1, 2, 0).numpy()
                ax.imshow(np.clip(img, 0, 1))
                ax.set_title(f"{sevs[c]:g}" if c else f"{sevs[c]:g}  (clean)", fontsize=9)
            ax.axis("off")
        # row label on the far left
        pos = axes[r, 0].get_position()
        fig.text(0.005, pos.y0 + pos.height / 2, title, rotation=90,
                 va="center", ha="left", fontsize=10, weight="bold")
    fig.suptitle("Test-time corruptions applied to a single thermogram", fontsize=12)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.savefig(out_dir / "noise_examples.png", dpi=180)
    plt.close(fig)

    # ---- what the model actually sees during training -------------------- #
    train_tf = D.train_transform(with_noise=True)
    img_pil = Image.open(paths[0]).convert("RGB")
    inv_mean = torch.tensor(config.MEAN).view(3, 1, 1)
    inv_std = torch.tensor(config.STD).view(3, 1, 1)

    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for ax in axes.ravel():
        t = train_tf(img_pil) * inv_std + inv_mean
        ax.imshow(np.clip(t.permute(1, 2, 0).numpy(), 0, 1))
        ax.axis("off")
    fig.suptitle("Ten random draws of the training augmentation pipeline "
                 "(geometry + brightness + injected noise)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "augmentation_examples.png", dpi=180)
    plt.close(fig)

    print(f"\nFigures written to: {out_dir.resolve()}")
    print("  class_samples.png        one thermogram per class")
    print("  noise_examples.png       corruption severities used at test time")
    print("  augmentation_examples.png  what the network sees during training")


if __name__ == "__main__":
    main()
