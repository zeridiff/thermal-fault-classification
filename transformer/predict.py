"""
predict.py  --  point it at one thermogram and get a diagnosis.
===============================================================

This is the demo script. Give it an image and it tells you the predicted fault
class, the confidence, the top-3 alternatives, and shows a Grad-CAM heat map of
where the network looked.

Two features that matter for a presentation:

  HELD-OUT VERIFICATION. Every fold model saved the list of images it was tested
  on. This script automatically picks the fold whose held-out set contains your
  image, so the prediction is always made by a model that never saw it during
  training. It prints a clear warning if that guarantee cannot be met. If a
  marker asks "was that image in your training data?", the answer is on screen.

  LIVE ROBUSTNESS DEMO (--compare). Runs the noise-augmented model and the
  geometry-only baseline on the same corrupted image, side by side. At around
  1% salt-and-pepper the baseline typically breaks while yours holds. That is
  your whole contribution in one slide.

Usage
-----
    # basic
    python predict.py --image "C:\\...\\IR_trans_bmp\\p5_320\\IM_0123.bmp"

    # with noise applied
    python predict.py --image "<path>" --noise salt_pepper --severity 0.01

    # the money shot: both models, same corrupted image
    python predict.py --image "<path>" --noise salt_pepper --severity 0.01 --compare

    # pick a random test image for you
    python predict.py --random --noise gaussian --severity 0.10 --compare
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
import data as D
from engine import get_device
from gradcam import GradCAM
from model import build_model


# --------------------------------------------------------------------------- #
def find_holdout_checkpoint(run_dir: Path, image_index: int):
    """
    Return the fold checkpoint that held this image out of training.

    Under k-fold every image sits in exactly one test fold, so this normally
    succeeds. Returns (path, True) when found, or (first checkpoint, False) when
    it could not be guaranteed.
    """
    ckpts = sorted(run_dir.glob("model_fold*.pt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No model_fold*.pt inside {run_dir}.\n"
            f"Run:  python train.py --name {run_dir.name}"
        )
    for ck in ckpts:
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        if image_index in set(meta.get("test_idx", [])):
            return ck, True
    return ckpts[0], False


def load_model(ckpt_path, device):
    meta = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = build_model(num_classes=len(meta["class_names"]),
                      backbone=meta.get("backbone", config.BACKBONE),
                      pretrained=False)
    net.load_state_dict(meta["state_dict"])
    net.to(device).eval()
    return net, meta


def predict_one(net, tensor, device):
    """Returns (probs, cam, pred_idx). Grad-CAM needs gradients, so no no_grad here."""
    cam_engine = GradCAM(net, net.last_conv_layer())
    cam, pred_idx, probs = cam_engine(tensor.unsqueeze(0).to(device))
    return probs, cam, pred_idx


# --------------------------------------------------------------------------- #
def panel(ax_img, ax_bar, base_img, cam, probs, class_names, true_name, title):
    ax_img.imshow(np.clip(base_img, 0, 1))
    ax_img.imshow(cam, cmap="jet", alpha=0.40)
    ax_img.axis("off")

    pred_i = int(np.argmax(probs))
    correct = (true_name is not None and class_names[pred_i] == true_name)
    verdict = "" if true_name is None else ("  CORRECT" if correct else "  WRONG")
    ax_img.set_title(f"{title}\npredicted: {class_names[pred_i]}  "
                     f"({probs[pred_i]*100:.1f}%){verdict}",
                     fontsize=10,
                     color=("black" if true_name is None else
                            ("darkgreen" if correct else "firebrick")))

    order = np.argsort(probs)[::-1]
    colors = ["tab:blue"] * len(probs)
    colors[pred_i] = "darkgreen" if correct or true_name is None else "firebrick"
    ax_bar.barh([class_names[i] for i in order][::-1],
                [probs[i] for i in order][::-1],
                color=[colors[i] for i in order][::-1])
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("probability", fontsize=9)
    ax_bar.tick_params(labelsize=8)
    ax_bar.grid(axis="x", alpha=0.3)


def main():
    ap = argparse.ArgumentParser(description="Diagnose a single thermogram")
    ap.add_argument("--image", default=None, help="Path to a .bmp thermogram")
    ap.add_argument("--random", action="store_true",
                    help="Pick a random image from the dataset instead")
    ap.add_argument("--run", default="noiseaug", help="Run folder under results/")
    ap.add_argument("--baseline-run", default="geomonly",
                    help="Run folder to compare against when --compare is used")
    ap.add_argument("--noise", default=None,
                    choices=["gaussian", "salt_pepper", "speckle"])
    ap.add_argument("--severity", type=float, default=0.0)
    ap.add_argument("--compare", action="store_true",
                    help="Show the noise-augmented model and the baseline side by side")
    ap.add_argument("--out", default=None, help="Output PNG path")
    args = ap.parse_args()

    D.set_seed(config.SEED)
    device = get_device()
    paths, labels, class_names = D.build_index()

    # ---- locate the image inside the dataset index ----------------------- #
    if args.random:
        idx = int(np.random.default_rng().integers(len(paths)))
    elif args.image:
        target = Path(args.image).resolve()
        matches = [i for i, p in enumerate(paths) if Path(p).resolve() == target]
        if not matches:
            raise FileNotFoundError(
                f"{target}\nis not inside DATA_ROOT ({config.DATA_ROOT}).\n"
                "Use an image from the dataset so the held-out check can run."
            )
        idx = matches[0]
    else:
        ap.error("give either --image <path> or --random")

    img_path = Path(paths[idx])
    true_name = class_names[int(labels[idx])]

    print("=" * 66)
    print("SINGLE-IMAGE DIAGNOSIS")
    print("=" * 66)
    print(f"image      : {img_path.name}")
    print(f"true class : {true_name}")
    print(f"corruption : {args.noise or 'none'}"
          + (f" @ {args.severity}" if args.noise else ""))

    tf = D.eval_transform(noise_kind=args.noise, severity=args.severity)
    pil = Image.open(img_path).convert("RGB")
    tensor = tf(pil)

    inv_mean = torch.tensor(config.MEAN).view(3, 1, 1)
    inv_std = torch.tensor(config.STD).view(3, 1, 1)
    base_img = (tensor * inv_std + inv_mean).permute(1, 2, 0).numpy()

    runs = [args.run] + ([args.baseline_run] if args.compare else [])
    results = []

    for run_name in runs:
        run_dir = Path(config.OUT_DIR) / run_name
        ck, held_out = find_holdout_checkpoint(run_dir, idx)
        net, meta = load_model(ck, device)
        probs, cam, pred_i = predict_one(net, tensor, device)
        results.append((run_name, ck, held_out, probs, cam))

        print("-" * 66)
        print(f"model      : {run_name}  ({ck.name})")
        if held_out:
            print("             this image was HELD OUT of that model's training set")
        else:
            print("             !! could not confirm this image was held out --")
            print("                the prediction may be optimistic")
        print(f"prediction : {class_names[pred_i]}   confidence {probs[pred_i]*100:.1f}%")
        print(f"correct    : {'YES' if class_names[pred_i] == true_name else 'NO'}")
        top3 = np.argsort(probs)[::-1][:3]
        print("top-3      : " + ",  ".join(
            f"{class_names[i]} {probs[i]*100:.1f}%" for i in top3))

    # ---- figure ----------------------------------------------------------- #
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5.4 * n, 8.6), squeeze=False)
    for c, (run_name, ck, held_out, probs, cam) in enumerate(results):
        tag = f"{run_name}" + ("" if held_out else "  [NOT held out]")
        panel(axes[0][c], axes[1][c], base_img, cam, probs, class_names,
              true_name, tag)

    sub = "clean image" if not args.noise else f"{args.noise} noise @ {args.severity}"
    fig.suptitle(f"{img_path.name}   |   true class: {true_name}   |   {sub}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)

    out_dir = Path(config.OUT_DIR) / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / (
        f"{img_path.stem}_{args.noise or 'clean'}"
        f"{'_compare' if args.compare else ''}.png")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    print("=" * 66)
    print(f"figure saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
