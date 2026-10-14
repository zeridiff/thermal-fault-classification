"""
gradcam.py  --  what is the network actually looking at?
========================================================

Produces Grad-CAM heat maps over the deepest convolutional feature map.
Red = regions that pushed the prediction toward the chosen class.

Why this belongs in a fault-diagnosis report: a model can reach high accuracy by
latching onto something irrelevant - a bench edge, a cable, a reflection that
happens to correlate with the day a class was captured. Grad-CAM is the cheapest
evidence that the network is responding to the winding hot-spot rather than the
background. If the heat lands off the transformer, the accuracy number is not
trustworthy no matter how high it is.

Usage:
    python gradcam.py --checkpoint results/noiseaug_kfold/model_fold1.pt
    python gradcam.py --checkpoint <path> --per-class 2 --noise gaussian --severity 0.1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
import data as D
from engine import get_device
from model import build_model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        logits[0, class_idx].backward()

        # channel weights = spatially averaged gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        return cam, class_idx, probs


def main():
    ap = argparse.ArgumentParser(description="Grad-CAM heat maps")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--noise", default=None,
                    choices=["gaussian", "salt_pepper", "speckle"])
    ap.add_argument("--severity", type=float, default=0.0)
    ap.add_argument("--out", default="gradcam")
    args = ap.parse_args()

    D.set_seed(config.SEED)
    device = get_device()
    paths, labels, class_names = D.build_index()

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(num_classes=len(ck["class_names"]),
                        backbone=ck.get("backbone", config.BACKBONE),
                        pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device)
    cam_engine = GradCAM(model, model.last_conv_layer())

    tf = D.eval_transform(noise_kind=args.noise, severity=args.severity)
    inv_mean = torch.tensor(config.MEAN).view(3, 1, 1)
    inv_std = torch.tensor(config.STD).view(3, 1, 1)

    picks = []
    for c in range(len(class_names)):
        idxs = np.where(labels == c)[0][: args.per_class]
        picks.extend((int(i), c) for i in idxs)

    ncol = min(6, max(1, len(picks)))
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (idx, true_c) in zip(axes, picks):
        img = Image.open(paths[idx]).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        cam, pred_c, probs = cam_engine(x)

        base = (x[0].detach().cpu() * inv_std + inv_mean).permute(1, 2, 0).numpy()
        ax.imshow(np.clip(base, 0, 1))
        ax.imshow(cam, cmap="jet", alpha=0.42)
        ok = "OK" if pred_c == true_c else "MISS"
        ax.set_title(f"true {class_names[true_c]} | pred {class_names[pred_c]}\n"
                     f"p={probs[pred_c]:.2f}  [{ok}]", fontsize=8)
        ax.axis("off")

    for ax in axes[len(picks):]:
        ax.axis("off")

    subtitle = "clean" if not args.noise else f"{args.noise} @ {args.severity}"
    fig.suptitle(f"Grad-CAM on the fused deep feature map ({subtitle})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.2)

    out_dir = Path(config.OUT_DIR) / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"gradcam_{args.noise or 'clean'}.png"
    fig.savefig(out_dir / name, dpi=170)
    plt.close(fig)
    print(f"Written: {(out_dir / name).resolve()}")


if __name__ == "__main__":
    main()
