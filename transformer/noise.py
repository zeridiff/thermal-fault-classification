"""
noise.py
========
Noise models used in two different roles:

  1. TRAINING AUGMENTATION  (RandomNoise)  -> random kind, random severity.
     Goal: force the CNN to learn thermal patterns that survive sensor noise.

  2. TEST-TIME CORRUPTION   (FixedNoise)   -> one kind, one fixed severity.
     Goal: measure how much accuracy degrades. Never used during training.

All transforms operate on a float tensor in [0, 1], shape (C, H, W), i.e. AFTER
transforms.ToTensor() and BEFORE transforms.Normalize(). That ordering matters:
adding noise after normalisation would make sigma mean something different per
channel and the numbers would not match the literature.

Definitions follow the standard image-processing formulations (equivalent to
MATLAB's imnoise, which is what the reference paper used):

  Gaussian      I' = I + N(0, sigma^2)                  additive, sigma on [0,1]
  Salt&pepper   fraction d of pixels forced to 0 or 1   impulse, d/2 each
  Speckle       I' = I + I * N(0, var) = I*(1 + N)      multiplicative
"""

import random
import torch


# --------------------------------------------------------------------------- #
# Primitive noise functions
# --------------------------------------------------------------------------- #
def add_gaussian(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Additive white Gaussian noise, zero mean, std = sigma."""
    if sigma <= 0:
        return img
    return torch.clamp(img + torch.randn_like(img) * sigma, 0.0, 1.0)


def add_salt_pepper(img: torch.Tensor, density: float) -> torch.Tensor:
    """
    Impulse noise. `density` is the total fraction of corrupted pixels;
    half become salt (1.0) and half become pepper (0.0).

    The same pixels are corrupted across all 3 channels, which is how a real
    dead/hot detector element behaves.
    """
    if density <= 0:
        return img
    out = img.clone()
    c, h, w = out.shape
    mask = torch.rand(1, h, w, device=out.device).expand(c, h, w)
    out = torch.where(mask < density / 2.0, torch.zeros_like(out), out)   # pepper
    out = torch.where(mask > 1.0 - density / 2.0, torch.ones_like(out), out)  # salt
    return out


def add_speckle(img: torch.Tensor, var: float) -> torch.Tensor:
    """Multiplicative noise: I' = I * (1 + N(0, var)). Scales with intensity."""
    if var <= 0:
        return img
    noise = torch.randn_like(img) * (var ** 0.5)
    return torch.clamp(img + img * noise, 0.0, 1.0)


NOISE_FUNCS = {
    "gaussian": add_gaussian,
    "salt_pepper": add_salt_pepper,
    "speckle": add_speckle,
}


# --------------------------------------------------------------------------- #
# Transform classes (drop into a torchvision Compose)
# --------------------------------------------------------------------------- #
class RandomNoise:
    """
    Training augmentation. With probability `p`, pick one noise kind at random
    and one severity at random from its configured range.

    Sampling the severity (rather than fixing it) is deliberate: it stops the
    network from overfitting to one specific noise level, which is what you want
    if the claim is 'robust across operating conditions'.
    """

    def __init__(self, p, kinds, gaussian_sigma, saltpepper_density, speckle_var):
        self.p = p
        self.kinds = tuple(kinds)
        self.ranges = {
            "gaussian": gaussian_sigma,
            "salt_pepper": saltpepper_density,
            "speckle": speckle_var,
        }

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p or not self.kinds:
            return img
        kind = random.choice(self.kinds)
        lo, hi = self.ranges[kind]
        severity = random.uniform(lo, hi)
        return NOISE_FUNCS[kind](img, severity)

    def __repr__(self):
        return f"RandomNoise(p={self.p}, kinds={self.kinds})"


class FixedNoise:
    """Test-time corruption at one known severity. Deterministic in its settings."""

    def __init__(self, kind: str, severity: float):
        if kind not in NOISE_FUNCS:
            raise ValueError(f"unknown noise kind '{kind}', expected one of {list(NOISE_FUNCS)}")
        self.kind = kind
        self.severity = severity

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return NOISE_FUNCS[self.kind](img, self.severity)

    def __repr__(self):
        return f"FixedNoise({self.kind}, {self.severity})"
