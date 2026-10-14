"""
filters.py
==========
Denoising filters for the transformer thermography robustness experiment.

All functions expect an RGB NumPy array in float32 [0, 1] with shape (H, W, 3)
and return the same shape/range. Filtering is applied AFTER synthetic corruption
and BEFORE ImageNet normalisation, matching the existing noise.py/data.py order.

The presets are intentionally mild. Strong smoothing can erase the thermal
boundaries/hot-spot patterns that the CNN needs for fault-severity classification.
"""

from __future__ import annotations

import cv2
import numpy as np


FILTER_NAMES = (
    "none",
    "median3",
    "median5",
    "gaussian3",
    "gaussian5",
    "bilateral",
    "nlm",
)


def _check(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB HxWx3 image, got shape {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def apply_filter(img: np.ndarray, kind: str) -> np.ndarray:
    """Apply one named denoising preset to an RGB float image in [0, 1]."""
    x = _check(img)
    kind = kind.lower()

    if kind == "none":
        return x.copy()

    if kind == "median3":
        return np.clip(cv2.medianBlur(x, 3), 0.0, 1.0)

    if kind == "median5":
        return np.clip(cv2.medianBlur(x, 5), 0.0, 1.0)

    if kind == "gaussian3":
        return np.clip(
            cv2.GaussianBlur(x, (3, 3), sigmaX=0.8, sigmaY=0.8,
                             borderType=cv2.BORDER_REFLECT101),
            0.0, 1.0,
        )

    if kind == "gaussian5":
        return np.clip(
            cv2.GaussianBlur(x, (5, 5), sigmaX=1.2, sigmaY=1.2,
                             borderType=cv2.BORDER_REFLECT101),
            0.0, 1.0,
        )

    if kind == "bilateral":
        # sigmaColor is on the same scale as pixel values (here 0..1).
        # This is deliberately mild so that hotspot edges are not flattened.
        return np.clip(
            cv2.bilateralFilter(x, d=5, sigmaColor=0.10, sigmaSpace=3.0,
                                borderType=cv2.BORDER_REFLECT101),
            0.0, 1.0,
        )

    if kind == "nlm":
        # OpenCV's coloured non-local-means implementation expects uint8.
        u8 = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
        den = cv2.fastNlMeansDenoisingColored(
            u8, None, h=5, hColor=5, templateWindowSize=7, searchWindowSize=21
        )
        return den.astype(np.float32) / 255.0

    raise ValueError(f"Unknown filter '{kind}'. Choices: {FILTER_NAMES}")
