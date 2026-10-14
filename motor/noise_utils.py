"""Synthetic corruption of display-image RGB values in [0, 1]. No filtering."""
import hashlib
import numpy as np


def noise_seed(seed, fold, family, level, repeat, sample_id):
    key = f'{seed}|{fold}|{family}|{level:.8f}|{repeat}|{sample_id}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def add_noise(image, family, level, rng):
    """HWC RGB float array. Gaussian level=std; speckle level=variance.
    Salt-and-pepper level=total corrupted pixel probability (half each).
    """
    x = np.asarray(image, dtype=np.float32)
    if level < 0:
        raise ValueError('Noise strength cannot be negative.')
    if level == 0 or family == 'clean':
        return x.copy()
    if family == 'gaussian':
        y = x + rng.normal(0, level, x.shape)
    elif family == 'speckle':
        y = x + x * rng.normal(0, np.sqrt(level), x.shape)
    elif family == 'salt_pepper':
        if level > 1:
            raise ValueError('Salt-and-pepper probability must be at most 1.')
        y = x.copy()
        mask = rng.random(x.shape[:2])
        y[mask < level / 2] = 0
        y[(mask >= level / 2) & (mask < level)] = 1
    else:
        raise ValueError(f'Unknown noise family: {family}')
    return np.clip(y, 0, 1).astype(np.float32)
