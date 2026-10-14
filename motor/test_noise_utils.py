"""Numerical noise checks; no training or motor dataset required."""
import unittest
import numpy as np
from noise_utils import add_noise, noise_seed


class NoiseTests(unittest.TestCase):
    def test_zero_is_clean(self):
        x = np.full((10, 10, 3), 0.5, dtype=np.float32)
        for family in ['gaussian', 'salt_pepper', 'speckle']:
            np.testing.assert_array_equal(add_noise(x, family, 0, np.random.default_rng(1)), x)

    def test_distribution_and_clipping(self):
        x = np.full((500, 500, 3), 0.5, dtype=np.float32)
        gaussian = add_noise(x, 'gaussian', 0.05, np.random.default_rng(10))
        self.assertAlmostEqual(float((gaussian-x).std()), 0.05, delta=0.001)
        speckle = add_noise(x, 'speckle', 0.01, np.random.default_rng(10))
        self.assertAlmostEqual(float(((speckle-x)/x).var()), 0.01, delta=0.0002)
        salt = add_noise(x, 'salt_pepper', 0.1, np.random.default_rng(10))
        self.assertAlmostEqual(float(np.mean(salt[:, :, 0] != 0.5)), 0.1, delta=0.002)
        np.testing.assert_array_equal(salt[:, :, 0], salt[:, :, 2])
        for family in ['gaussian', 'speckle', 'salt_pepper']:
            y = add_noise(x, family, 1.0, np.random.default_rng(10))
            self.assertTrue(np.isfinite(y).all() and y.min() >= 0 and y.max() <= 1)

    def test_paired_test_noise(self):
        x = np.full((30, 30, 3), 0.5, dtype=np.float32)
        seed = noise_seed(42, 1, 'gaussian', 0.1, 0, 6)
        a = add_noise(x, 'gaussian', 0.1, np.random.default_rng(seed))
        b = add_noise(x, 'gaussian', 0.1, np.random.default_rng(seed))
        np.testing.assert_array_equal(a, b)
        other = noise_seed(42, 1, 'gaussian', 0.1, 1, 6)
        self.assertNotEqual(seed, other)
        self.assertFalse(np.array_equal(a, add_noise(x, 'gaussian', 0.1, np.random.default_rng(other))))


if __name__ == '__main__':
    unittest.main()
