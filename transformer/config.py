"""
config.py
=========
Every setting for the project lives here. This is the ONLY file you normally
need to edit.

Project: Inter-turn fault detection & short-circuit severity classification in a
1-phase dry-type transformer from infrared thermograms (IRT).

Dataset: Najafi, M.; Baleghi, Y.; Mirimani, S. M. (2020),
         "Thermal images_1-phase_dry type_Transformer", Mendeley Data V1,
         doi: 10.17632/8mg8mkc7k5.1
         255 BMP images, 9 classes (1 healthy + 8 short-circuit severities).
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. PATHS  --  EDIT THIS ONE LINE
# --------------------------------------------------------------------------- #
# Point this at the folder that directly contains p1_Noload, p2_80, ... p9_600.
# Windows users: keep the r"..." prefix so backslashes are treated literally.
DATA_ROOT = Path(r"C:\Users\dukeb\Downloads\8mg8mkc7k5-3 (4)\IR_trans_bmp")

# Everything the scripts produce (models, plots, CSVs) goes here.
OUT_DIR = Path(__file__).parent / "results"

# --------------------------------------------------------------------------- #
# 2. CLASSES
# --------------------------------------------------------------------------- #
# Folder names are read automatically and sorted alphabetically, which for this
# dataset gives the correct severity order p1 -> p9. The map below is only used
# to print/plot nicer labels.
PRETTY_NAMES = {
    "p1_Noload": "Healthy",
    "p2_80":     "80 SC",
    "p3_160":    "160 SC",
    "p4_240":    "240 SC",
    "p5_320":    "320 SC",
    "p6_400":    "400 SC",
    "p7_480":    "480 SC",
    "p8_560":    "560 SC",
    "p9_600":    "600 SC",
}
IMG_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")

# --------------------------------------------------------------------------- #
# 3. REPRODUCIBILITY
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# 4. MODEL
# --------------------------------------------------------------------------- #
BACKBONE = "resnet18"        # "resnet18" or "mobilenet_v2"
PRETRAINED = True            # ImageNet transfer learning (needs internet once)
IMG_SIZE = 224               # 224 x 224 x 3, same as the reference paper
DROPOUT = 0.4
FREEZE_BACKBONE = False      # True = train head only. ~8x faster, lower ceiling.

# --------------------------------------------------------------------------- #
# 5. TRAINING
# --------------------------------------------------------------------------- #
EPOCHS = 30
BATCH_SIZE = 16
LR_HEAD = 1e-3               # new classifier layers learn fast
LR_BACKBONE = 1e-4           # pretrained weights are nudged, not shoved
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adamw"          # "adamw" or "sgd"
MOMENTUM = 0.9               # only used when OPTIMIZER == "sgd"
LABEL_SMOOTHING = 0.05
USE_CLASS_WEIGHTS = True     # classes range from 22 to 40 images
EARLY_STOP_PATIENCE = 10     # epochs without val macro-F1 improvement
NUM_WORKERS = 0              # 0 is the safe choice on Windows / PyCharm

# --------------------------------------------------------------------------- #
# 6. EVALUATION PROTOCOL
# --------------------------------------------------------------------------- #
# "kfold"  -> stratified K-fold cross-validation over all 255 images (reported
#             as mean +/- std). This is the honest protocol for a small dataset
#             and mirrors the reference paper's 10-fold validation.
# "single" -> one stratified train/val/test split. Much faster, use it while you
#             are still debugging.
PROTOCOL = "kfold"
N_FOLDS = 5
VAL_FRACTION = 0.15          # carved out of the training portion, for early stopping
TEST_FRACTION = 0.20         # only used when PROTOCOL == "single"

# --------------------------------------------------------------------------- #
# 7. AUGMENTATION  --  THIS IS THE PART THAT DIFFERS FROM THE PAPER
# --------------------------------------------------------------------------- #
# The reference paper augments with geometry only (flip / rotate / shear /
# translate / scale) and then injects Gaussian and speckle noise ONLY at test
# time, to measure how much the accuracy collapses.
#
# This project instead injects noise during TRAINING as well, so the network
# learns noise-invariant thermal features. Test-time noise is still applied
# separately in noise_robustness.py, so the two conditions stay comparable.
#
# NOTE: augmentation is applied to training images only, and always AFTER the
# train/test split. Augmenting before splitting leaks near-duplicate images into
# the test set and inflates accuracy - a common flaw in this literature.

# -- geometric (kept, so the comparison with the paper stays fair) --
AUG_HFLIP_P = 0.5
AUG_ROTATION_DEG = 15
AUG_TRANSLATE = 0.10         # fraction of image width/height
AUG_SCALE = (0.85, 1.15)
AUG_SHEAR_DEG = 8
AUG_BRIGHTNESS = 0.15        # +/- 15%, mimics gain drift between captures
AUG_CONTRAST = 0.15

# -- noise injection (the new part) --
NOISE_AUG_ENABLED = True
NOISE_AUG_P = 0.5            # probability that a training image gets noised
NOISE_AUG_KINDS = ("gaussian", "salt_pepper", "speckle")

# Ranges are sampled uniformly per image so the model sees many severities.
# sigma is the std-dev of additive noise on a [0, 1] image (var = sigma^2).
AUG_GAUSSIAN_SIGMA = (0.01, 0.08)
# density = fraction of pixels forced to pure black or pure white.
AUG_SALTPEPPER_DENSITY = (0.005, 0.05)
# multiplicative noise variance: I_noisy = I * (1 + N(0, var)).
AUG_SPECKLE_VAR = (0.01, 0.08)

# --------------------------------------------------------------------------- #
# 8. NOISE ROBUSTNESS SWEEP (test time, run by noise_robustness.py)
# --------------------------------------------------------------------------- #
# Fixed severities, applied to the held-out test images only. Same grid is used
# for the noise-trained and the geometry-only model, which is what makes the
# comparison a controlled experiment.
TEST_GAUSSIAN_SIGMAS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)
TEST_SALTPEPPER_DENSITIES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15)
TEST_SPECKLE_VARS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20)

# --------------------------------------------------------------------------- #
# 9. IMAGENET NORMALISATION (do not change while PRETRAINED = True)
# --------------------------------------------------------------------------- #
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
