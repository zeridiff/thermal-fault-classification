# Induction Motor Thermal Fault Detection (CNN)

Trains a transfer-learned CNN to classify your induction-motor thermal
(IRT) images into fault classes, using stratified k-fold cross-validation
so the accuracy/F1 numbers are trustworthy on a dataset this size (~369
images spread across ~11 classes).

## 1. Setup (on your PC, Windows)

```bat
cd "C:\Users\dukeb\Downloads\Motor motor rrrrrrr"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU and want it used, install the CUDA build of
PyTorch instead (see https://pytorch.org/get-started/locally/) before
`pip install -r requirements.txt` — otherwise the CPU build works fine,
just slower.

## 2. Point the script at your data

Open `train_cnn.py` and check the `CONFIG["DATA_DIR"]` value near the top
— it's already set to the path from your screenshot
(`C:\Users\dukeb\Downloads\Motor motor rrrrrrr`). The script expects one
sub-folder per class inside that folder (which is exactly what you have:
`Noload`, `Rotor-0`, `Fan`, `A50`, `A30`, `A10`, `A&C30`, `A&C10`,
`A&C&B30`, `A&C&B10`, `A&B50`, ...). It doesn't matter how many classes
there are or what they're named — the folder name becomes the label
automatically.

## 3. Train

```bat
python train_cnn.py
```

On CPU, with the default `mobilenet_v3_small` backbone and ~369 images,
expect somewhere around 15–40 minutes depending on your machine (5 folds
× up to 40 epochs each, with early stopping usually cutting that short).

## 4. What you get, in `outputs/`

| File | What it is |
|---|---|
| `cv_classification_report.txt` / `.csv` | Per-class precision, recall, F1 |
| `cv_summary.json`, `cv_per_fold_metrics.csv` | Accuracy, macro/weighted F1, Cohen's kappa, ROC-AUC — mean ± std across folds |
| `confusion_matrix.png` | Confusion matrix built from out-of-fold predictions (every image predicted exactly once, by the fold that held it out) |
| `training_curves.png` | Loss/accuracy curves for the final model |
| `motor_fault_model.pth` + `class_names.json` | The trained model, for `predict.py` / `camera_inference.py` |

**Report the cross-validation numbers, not a single train/test split.**
With ~25–42 images per class, one random split can look great or terrible
by luck; the k-fold numbers (mean ± std) are the defensible number for a
report or thesis.

## 5. Run inference

```bat
python predict.py path\to\some_image.jpg
python predict.py path\to\a_folder_of_images
```

## 6. About "using my laptop camera" — important

Your laptop's built-in webcam captures ordinary visible-light video. The
model is trained on thermal (infrared) images, where colour encodes
*temperature*, not reflected light — physically a different signal. A
model trained this way **will not give meaningful fault predictions from
a normal webcam**, no matter how the code is written; it's not something
more training data or a bigger model fixes, it's a mismatch between what
the model learned and what the webcam sees.

`camera_inference.py` is included and works as a live pipeline (camera →
preprocess → model → on-screen label), and it becomes genuinely useful if
you have an actual thermal/IR camera — many (including some Dali-tech
models) expose a standard USB video stream that OpenCV can read just like
a webcam, so pointing `CAMERA_INDEX` at that device is often all it takes.
Without thermal hardware, treat it as a demo of the plumbing rather than a
working detector — `predict.py` on saved thermal images is the accurate
path today.

## 7. Design choices, and why (grounded in the IRT + CNN literature)

This dataset (Najafi, Baleghi & Mirimani, *Thermal Image of Equipment
(Induction Motor)*, Mendeley Data) has been used directly in at least one
published study, and the broader approach here follows patterns common
across that literature:

- **Transfer learning over training from scratch.** With only ~30–40
  images per class, a CNN trained from scratch would overfit badly.
  Fine-tuning an ImageNet-pretrained backbone (MobileNetV3-Small by
  default; ResNet18 and SqueezeNet1.1 are also wired in via
  `CONFIG["MODEL_NAME"]`) is the standard fix, and it's the same idea
  behind Calderon-Uribe et al. (2024, *Machines*), who use a CNN trained
  on this same dataset purely as a feature extractor, with a separate
  classifier on top, rather than training a large network end-to-end from
  zero on a small set.
- **Stratified k-fold cross-validation instead of one split.** Small,
  imbalanced datasets need this to get an honest error estimate — a
  single lucky/unlucky split of ~35 images per class isn't reliable.
- **Class-weighted loss.** Your class counts range roughly 25–42 images,
  so the loss up-weights rarer classes to stop the model from just
  learning to predict the majority classes.
- **No hue/saturation augmentation.** Ordinary photo augmentation
  pipelines jitter hue, but in a thermal image, colour *is* the
  temperature reading — shifting it would teach the model on physically
  wrong data. Only geometric augmentation (flip/rotate/crop) and mild
  brightness/contrast jitter are applied.
- **Optional CLAHE preprocessing** (`CONFIG["ENABLE_CLAHE"]`) — local
  contrast enhancement before the CNN sees the image, in the spirit of
  the CLAHE-based preprocessing used in some of the more recent IRT+CNN
  papers. It's off by default; toggle it on and compare the CV numbers,
  since whether it helps is somewhat dataset-dependent.
- **Explainability / edge deployment** — Piechocki et al. (2023), *"Unraveling Induction Motor State through Thermal Imaging and Edge Processing: A Step towards Explainable Fault Diagnosis"* (their own motor thermal dataset, not this one), verified their CNN's predictions with model-interpretability methods and deployed the best-performing architecture on resource-constrained hardware. That's a natural next step here too (e.g. Grad-CAM over the trained model, or exporting to ONNX/TFLite for a Raspberry Pi) — not included in this first version since it's worth adding once the base classifier's numbers look good, but ask if you'd like it added.

A general pattern across this literature (e.g. Calderon-Uribe et al. 2024;
*"A CNN-Based Methodology for Identifying Mechanical Faults in Induction
Motors Using Thermography"*, *Machines* 2023, DOI 10.3390/machines11070752)
is that CNN-based methods on IRT images commonly report accuracy in the 90–95%+
range on their own datasets — useful as a rough benchmark, though not a
guarantee, since dataset size, class count, and image quality all differ
from paper to paper. I wasn't able to independently verify every title on
your reading list (Khanjani & Ezoji 2021; Javed et al. 2022; Li et al.
2024; Siraj et al. 2024; Xu et al. 2024) in this session — worth pulling
those directly for your citations — but the design choices above reflect
the consistent pattern across the ones I could confirm.

## 8. If accuracy is disappointing

With this little data per class, a few things help, in rough order of
effort:
1. Merge very similar/rare classes if your report allows it (e.g. if
   10%-stator faults are hard to visually distinguish from healthy).
2. Turn `ENABLE_CLAHE` on/off and compare.
3. Try `MODEL_NAME = "resnet18"` or `"squeezenet1_1"`.
4. Collect more images if at all possible — even 10–15 more per class
   meaningfully helps at this scale.
5. Look at `confusion_matrix.png` to see which classes are actually being
   confused — thermally similar fault levels (e.g. 10% vs 30% short
   circuit in the same phase) are the likely culprits.
