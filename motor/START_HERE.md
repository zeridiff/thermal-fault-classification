# Motor noise experiment — PyCharm

Run `train_motor_noise.py`. Keep `train_cnn.py` and `noise_utils.py` in the same folder: the new script reuses your original model and training functions. Your original uploaded training script is included unchanged. Do not run that original script for this experiment.

## Start in PyCharm

1. Extract this ZIP into a new project folder and open that folder in PyCharm.
2. Select a Python interpreter with your existing PyTorch environment, or create a virtual environment. Python 3.11 is a suitable starting point.
3. In the PyCharm terminal, install the requirements using the selected interpreter:

   `python -m pip install -r requirements.txt`

4. Open `train_motor_noise.py` and check `DATA_DIR` near the top. It currently uses the path from your original code:

   `DATA_DIR = r"C:\Users\dukeb\Downloads\Motor motor rrrrrrrr"`

   This folder must contain the class folders, such as `Noload`, `Rotor-0`, `Fan`, `A10`, etc. Each class folder contains its images. Keep outputs outside this dataset folder.

5. Check the dataset and splits first:

   `python train_motor_noise.py --check-data`

6. Run the short installation check:

   `python train_motor_noise.py --smoke-test`

   This trains both conditions for one epoch on only the first fold, with no pretrained weights download. It tests each noise family at one strength. These are **not report results**. Its output folder begins `SMOKE_NOT_REPORT`.

7. Run the full experiment:

   `python train_motor_noise.py`

   You can also right-click `train_motor_noise.py` and select Run, with no script parameters. Remove `--smoke-test` from any saved run configuration before the full run.

The first full run may download the pretrained MobileNetV3-Small weights. Keep internet access available for that download. CUDA is used if available in your installed PyTorch; otherwise the script uses the CPU. The full experiment trains ten models (two conditions × five folds), so it will take longer than the original five-fold run. Leave PyCharm running until the console prints Finished.

## What is compared

- **baseline:** your existing crop, flip, rotation and brightness/contrast augmentation, without added noise.
- **noise_augmented:** the same augmentation, with a 50% chance of adding one randomly chosen noise family to each training image on each pass.

Validation images stay clean. Both models are evaluated on clean test images and identical randomly corrupted copies of those test images. No filtering or CLAHE is applied. This experiment does not replace the Raspberry Pi deployment model.

Noise is added after resizing/augmentation, to RGB image values between 0 and 1, before ImageNet normalisation. Gaussian and speckle random values are independent across RGB channels. Salt-and-pepper corruption sets all three channels of a selected pixel to black or white.

| Noise | Meaning of strength | Training strength range | Test strengths, in addition to clean |
| --- | --- | --- | --- |
| Gaussian | Additive standard deviation | 0.01–0.08 | 0.02, 0.05, 0.10, 0.15, 0.20 |
| Salt-and-pepper | Total probability of corrupting a pixel; half black, half white | 0.005–0.05 | 0.01, 0.02, 0.05, 0.10, 0.15 |
| Speckle | Variance of multiplicative zero-mean Gaussian noise | 0.01–0.08 | 0.02, 0.05, 0.10, 0.15, 0.20 |

Values are clipped to [0, 1]. The three strength scales have different meanings; equal numbers across noise families do not imply equal severity. Training strengths are sampled uniformly within the ranges shown. The higher test strengths deliberately go beyond the training range. These are starting experimental choices, not settings established by your articles or proven to be optimal.

Each nonzero test strength uses three independently seeded corruption repeats. These are repeated corruptions of the same test images, not three independent datasets or three training runs. The reported standard deviation describes variation from those corruptions only. It is not a confidence interval or training variability estimate. Clean data have a single evaluation and a displayed standard deviation of zero by convention.

These are synthetic corruptions of saved thermal display images. They do not establish how actual sensor noise affects raw radiometric temperature measurements.

## How test leakage and ordering are handled

Five stratified outer folds are fixed and shared between conditions. Inside each outer training fold, a separate stratified validation subset is used for early stopping and scheduling. The outer test fold is not used to choose training epochs. Fitting-set class counts alone determine loss weights. Each condition starts from the same seeded model initialisation and data order for its fold.

Your original script used the held-out fold for early stopping and also reported performance on it. It also combined fold-order labels with dataset-order probabilities in one saved output. The new script separates validation from testing and records labels and probabilities alongside each sample identifier. Because this protocol is different, its clean results may differ from your earlier results: compare the two conditions within this new experiment.

This remains image-level cross-validation. If several images are consecutive frames from the same recording or repeated views of the same acquisition, those images may be related across folds. A grouped split by recording/session would then be needed to assess generalisation to unseen sessions. This package cannot infer recording identifiers from the provided scripts. Do not interpret these results as proven performance on unseen motors or recording sessions.

## Results to send back

Every run creates a new timestamped folder inside `motor_noise_results`. Existing results are not overwritten. A completed full run contains:

- `noise_summary.csv`: mean accuracy and macro-F1 for each condition/noise/strength, plus corruption-repeat standard deviations. Scores are proportions (0.95 means 95%).
- `accuracy_noise_sweep.png` and `macro_f1_noise_sweep.png`: comparison curves, with standard-deviation error bars.
- `baseline_confusion_matrix.png` and `noise_augmented_confusion_matrix.png`: clean held-out confusion matrices.
- `baseline_clean_oof.csv` and `noise_augmented_clean_oof.csv`: clean predictions aligned to original sample IDs.
- `all_test_predictions.csv`: every test prediction, with class probabilities. Probability-column indices map to the ordered `classes` list in `run_config.json`.
- `metrics_each_repeat.csv`: metrics pooled across the held-out folds for each corruption repeat.
- `dataset_manifest.csv`, `splits.json` and `run_config.json`: the images, fold membership and settings used.
- `fold_*/baseline` and `fold_*/noise_augmented`: best model checkpoints, training curves/history and fold test predictions. These checkpoints and fixed test-noise seeds can support the later filtering comparison.
- `COMPLETED.txt`: confirms completion; smoke runs are clearly marked.

Send the summary CSV, both noise-sweep plots and the two confusion matrices first. We can interpret the actual results and then add filtering. An interrupted run retains completed fold outputs but does not automatically resume; restarting creates a new folder.

## Troubleshooting

- **Dataset folder does not exist:** correct `DATA_DIR`; preserve the `r` before the Windows path string.
- **No module named ...:** install requirements with the same interpreter PyCharm uses to run the file.
- **Too few images / splitting error:** inspect the class counts printed by `--check-data`. Do not delete classes simply to make a run pass.
- **Pretrained download error:** restore internet access and rerun the full experiment. The smoke test does not need pretrained downloads.
- **Out of memory:** reduce `CFG['BATCH_SIZE']` near the top by adding `CFG['BATCH_SIZE'] = 8` after its definition. Keep it at least 2 for training and record any settings change.

## Verification supplied with this package

`python test_noise_utils.py` checks noise strength, clipping, deterministic seeds and paired corruptions. The package was also checked using a small synthetic dataset through both training conditions and the result-export path. That installation test does not measure performance on your motor dataset. No motor accuracy result is supplied or invented here.
