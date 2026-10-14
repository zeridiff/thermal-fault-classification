"""Run this file in PyCharm. See START_HERE.md. No filters are applied."""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from train_cnn import CONFIG as ORIGINAL_CONFIG, build_model, train_model, get_probs
from noise_utils import add_noise, noise_seed

# EDIT THIS PATH if your dataset has moved. It contains one folder per class.
DATA_DIR = r'C:\Users\dukeb\Downloads\Motor motor rrrrrrrr'
OUTPUT_DIR = Path(__file__).resolve().parent / 'motor_noise_results'
CFG = dict(ORIGINAL_CONFIG, ENABLE_CLAHE=False, NUM_WORKERS=0)
SEED = 42
INNER_VALIDATION_FRACTION = 0.15
NOISE_PROBABILITY = 0.5
TRAIN_RANGES = {'gaussian': (0.01, 0.08), 'salt_pepper': (0.005, 0.05), 'speckle': (0.01, 0.08)}
TEST_LEVELS = {'gaussian': [0.02, 0.05, 0.10, 0.15, 0.20],
               'salt_pepper': [0.01, 0.02, 0.05, 0.10, 0.15],
               'speckle': [0.02, 0.05, 0.10, 0.15, 0.20]}
TEST_REPEATS = 3
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scan_data(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Dataset folder does not exist: {root}\nEdit DATA_DIR near the top of this file.')
    excluded = {'venv', 'env', '__pycache__', 'outputs', 'output', 'motor_noise_results', 'node_modules'}
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    rows, classes = [], []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith('.') or folder.name in excluded:
            continue
        files = sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in extensions)
        if not files:
            continue
        label = len(classes)
        classes.append(folder.name)
        for p in files:
            with Image.open(p) as im:
                im.convert('RGB').load()  # fail early with the filename if unreadable
            rows.append({'sample_id': len(rows), 'path': str(p.resolve()),
                         'relative_path': str(p.relative_to(root)), 'label': label, 'class_name': folder.name})
    if len(classes) < 2:
        raise ValueError('At least two non-empty class folders are required.')
    return pd.DataFrame(rows), classes


class MotorDataset(Dataset):
    def __init__(self, manifest, indices, cfg, training=False, augment_noise=False,
                 fold=0, family='clean', level=0.0, repeat=0):
        self.manifest, self.indices, self.cfg = manifest, list(indices), cfg
        self.training, self.augment_noise = training, augment_noise
        self.fold, self.family, self.level, self.repeat = fold, family, level, repeat
        self.rng = np.random.default_rng(SEED + 10000 + fold)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(cfg['IMG_SIZE'], scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15)
        ]) if training else transforms.Resize((cfg['IMG_SIZE'], cfg['IMG_SIZE']))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = self.indices[item]
        row = self.manifest.iloc[idx]
        with Image.open(row['path']) as im:
            x = np.asarray(self.transform(im.convert('RGB')), dtype=np.float32) / 255.0
        if self.training and self.augment_noise and self.rng.random() < NOISE_PROBABILITY:
            family = self.rng.choice(list(TRAIN_RANGES))
            level = self.rng.uniform(*TRAIN_RANGES[family])
            x = add_noise(x, family, level, self.rng)
        elif not self.training and self.family != 'clean':
            rng = np.random.default_rng(noise_seed(SEED, self.fold, self.family, self.level, self.repeat, idx))
            x = add_noise(x, self.family, self.level, rng)
        tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
        return NORMALIZE(tensor), int(row['label'])


def make_loader(ds, batch, shuffle=False, seed=SEED):
    # Zero workers makes the dedicated augmentation RNG predictable on Windows.
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def save_history(history, folder):
    pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, suffix in zip(axes, ['loss', 'acc']):
        for split in ['train', 'val']:
            ax.plot(range(1, len(history[f'{split}_{suffix}']) + 1), history[f'{split}_{suffix}'], label=split)
        ax.set(xlabel='Epoch', ylabel='Loss' if suffix == 'loss' else 'Accuracy')
        ax.legend()
    fig.tight_layout(); fig.savefig(folder / 'training_curves.png', dpi=180); plt.close(fig)


def summarize(records, classes, out):
    records.to_csv(out / 'all_test_predictions.csv', index=False)
    metrics = []
    keys = ['condition', 'noise', 'strength', 'repeat']
    for key, data in records.groupby(keys):
        metrics.append(dict(zip(keys, key), accuracy=accuracy_score(data.label, data.prediction),
                            macro_f1=f1_score(data.label, data.prediction, labels=range(len(classes)),
                                              average='macro', zero_division=0), n_images=len(data)))
    scores = pd.DataFrame(metrics)
    scores.to_csv(out / 'metrics_each_repeat.csv', index=False)
    summary = scores.groupby(keys[:3]).agg(accuracy_mean=('accuracy', 'mean'), accuracy_std=('accuracy', 'std'),
               macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'), repeats=('repeat', 'count')).reset_index()
    summary[['accuracy_std', 'macro_f1_std']] = summary[['accuracy_std', 'macro_f1_std']].fillna(0)
    summary.to_csv(out / 'noise_summary.csv', index=False)
    for metric in ['accuracy', 'macro_f1']:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, family in zip(axes, TEST_LEVELS):
            for condition in ['baseline', 'noise_augmented']:
                values = summary[(summary.condition == condition) & summary.noise.isin(['clean', family])].sort_values('strength')
                ax.errorbar(values.strength, values[f'{metric}_mean'], yerr=values[f'{metric}_std'], marker='o', capsize=3, label=condition)
            units = {'gaussian': 'Standard deviation', 'salt_pepper': 'Corrupted pixel probability', 'speckle': 'Noise variance'}
            ax.set(title=family.replace('_', ' ').title(), xlabel=units[family], ylabel=metric.replace('_', ' ').title(), ylim=(0, 1.02))
            ax.legend(fontsize=8); ax.grid(alpha=0.2)
        fig.tight_layout(); fig.savefig(out / f'{metric}_noise_sweep.png', dpi=200); plt.close(fig)
    for condition in ['baseline', 'noise_augmented']:
        clean = records[(records.condition == condition) & (records.noise == 'clean')].sort_values('sample_id')
        clean.to_csv(out / f'{condition}_clean_oof.csv', index=False)
        fig, ax = plt.subplots(figsize=(10, 9))
        ConfusionMatrixDisplay(confusion_matrix(clean.label, clean.prediction, labels=range(len(classes))), display_labels=classes).plot(ax=ax, xticks_rotation=90, colorbar=False, cmap='Greys')
        ax.set_title(f'{condition}: clean held-out predictions')
        fig.tight_layout(); fig.savefig(out / f'{condition}_confusion_matrix.png', dpi=180); plt.close(fig)
    print('\nClean held-out scores:\n', summary[summary.noise == 'clean'].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=DATA_DIR)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='One fold, one epoch, no pretrained download. NOT report results.')
    args = parser.parse_args()
    cfg = dict(CFG)
    manifest, classes = scan_data(args.data_dir)
    counts = manifest.groupby('class_name').size()
    print(f'\n{len(manifest)} images, {len(classes)} classes\n{counts.to_string()}')
    if counts.min() < cfg['N_FOLDS']:
        raise ValueError(f"Every class needs at least {cfg['N_FOLDS']} images for the selected folds.")
    labels = manifest.label.to_numpy()
    splits = []
    outer = StratifiedKFold(n_splits=cfg['N_FOLDS'], shuffle=True, random_state=SEED)
    for fold, (outer_train, test) in enumerate(outer.split(np.zeros(len(labels)), labels), start=1):
        # Reserve at least one validation image per class and two fitting images per class.
        nval = max(len(classes), int(np.ceil(INNER_VALIDATION_FRACTION * len(outer_train))))
        fit, val = train_test_split(outer_train, test_size=nval, stratify=labels[outer_train], random_state=SEED + fold)
        if min(np.bincount(labels[fit], minlength=len(classes))) < 2:
            raise ValueError('Too few fitting images per class after splitting. Add data or reduce N_FOLDS.')
        assert not (set(fit) & set(val) or set(fit) & set(test) or set(val) & set(test))
        splits.append({'fold': fold, 'fit': fit.tolist(), 'validation': val.tolist(), 'test': test.tolist()})
    if args.check_data:
        print('\nData and all fold splits checked. No training was performed.'); return
    if args.smoke_test:
        cfg.update(EPOCHS=1, PRETRAINED=False)
        splits = splits[:1]
    out = OUTPUT_DIR / (('SMOKE_NOT_REPORT_' if args.smoke_test else 'RUN_') + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True)
    manifest.to_csv(out / 'dataset_manifest.csv', index=False)
    (out / 'splits.json').write_text(json.dumps(splits, indent=2))
    settings = {'training_config': cfg, 'data_dir': str(Path(args.data_dir).resolve()), 'seed': SEED,
                'classes': classes, 'smoke_test': args.smoke_test, 'noise_probability': NOISE_PROBABILITY,
                'train_ranges': TRAIN_RANGES, 'test_levels': TEST_LEVELS, 'test_repeats': TEST_REPEATS,
                'inner_validation_fraction': INNER_VALIDATION_FRACTION, 'filtering': False,
                'torch_version': torch.__version__}
    (out / 'run_config.json').write_text(json.dumps(settings, indent=2, default=str))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nDevice: {device}\nResults: {out}\nFiltering: OFF', flush=True)
    all_records = []
    for split in splits:
        fold, fit, val, test = (split[k] for k in ['fold', 'fit', 'validation', 'test'])
        for condition in ['baseline', 'noise_augmented']:
            seed_all(SEED + fold)  # identical initial model and shared transforms for each pair
            folder = out / f'fold_{fold}' / condition
            folder.mkdir(parents=True)
            print(f'\nFold {fold}: {condition}', flush=True)
            train_ds = MotorDataset(manifest, fit, cfg, True, condition == 'noise_augmented', fold)
            val_ds = MotorDataset(manifest, val, cfg, fold=fold)
            model = build_model(cfg['MODEL_NAME'], len(classes), cfg['PRETRAINED']).to(device)
            class_counts = np.bincount(labels[fit], minlength=len(classes))
            weights = 1.0 / class_counts
            weights /= weights.mean()
            model, history, _, _, _ = train_model(model, make_loader(train_ds, cfg['BATCH_SIZE'], True, SEED + fold),
                make_loader(val_ds, cfg['BATCH_SIZE']), torch.tensor(weights, dtype=torch.float32), device, cfg,
                cfg['EPOCHS'], verbose_prefix=f'[{fold}/{condition}] ')
            save_history(history, folder)
            torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'classes': classes, 'config': cfg, 'fold': fold, 'condition': condition,
                        'best_epoch': int(np.argmax(history['val_macro_f1'])) + 1}, folder / 'best_model.pth')
            scenarios = [('clean', 0.0, 0)]
            for family, levels in TEST_LEVELS.items():
                for level in (levels[:1] if args.smoke_test else levels):
                    scenarios.extend((family, level, repeat) for repeat in range(1 if args.smoke_test else TEST_REPEATS))
            fold_records = []
            for family, level, repeat in scenarios:
                ds = MotorDataset(manifest, test, cfg, fold=fold, family=family, level=level, repeat=repeat)
                probs, truth = get_probs(model, make_loader(ds, cfg['BATCH_SIZE']), device)
                assert np.array_equal(truth, labels[test]), 'Prediction/label order mismatch.'
                rows = manifest.iloc[test].copy()
                rows['prediction'] = probs.argmax(axis=1)
                rows['condition'], rows['fold'] = condition, fold
                rows['noise'], rows['strength'], rows['repeat'] = family, level, repeat
                for i in range(len(classes)):
                    rows[f'probability_class_{i}'] = probs[:, i]
                fold_records.append(rows)
                print(f'  Test {family} {level:g}, repeat {repeat+1}: accuracy={accuracy_score(truth, probs.argmax(1)):.3f}', flush=True)
            completed = pd.concat(fold_records, ignore_index=True)
            completed.to_csv(folder / 'test_predictions.csv', index=False)
            all_records.append(completed)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summarize(pd.concat(all_records, ignore_index=True), classes, out)
    (out / 'COMPLETED.txt').write_text('SMOKE TEST ONLY - NOT REPORT RESULTS' if args.smoke_test else 'All requested folds and corruption repeats completed.')
    print(f'\nFinished. Results saved in: {out}', flush=True)


if __name__ == '__main__':
    main()
