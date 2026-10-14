"""Motor filter comparison. Place beside train_cnn.py and noise_utils.py.

Run in the PyCharm terminal:
  python evaluate_motor_filters.py --check-files
  python evaluate_motor_filters.py --smoke-test
  python evaluate_motor_filters.py

If multiple full noise runs exist, add --run-dir "motor_noise_results/RUN_...".
No training is performed. Existing results and weights are not modified.
Filters are fixed exploratory choices, not parameters selected on test scores.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path, PureWindowsPath
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from train_cnn import build_model, get_probs
from noise_utils import add_noise, noise_seed

FILTERS = ['none', 'median_3', 'gaussian_5', 'bilateral_5']
FILTER_SETTINGS = {
    'none': 'Unfiltered float32 RGB image',
    'median_3': '3 x 3 median, each channel separately, replicated border',
    'gaussian_5': '5 x 5 Gaussian, sigmaX=sigmaY=1.0, reflect101 border',
    'bilateral_5': 'diameter=5, sigmaColor=0.10 on [0,1] RGB, sigmaSpace=2.0, reflect101 border'
}
NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def apply_filter(x, name):
    x = np.ascontiguousarray(x, dtype=np.float32)
    if name == 'none':
        return x.copy()
    if name == 'median_3':
        y = cv2.medianBlur(x, 3)
    elif name == 'gaussian_5':
        y = cv2.GaussianBlur(x, (5, 5), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REFLECT_101)
    elif name == 'bilateral_5':
        y = cv2.bilateralFilter(x, 5, 0.10, 2.0, borderType=cv2.BORDER_REFLECT_101)
    else:
        raise ValueError(name)
    return np.clip(y, 0, 1).astype(np.float32)


class FilterDataset(Dataset):
    def __init__(self, manifest, indices, size, seed, fold, family, strength, repeat, filter_name):
        self.manifest, self.indices = manifest, indices
        self.resize = transforms.Resize((size, size))
        self.seed, self.fold, self.family = seed, fold, family
        self.strength, self.repeat, self.filter_name = strength, repeat, filter_name

    def __len__(self):
        return len(self.indices)

    def raw(self, pos):
        idx = self.indices[pos]
        with Image.open(self.manifest.loc[idx, 'path']) as im:
            clean = np.asarray(self.resize(im.convert('RGB')), dtype=np.float32) / 255.0
        rng = np.random.default_rng(noise_seed(self.seed, self.fold, self.family, self.strength, self.repeat, idx))
        noisy = add_noise(clean, self.family, self.strength, rng)
        return clean, noisy

    def __getitem__(self, pos):
        _, noisy = self.raw(pos)
        filtered = apply_filter(noisy, self.filter_name)
        x = torch.from_numpy(np.ascontiguousarray(filtered.transpose(2, 0, 1)))
        return NORMALIZE(x), int(self.manifest.loc[self.indices[pos], 'label'])


def load_run(args):
    if args.run_dir:
        run = Path(args.run_dir).resolve()
    else:
        candidates = sorted(p for p in (Path(__file__).resolve().parent / 'motor_noise_results').glob('RUN_*')
                            if (p / 'COMPLETED.txt').exists())
        if len(candidates) != 1:
            raise ValueError('Specify your completed run with --run-dir "motor_noise_results/RUN_...".\nFound: '
                             + ', '.join(str(p) for p in candidates))
        run = candidates[0]
    for name in ['run_config.json', 'dataset_manifest.csv', 'splits.json', 'all_test_predictions.csv', 'COMPLETED.txt']:
        if not (run / name).is_file():
            raise ValueError(f'Missing file: {run / name}')
    config = json.loads((run / 'run_config.json').read_text())
    if config.get('smoke_test') and not args.smoke_test:
        raise ValueError('Select the full RUN folder, not a SMOKE_NOT_REPORT folder.')
    manifest = pd.read_csv(run / 'dataset_manifest.csv').set_index('sample_id', drop=False)
    splits = json.loads((run / 'splits.json').read_text())
    previous = pd.read_csv(run / 'all_test_predictions.csv')
    if args.data_dir:
        root = Path(args.data_dir).resolve()
        manifest['path'] = [str(root.joinpath(*PureWindowsPath(p).parts)) for p in manifest.relative_path]
    for p in manifest.path:
        if not Path(p).is_file():
            raise ValueError(f'Image missing: {p}\nIf the dataset moved, add --data-dir "path to dataset".')
    seen = []
    for split in splits:
        a,b,t = [set(split[k]) for k in ['fit','validation','test']]
        if a&b or a&t or b&t or a|b|t != set(manifest.index):
            raise ValueError('Invalid or overlapping saved split.')
        seen.extend(t)
        for condition in ['baseline', 'noise_augmented']:
            checkpoint = run / f"fold_{split['fold']}" / condition / 'best_model.pth'
            if not checkpoint.is_file():
                raise ValueError(f'Missing model: {checkpoint}')
    if not config.get('smoke_test') and sorted(seen) != sorted(manifest.index):
        raise ValueError('Test folds do not cover each image exactly once.')
    print(f'Using: {run}\nImages: {len(manifest)} | Classes: {len(config["classes"])}\nSaved models found. No retraining.', flush=True)
    return run,config,manifest,splits,previous


def summarize(all_rows, out):
    keys = ['condition', 'noise', 'strength', 'repeat', 'filter']
    metrics = []
    for key, d in all_rows.groupby(keys):
        metrics.append(dict(zip(keys, key), accuracy=accuracy_score(d.label, d.prediction),
                            macro_f1=f1_score(d.label, d.prediction, average='macro', zero_division=0), n_images=len(d)))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(out / 'filter_metrics_each_repeat.csv', index=False)
    summary = metrics.groupby(['condition','noise','strength','filter']).agg(
        accuracy_mean=('accuracy','mean'), accuracy_std=('accuracy','std'),
        macro_f1_mean=('macro_f1','mean'), macro_f1_std=('macro_f1','std'), repeats=('repeat','count')).reset_index()
    summary[['accuracy_std','macro_f1_std']] = summary[['accuracy_std','macro_f1_std']].fillna(0)
    unfiltered = summary[summary['filter']=='none'][['condition','noise','strength','accuracy_mean','macro_f1_mean']]
    unfiltered = unfiltered.rename(columns={'accuracy_mean':'unfiltered_accuracy','macro_f1_mean':'unfiltered_macro_f1'})
    summary = summary.merge(unfiltered, on=['condition','noise','strength'], validate='many_to_one')
    summary['accuracy_change_percentage_points'] = 100*(summary.accuracy_mean-summary.unfiltered_accuracy)
    summary['macro_f1_change'] = summary.macro_f1_mean-summary.unfiltered_macro_f1
    summary.to_csv(out / 'filter_summary.csv', index=False)
    summary[summary.noise=='clean'].to_csv(out / 'clean_filter_comparison.csv', index=False)
    for metric in ['accuracy','macro_f1']:
        for condition in ['baseline','noise_augmented']:
            fig, axes = plt.subplots(1,3,figsize=(15,4))
            for ax,family in zip(axes,['gaussian','salt_pepper','speckle']):
                for filter_name in FILTERS:
                    d=summary[(summary.condition==condition)&(summary['filter']==filter_name)&summary.noise.isin(['clean',family])].sort_values('strength')
                    ax.errorbar(d.strength,d[f'{metric}_mean'],yerr=d[f'{metric}_std'],label=filter_name,marker='o',capsize=2)
                unit={'gaussian':'Noise standard deviation','salt_pepper':'Corrupted pixel probability','speckle':'Noise variance'}[family]
                ax.set(title=family.replace('_',' ').title(),xlabel=unit,ylabel=metric.replace('_',' ').title(),ylim=(0,1.02))
                ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.suptitle(condition.replace('_',' '));fig.tight_layout()
            fig.savefig(out/f'{condition}_{metric}_filters.png',dpi=200);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir')
    parser.add_argument('--data-dir',help='Only needed if original images moved.')
    parser.add_argument('--check-files',action='store_true')
    parser.add_argument('--smoke-test',action='store_true',help='First fold, first strength, one repeat. NOT report results.')
    args=parser.parse_args()
    run,config,manifest,splits,previous=load_run(args)
    if args.check_files:
        print('File check passed. No evaluation was performed.');return
    out=run / (('FILTER_SMOKE_NOT_REPORT_' if args.smoke_test else 'FILTER_RUN_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir()
    (out/'filter_settings.json').write_text(json.dumps({'source_run':str(run),'filters':FILTER_SETTINGS,
        'smoke_test':args.smoke_test,'cv2_version':cv2.__version__,'torch_version':str(torch.__version__),
        'noise_seed':config['seed'],'parameters_selected_using_test_scores':False},indent=2))
    (out/'READ_RESULTS.txt').write_text(
        'Fixed exploratory filter comparison; no retraining. Noise is added to resized RGB values in [0,1], '
        'then filtered, then ImageNet-normalised. Each filter receives the same corruption for each sample/repeat. '
        'Clean images are also tested through every filter. Positive change means improvement over no filter. '
        'Error bars are standard deviations across corruption repeats, not training runs or confidence intervals. '
        'Comparing filters on these test data is exploratory; choosing and claiming a final best filter needs '
        'validation-based selection and independent confirmation. Source images and models are unchanged.')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg=config['training_config'];classes=config['classes']
    scenarios=[('clean',0.0,0)]
    for family,levels in config['test_levels'].items():
        for strength in (levels[:1] if args.smoke_test else levels):
            scenarios.extend((family,strength,rep) for rep in range(1 if args.smoke_test else config['test_repeats']))
    all_rows=[]; checks=[]
    for split in (splits[:1] if args.smoke_test else splits):
        fold=split['fold'];indices=split['test']
        for condition in ['baseline','noise_augmented']:
            print(f'\nFold {fold}: {condition} | Device: {device}',flush=True)
            ckpt=torch.load(run/f'fold_{fold}'/condition/'best_model.pth',map_location='cpu',weights_only=True)
            if ckpt['classes']!=classes or ckpt['fold']!=fold or ckpt['condition']!=condition:
                raise ValueError('Saved checkpoint metadata does not match this run.')
            model=build_model(cfg['MODEL_NAME'],len(classes),pretrained=False)
            model.load_state_dict(ckpt['state_dict']);model.to(device);model.eval()
            fold_rows=[]
            for family,strength,rep in scenarios:
                for filter_name in FILTERS:
                    ds=FilterDataset(manifest,indices,cfg['IMG_SIZE'],config['seed'],fold,family,strength,rep,filter_name)
                    probs,truth=get_probs(model,DataLoader(ds,batch_size=cfg['BATCH_SIZE'],shuffle=False,num_workers=0),device)
                    if not np.array_equal(truth,manifest.loc[indices,'label']):
                        raise ValueError('Label order mismatch.')
                    predictions=probs.argmax(1)
                    if filter_name=='none':
                        old=previous[(previous.condition==condition)&(previous.fold==fold)&(previous.noise==family)&
                                     np.isclose(previous.strength,strength)&(previous['repeat']==rep)].set_index('sample_id').loc[indices]
                        old_probs=old[[f'probability_class_{i}' for i in range(len(classes))]].to_numpy()
                        difference=float(np.max(np.abs(probs-old_probs)))
                        matches=bool(np.array_equal(predictions,old.prediction))
                        checks.append(dict(fold=fold,condition=condition,noise=family,strength=strength,repeat=rep,
                                           max_probability_difference=difference,predictions_match=matches))
                        pd.DataFrame(checks).to_csv(out/'unfiltered_reproduction_check.csv',index=False)
                        if not matches or difference>1e-4:
                            raise ValueError('Unfiltered predictions differ from the original run. Stop and share '
                                             'unfiltered_reproduction_check.csv before interpreting filter results.')
                    rows=manifest.loc[indices,['sample_id','class_name','label']].copy()
                    rows['prediction']=predictions
                    rows['fold']=fold;rows['condition']=condition;rows['noise']=family
                    rows['strength']=strength;rows['repeat']=rep;rows['filter']=filter_name
                    for i in range(len(classes)): rows[f'probability_class_{i}']=probs[:,i]
                    fold_rows.append(rows)
                    print(f'  {family} {strength:g}, repeat {rep+1}, {filter_name}: accuracy={accuracy_score(truth,predictions):.4f}',flush=True)
                if fold==splits[0]['fold'] and condition=='baseline' and rep==0 and strength==config['test_levels'].get(family,[None])[0]:
                    clean,noisy=ds.raw(0)
                    fig,axes=plt.subplots(1,5,figsize=(15,3))
                    for ax,name,img in zip(axes,['Clean','Noisy','Median 3x3','Gaussian 5x5','Bilateral'],
                        [clean,noisy,apply_filter(noisy,'median_3'),apply_filter(noisy,'gaussian_5'),apply_filter(noisy,'bilateral_5')]):
                        ax.imshow(img);ax.set_title(name);ax.axis('off')
                    fig.suptitle(f'{family}, strength {strength:g}; sample {indices[0]} ({manifest.loc[indices[0],"class_name"]})')
                    fig.tight_layout();fig.savefig(out/f'example_{family}.png',dpi=180);plt.close(fig)
            completed=pd.concat(fold_rows,ignore_index=True)
            completed.to_csv(out/f'fold_{fold}_{condition}_predictions.csv',index=False)
            all_rows.append(completed)
            del model,ckpt
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    summarize(pd.concat(all_rows,ignore_index=True),out)
    (out/'COMPLETED.txt').write_text('SMOKE TEST ONLY - NOT REPORT RESULTS' if args.smoke_test else 'Filter evaluation complete. No retraining.')
    print(f'\nFinished. Filter results saved in: {out}',flush=True)


if __name__=='__main__':
    main()
