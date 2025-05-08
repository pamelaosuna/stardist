import os
from glob import glob
import argparse

import tifffile as tiff
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt

from stardist.matching import matching_dataset

def plot_stats(taus, stats, out_dir):
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'stats.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantitative evaluation of 3D StarDist model')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Directory containing ground truth masks')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Directory containing predicted masks')
    args = parser.parse_args()

    # load data
    gt_files = sorted(glob(os.path.join(args.gt_dir, '*.tif')))
    pred_files = sorted(glob(os.path.join(args.pred_dir, '*.tif')))

    assert [os.path.basename(gt_file) == os.path.basename(pred_file) for gt_file, pred_file in zip(gt_files, pred_files)], \
        "Ground truth and predicted files do not match"
    
    gt_imgs = [tiff.imread(gt_file) for gt_file in gt_files]
    pred_imgs = [tiff.imread(pred_file) for pred_file in pred_files]

    print(f'Number of objects in GT: {[len(np.unique(gt_img-1)) for gt_img in gt_imgs]}')
    print(f'Number of objects in pred: {[len(np.unique(pred_img-1)) for pred_img in pred_imgs]}')
    
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching_dataset(gt_imgs, pred_imgs, thresh=t, show_progress=False) for t in tqdm(taus)]

    plot_stats(taus, stats, args.pred_dir)





    
