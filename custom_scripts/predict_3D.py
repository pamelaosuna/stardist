import argparse
import os
import sys
from glob import glob

import numpy as np
import tifffile as tiff
from tqdm import tqdm

import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt

from csbdeep.utils import normalize

from stardist.models import StarDist3D, Config3D
from stardist import random_label_cmap
from finetune_3D import (
    load_data
)

np.random.seed(6)
lbl_cmap = random_label_cmap()

def load_data_wo_GT(img_dir):
    filepaths = sorted(glob(os.path.join(img_dir, '*.tif')))

    X = list(map(tiff.imread,filepaths))

    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0,1,2)   # normalize channels independently

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]

    return X, filepaths

def load_model(model_name):
    model = StarDist3D(
        config=None,
        name=model_name,
        basedir='checkpoints/3D/'
    )

    return model

def plot_img_label(model, img, labels, out_dir, out_fn, show_dist=True):
    plt.figure(figsize=(13,8))
    z = img.shape[0] // 2
    y = img.shape[1] // 2
    img_show = img if img.ndim==3 else img[...,:3]    
    plt.subplot(221); plt.imshow(img_show[z],   cmap='gray', clim=(0,1)); plt.axis('off'); plt.title('XY slice')
    plt.subplot(222); plt.imshow(img_show[:,y], cmap='gray', clim=(0,1)); plt.axis('off'); plt.title('XZ slice')
    plt.subplot(223); plt.imshow(img_show[z],   cmap='gray', clim=(0,1)); plt.axis('off'); plt.title('XY slice')
    plt.imshow(labels[z], cmap=lbl_cmap, alpha=0.5)
    plt.subplot(224); plt.imshow(img_show[:,y], cmap='gray', clim=(0,1)); plt.axis('off'); plt.title('XZ slice')
    plt.imshow(labels[:,y], cmap=lbl_cmap, alpha=0.5)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(out_dir, out_fn))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Predict 2D images with a trained StarDist model')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing the input images')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the trained StarDist model')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save the predicted masks')
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)

    _, Y_tmp = load_data(
        args.img_dir,
        os.path.join(args.img_dir, '..', 'masks'),
    )

    X, filepaths = load_data_wo_GT(args.img_dir)
    model = StarDist3D(
        None, 
        name=args.model_name, 
        basedir='checkpoints/3D/'
    )

    for i, img in enumerate(X):
        fp = filepaths[i]
        plot_img_label(model, img, Y_tmp[i], out_dir, os.path.basename(fp).replace('.tif', '_GT.png'))

        labels, details = model.predict_instances(img)
        print(f'Predicted {len(np.unique(labels))-1} objects in {os.path.basename(fp)}')
        tiff.imwrite(os.path.join(out_dir, os.path.basename(fp)), labels.astype(np.uint16))
        out_fn = os.path.basename(fp).replace('.tif', '_pred.png')
        plot_img_label(model, img, labels, out_dir, out_fn)
        