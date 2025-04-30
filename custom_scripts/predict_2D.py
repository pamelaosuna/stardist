import os
import sys
from glob import glob
import argparse

import tifffile as tiff
from skimage import io
from tqdm import tqdm
import numpy as np

from csbdeep.utils import normalize

from stardist.models import StarDist2D, Config2D
from stardist.plot import render_label
from stardist import fill_label_holes
from stardist import random_label_cmap


import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
np.random.seed(42)
lbl_cmap = random_label_cmap()

def load_data_wo_GT(img_dir):
    filepaths = sorted(glob(os.path.join(img_dir, '*.tif')))

    X = list(map(tiff.imread,filepaths))

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    axis_norm = (0,1)   # normalize channels independently

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]

    return X, filepaths

def load_model(model_path):
    # model = StarDist2D.from_pretrained(model_path)
    model = StarDist2D(
        config=None, 
        name='synapse_stardist_2D',
        basedir=os.path.dirname(model_path)
    )
    
    return model

def predict_2D(model, X):
    Y_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0]
              for x in tqdm(X)]
    
    return Y_pred

def plot_img_label(img, lbl, out_dir, out_fn,
                   img_title="image (XY slice)", 
                   lbl_title="label (XY slice)", 
                   z=None, **kwargs):
    if z is None:
        z = img.shape[0] // 2    
    fig, (ai,al) = plt.subplots(1,2, figsize=(12,5), gridspec_kw=dict(width_ratios=(1.25,1)))
    im = ai.imshow(img[z], cmap='gray', clim=(0,1))
    ai.set_title(img_title)    
    fig.colorbar(im, ax=ai)
    al.imshow(lbl[z], cmap=lbl_cmap)
    al.set_title(lbl_title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, out_fn))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Predict 2D images with a trained StarDist model')
    parser.add_argument('--img_dir', type=str, required=True,
                        help='Directory containing the input images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained StarDist model')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save the predicted masks')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, filepaths = load_data_wo_GT(args.img_dir)
    model = load_model(args.model_path)
    Y_pred = predict_2D(model, X)

    plot_img_label(X[0],Y_pred[0], lbl_title="label Pred (XY slice)")

    for y, fp in zip(Y_pred, filepaths):
        out_fn = os.path.basename(fp).replace('.tif', '_pred.png')
        plot_img_label(X[0],y, img_title="image (XY slice)", lbl_title="label Pred (XY slice)", out_dir=args.out_dir, out_fn=out_fn)


