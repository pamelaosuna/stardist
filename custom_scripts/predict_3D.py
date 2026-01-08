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
from matplotlib import colors

from csbdeep.utils import normalize

from stardist.models import StarDist3D, Config3D
from stardist import random_label_cmap
from finetune_3D import (
    load_data
)

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

def labels_to_rgb(labels, cmap=None):
    """
    Convert a 3D label array into an 4D RGB image (z, y, x, 3) where each label is assigned a unique color.
    """
    if cmap is None:
        cmap = random_label_cmap()
    max_label = labels.max()
    norm = colors.Normalize(vmin=0, vmax=max_label)
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for z in range(labels.shape[0]):
        rgb_slice = cmap(norm(labels[z]))
        rgb[z] = (rgb_slice[..., :3] * 255).astype(np.uint8)
    return rgb

def overlay_labels_on_image(img, rgb_labels):
    """
    Overlay the RGB labels on the original image.
    """
    # convert the image to RGB if it's grayscale
    img = np.stack([img] * 3, axis=-1)

    # Ensure the image and labels have the same shape
    assert img.shape[:2] == rgb_labels.shape[:2], "Image and labels must have the same spatial dimensions"
    # Overlay the labels on the image using alpha blending
    overlay = np.clip(img * 0.5 + rgb_labels * 0.5, 0, 255).astype(np.uint8)

    return overlay
        
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

    np.random.seed(6)
    lbl_cmap = random_label_cmap()

    _, Y_GT = load_data(
        args.img_dir,
        os.path.join(args.img_dir, '..', 'masks'),
    )

    X, filepaths = load_data_wo_GT(args.img_dir)
    model = StarDist3D(
        None, 
        name=args.model_name, 
        basedir='checkpoints/3D/'
    )
    out_dir_viz = os.path.join(out_dir, 'viz')
    os.makedirs(out_dir_viz, exist_ok=True)

    for i, img in enumerate(X):
        fp = filepaths[i]
        plot_img_label(model, img, Y_GT[i], out_dir_viz, os.path.basename(fp).replace('.tif', '_GT.png'))
        rgb_labels_gt = labels_to_rgb(Y_GT[i].astype(np.uint16))
        tiff.imwrite(os.path.join(out_dir_viz, os.path.basename(fp).replace('.tif', '_GT_colored.tif')), rgb_labels_gt, photometric='rgb')

        labels, details = model.predict_instances(img)
        rgb_labels = labels_to_rgb(labels.astype(np.uint16))
        tiff.imwrite(os.path.join(out_dir_viz, os.path.basename(fp).replace('.tif', '_pred_colored.tif')), rgb_labels, photometric='rgb')

        print(f'Predicted {len(np.unique(labels))-1} objects in {os.path.basename(fp)}')
        tiff.imwrite(os.path.join(out_dir, os.path.basename(fp)), labels.astype(np.uint16))
        out_fn = os.path.basename(fp).replace('.tif', '_pred.png')
        plot_img_label(model, img, labels, out_dir_viz, out_fn)
