import os
import sys
from glob import glob
import argparse

import tifffile as tiff
from skimage import io
from tqdm import tqdm
import numpy as np

from csbdeep.utils import Path, normalize

from stardist.models import StarDist2D, Config2D
from stardist.plot import render_label
from stardist import fill_label_holes

def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    x, y = random_fliprot(x, y)
    x = random_intensity_change(x)
    # add some gaussian noise
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

def load_data(img_dir, mask_dir):
    X = sorted(glob(os.path.join(img_dir, '*.tif')))
    Y = sorted(glob(os.path.join(mask_dir, '*.png')))

    assert all(x.split('/')[-1].split('.')[0]==y.split('/')[-1].split('.')[0] for x,y in zip(X,Y)), \
        "Image and mask filenames do not match. Please check the directories."

    X = list(map(tiff.imread,X))
    Y = list(map(io.imread,Y))

    print("Min and max of images and masks:")
    print(X[0].min(), X[0].max())
    print(Y[0].min(), Y[0].max())

    n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]
    axis_norm = (0,1)   # normalize channels independently

    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 2 in axis_norm else 'independently'))
    sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    print("Min and max of images and masks after normalization:")
    print(X[0].min(), X[0].max())
    print(Y[0].min(), Y[0].max())

    return X, Y

def split_train_val(X, Y, perc_val=0.15):
    assert len(X) > 1, "not enough training data"
    rng = np.random.RandomState(42)
    ind = rng.permutation(len(X))
    n_val = max(1, int(round(perc_val * len(ind))))
    ind_train, ind_val = ind[:-n_val], ind[-n_val:]
    X_val, Y_val = [X[i] for i in ind_val], [Y[i] for i in ind_val]
    X_trn, Y_trn = [X[i] for i in ind_train], [Y[i] for i in ind_train] 
    print('number of images: %3d' % len(X))
    print('- training:       %3d' % len(X_trn))
    print('- validation:     %3d' % len(X_val))

    return X_trn, Y_trn, X_val, Y_val

def resume_training(X_trn, Y_trn, X_val, Y_val, epochs=2, steps_per_epoch=10):
    # prints a list of available models
    StarDist2D.from_pretrained()

    # loads a pretrained model
    model = StarDist2D.from_pretrained('2D_versatile_fluo')

    model.train(X_trn, Y_trn, 
                validation_data=(X_val,Y_val), 
                augmenter=augmenter,
                epochs=epochs, 
                steps_per_epoch=steps_per_epoch)
    
    # # saves the model weights and configuration?
    # model.export_TF(fname=os.path.join(out_dir, 'weights_best.h5'))
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fine-tune a pretrained StarDist model')
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str,
                        default='checkpoints/')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, Y = load_data(args.img_dir, args.mask_dir)
    X_trn, Y_trn, X_val, Y_val = split_train_val(X, Y)
    model = resume_training(X_trn, Y_trn, X_val, Y_val)

    model.export_TF(fname=os.path.join(out_dir, 'weights_best.h5'))