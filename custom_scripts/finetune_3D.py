import os
import sys
import argparse
from glob import glob

from tqdm import tqdm
import tifffile as tiff
import numpy as np

from csbdeep.utils import normalize

from stardist import fill_label_holes
from stardist.models import StarDist3D, Config3D
from stardist import Rays_GoldenSpiral

from stardist import gputools_available

from finetune_2D import (
    split_train_val, 
    random_intensity_change,
)

def random_fliprot(img, mask, axis=None): 
    if axis is None:
        axis = tuple(range(mask.ndim))
    axis = tuple(axis)
            
    assert img.ndim>=mask.ndim
    perm = tuple(np.random.permutation(axis))
    transpose_axis = np.arange(mask.ndim)
    for a, p in zip(axis, perm):
        transpose_axis[a] = p
    transpose_axis = tuple(transpose_axis)
    img = img.transpose(transpose_axis + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(transpose_axis) 
    for ax in axis: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

def load_data(img_dir, mask_dir):
    X = sorted(glob(os.path.join(img_dir, '*.tif')))
    Y = sorted(glob(os.path.join(mask_dir, '*.tif')))

    assert all(x.split('/')[-1].split('.')[0]==y.split('/')[-1].split('.')[0] for x,y in zip(X,Y)), \
        "Image and mask filenames do not match. Please check the directories."
    
    X = list(map(tiff.imread,X))
    Y = list(map(tiff.imread,Y))

    n_channel = 1 if X[0].ndim == 3 else X[0].shape[-1]
    axis_norm = (0,1,2) # normalize channels independently
    if n_channel > 1:
        print("Normalizing image channels %s." % ('jointly' if axis_norm is None or 3 in axis_norm else 'independently'))
        sys.stdout.flush()

    X = [normalize(x,1,99.8,axis=axis_norm) for x in tqdm(X)]
    Y = [fill_label_holes(y) for y in tqdm(Y)]

    return X, Y

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    # Note that we only use fliprots along axis=(1,2), i.e. the yx axis 
    # as 3D microscopy acquisitions are usually not axially symmetric
    x, y = random_fliprot(x, y, axis=(1,2))
    x = random_intensity_change(x)
    return x, y

def resume_training(X_trn, Y_trn, X_val, Y_val, model_name, nepochs, batch_size, out_dir):
    # 96 is a good default choice (see 1_data.ipynb)
    n_rays = 96
    anisotropy = (5.0, 1.0, 1.0)

    # Use OpenCL-based computations for data generator during training (requires 'gputools')
    print('Using GPU for data generator:', gputools_available())
    # use_gpu = False and gputools_available()

    # Predict on subsampled grid for increased efficiency and larger field of view
    # grid = tuple(1 if a > 1.5 else 2 for a in anisotropy)
    grid = (1, 1, 1)

    # Use rays on a Fibonacci lattice adjusted for measured anisotropy of the training data
    rays = Rays_GoldenSpiral(n_rays, anisotropy=anisotropy)

    conf = Config3D (
        rays             = rays,
        grid             = grid,
        anisotropy       = anisotropy,
        use_gpu          = False,
        n_channel_in     = 1,
        # adjust for your data below (make patch size as large as possible)
        train_patch_size = (8,128,128),
        train_batch_size = batch_size, # TODO: try increasing this
    )
    print(conf)
    model = StarDist3D(
        conf,
        name=model_name,
        basedir=out_dir)
    # model = StarDist3D.from_pretrained('3D_demo')

    model.train(
        X_trn, Y_trn,
        validation_data=(X_val, Y_val),
        augmenter=augmenter, # try changing this # None
        epochs=nepochs,
        steps_per_epoch=10
        )
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Fine-tune a pretrained StarDist model')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str,
                        default='checkpoints/3D/')
    parser.add_argument('--model_name', type=str,
                        default='synapse_stardist_3D')
    parser.add_argument('--nepochs', type=int,
                        default=100,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help='batch size for training')

    args = parser.parse_args()
    print(args)

    os.makedirs(args.out_dir, exist_ok=True)

    X_trn, Y_trn = load_data(
        os.path.join(args.train_dir, 'images'),
        os.path.join(args.train_dir, 'masks')
    )

    X_val, Y_val = load_data(
        os.path.join(args.val_dir, 'images'),
        os.path.join(args.val_dir, 'masks')
    )

    print(f'number of images: {len(X_trn) + len(X_val)}')
    print(f'- training:       {len(X_trn)}')
    print(f'- validation:     {len(X_val)}')

    # X_trn, Y_trn, X_val, Y_val = split_train_val(X, Y, perc_val=0.15)
    model = resume_training(
        X_trn, Y_trn, 
        X_val, Y_val, 
        args.model_name,
        args.nepochs,
        args.batch_size,
        args.out_dir
    )