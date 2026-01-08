import os
from glob import glob
import tifffile as tiff

import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np

def random_label_cmap(n=2**16, h = (0,1), l = (.4,1), s =(.2,.8)):
    import matplotlib
    import colorsys
    # cols = np.random.rand(n,3)
    # cols = np.random.uniform(0.1,1.0,(n,3))
    h,l,s = np.random.uniform(*h,n), np.random.uniform(*l,n), np.random.uniform(*s,n)
    cols = np.stack([colorsys.hls_to_rgb(_h,_l,_s) for _h,_l,_s in zip(h,l,s)],axis=0)
    cols[0] = 0
    return matplotlib.colors.ListedColormap(cols)

def labels_to_rgb(labels):
    """
    Convert a 3D label array into an RGB image where each label is assigned a unique color.
    """
    # Apply the colormap to the labels
    rgb_image = random_label_cmap()(labels)
    # rgb_image = colormap(norm(labels))

    # Convert to uint8 (0-255 range) for saving as an image
    rgb_image = (rgb_image[..., :3] * 255).astype(np.uint8)  # Drop alpha channel
    return rgb_image

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
    img_dir = 'data_altug_3D/val_crop/images'
    gt_labels_dir = 'data_altug_3D/val_crop/masks'
    pred_labels_dir = 'data_altug_3D/predictions/synapse_stardist_3D_grid=1-1-1_aug=False_bs=1_old'

    img_paths = sorted(glob(os.path.join(img_dir, '*.tif')))
    # img_path = 'data_altug_3D/val_crop/images/1713_FOV_006_t_01_x1_508_y1_390_z1_sub11_zi0.tif'
    for f in img_paths:    
        fn = os.path.basename(f)
        img = tiff.imread(f)
        gt_labels = tiff.imread(os.path.join(gt_labels_dir, fn)).astype(np.uint16)
        pred_labels = tiff.imread(os.path.join(pred_labels_dir, fn)).astype(np.uint16)

        rgb_labels_gt = labels_to_rgb(gt_labels.astype(np.uint16))

        img_plus_gt = overlay_labels_on_image(img, rgb_labels_gt)
        tiff.imwrite(fn.replace('.tif', '_gt.tif'), img_plus_gt)

        rgb_labels_pred = labels_to_rgb(pred_labels.astype(np.uint16))
        img_plus_pred = overlay_labels_on_image(img, rgb_labels_pred)
        tiff.imwrite(fn.replace('.tif', '_pred.tif'), img_plus_pred)




