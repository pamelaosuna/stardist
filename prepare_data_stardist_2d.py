import os
import glob
import sys
import argparse

import pandas as pd
from PIL import Image, ImageDraw

def create_img_masks(input, out_dir, batch_folder, height, width, fill_binary=True, save=True):
    drawn_imgs = []

    if batch_folder:
        fps = glob.glob(os.path.join(input, "*.csv"))
    else:
        fps = [input]

    for f in fps:
        df = pd.read_csv(f)
        n_polygons = df['index'].unique()
        axis_cols = sorted([c for c in df.columns if 'axis-' in c])
        x_axis, y_axis = axis_cols[-1], axis_cols[-2]

        img = Image.new('L', (height, width), 0)

        for i_poly in n_polygons:
            df_poly = df[df['index'] == i_poly]
            poly_vertices = list(zip(df_poly[x_axis], df_poly[y_axis])) # (x, y) format
            
            if fill_binary:
                fill = 255
            else:
                fill = int(i_poly+1)
            ImageDraw.Draw(img).polygon(poly_vertices, fill=fill) # outline=255,
        
        if save:
            img.save(os.path.join(out_dir, os.path.basename(f).replace('.csv', '.png')))

        drawn_imgs.append(img)

    return drawn_imgs 

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Create segmentation masks from annotations')
    parser.add_argument('--annot_dir', type=str, required=True,
                        help='Directory containing the csv annotation files (acquired with Napari)')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='Directory to save the segmentation masks')
    parser.add_argument('--height', type=int, default=288)
    parser.add_argument('--width', type=int, default=288)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files_annot = glob.glob(os.path.join(args.annot_dir, "*.csv"))

    imgs = create_img_masks(
        input=args.annot_dir,
        out_dir=args.out_dir,
        batch_folder=True,
        height=args.height,
        width=args.width,
        fill_binary=False,
        save=True)
