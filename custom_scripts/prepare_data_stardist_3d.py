import os
import glob

import tifffile as tiff

def crop_in_xyz(img, out_dir, fp, sub):
    xmin, xmax = (sub%10 - 1) * 128, sub%10 * 128
    ymin, ymax = (sub//10 - 1) * 128, sub//10 * 128
    # print(sub, xmin, xmax, ymin, ymax)

    img_crop = img[:, ymin:ymax, xmin:xmax]
    # print(img_crop.shape)

    zlen = 8
    n_zranges = img_crop.shape[0] // zlen
    # if n_zranges > 1:
    for i_zrange in range(n_zranges):
        zmin, zmax = i_zrange * zlen, (i_zrange + 1) * zlen
        img_crop_z = img_crop[zmin:zmax]
        print(img_crop_z.shape)

        out_fn = os.path.join(out_dir, os.path.basename(fp).replace('.tif', f'_sub{sub}_zi{i_zrange}.tif'))
        tiff.imwrite(out_fn, img_crop_z)


if __name__ == '__main__':
    # crop Z x 256 x 256 patches into Z x 128 x 128 patches
    # remove somata and blood-vessels from the raw images

    data_dir = 'data_altug_3D'
    partitions = ['train', 'val']
    subdirs = ['images', 'masks'] #, 'sm-bv']

    for part in partitions:
        for sd in subdirs:
            out_dir = os.path.join(data_dir, part + '_crop', sd)
            os.makedirs(out_dir, exist_ok=True)

            filepaths = sorted(glob.glob(os.path.join(data_dir, part, sd, '*.tif')))
            for fp in filepaths:
                img = tiff.imread(fp)
                # z, y, x = img.shape
                # print(fp, img.shape)

                # read the sm-bv mask and set those pixels to 0
                if sd == 'images':
                    sm_bv_fp = os.path.join(data_dir, part, 'sm-bv', os.path.basename(fp))
                    sm_bv_mask = tiff.imread(sm_bv_fp)
                    img[sm_bv_mask > 0] = 0
                    # print('removed somata and blood vessels')
                
                if '1802_FOV_005_t_' in os.path.basename(fp) and '_x1_168_y1_527' in os.path.basename(fp):
                    # for these images, only the first 10 slices were labeled
                    img = img[:10]

                if img.shape[1:] == (256, 256):
                    subs = [11, 12, 21, 22]
                elif img.shape[1:] == (128, 128):
                    subs = [11]
                else:
                    raise NotImplementedError(f'Unexpected image shape: {img.shape}')
                for sub in subs:
                    crop_in_xyz(img, out_dir, fp, sub)
            files_out = sorted(glob.glob(os.path.join(out_dir, '*.tif')))
            for fo in files_out:
                img_out = tiff.imread(fo)
                print(img_out.shape)
