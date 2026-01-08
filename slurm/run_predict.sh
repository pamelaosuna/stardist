#!/bin/bash

#SBATCH --job-name=sdist
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --account=dldevel
#SBATCH --mem=50G
#SBATCH --time=10:00:00

# module load nvidia/cuda/12.3.0
echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate xtc
which python3
echo "loaded conda environment"
echo "start prediction with stardist model..."
date

# python3 -u custom_scripts/predict_2D.py \
#     --img_dir data_altug_2D/images/ \
#     --out_dir data_altug_2D/predictions/ \
#     --model_path checkpoints/2D/weights_best.h5

python3 -u custom_scripts/predict_3D.py \
    --model_name synapse_stardist_3D_grid=1-1-1_aug=True_bs=1_2025-07-23 \
    --img_dir data_altug_3D/val_crop/images/ \
    --out_dir data_altug_3D/predictions/

echo job finished
date