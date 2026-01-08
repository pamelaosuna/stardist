#!/bin/bash

#SBATCH --job-name=sdist
#SBATCH --output=out_sbatch/%j.out
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1
#SBATCH --account=dldevel
#SBATCH --mem=50G
#SBATCH --time=10:00:00

module load nvidia/cuda/12.3.0
echo "load conda environment"
eval "$(/scratch/dldevel/osuna/miniconda3/bin/conda shell.bash hook)"
conda activate xtc
which python3
echo "loaded conda environment"
echo "start fine-tuning stardist model..."
date

# python3 -u custom_scripts/finetune_2D.py \
#  --img_dir data_altug_2D/images/ \
#  --mask_dir data_altug_2D/masks/ \
#  --out_dir checkpoints/2D/

python3 -u custom_scripts/finetune_3D.py \
 --train_dir data_altug_3D/train_crop/ \
 --val_dir data_altug_3D/val_crop/ \
 --out_dir checkpoints/3D/ \
 --nepochs 200 \
 --batch_size 1 \
 --aug \
 --model_name synapse_stardist_3D_grid=1-1-1_aug=True_bs=1_2025-07-23

echo job finished
date
