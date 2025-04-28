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
echo "start fine-tuning stardist model..."
date


python3 -u custom_scripts/finetune_2D.py \
 --img_dir data_altug/images/ \
 --mask_dir data_altug/masks/ \
 --out_dir models/custom/

echo job finished
date
