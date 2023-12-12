#!/bin/bash
#SBATCH --account=def-miranska
#SBATCH --gres=gpu:a100:1        # request GPU "generic resource"
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=64G       # memory per node
#SBATCH --time=0-10:00      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID

module load cuda cudnn 
module load python/3.9

source ./venv/bin/activate


python train.py --normalization time --epochs 20 --batch_size 64 --loss mse --augmentations
python train.py --normalization frequency --epochs 20 --batch_size 64 --loss mse --augmentations
python train.py --normalization time --epochs 20 --batch_size 64 --quantile_scaler --loss mse --augmentations
python train.py --normalization frequency --epochs 20 --batch_size 64 --quantile_scaler --loss mse --augmentations

python train.py --normalization time --epochs 20 --batch_size 64 --loss mae --augmentations
python train.py --normalization frequency --epochs 20 --batch_size 64 --loss mae --augmentations
python train.py --normalization time --epochs 20 --batch_size 64 --quantile_scaler --loss mae --augmentations
python train.py --normalization frequency --epochs 20 --batch_size 64 --quantile_scaler --loss mae --augmentations

# Mask does not work
# python train.py --normalization time --mask --epochs 10 --batch_size 25 --dataset_size 6000
# python train.py --normalization frequency --mask --epochs 10 --batch_size 25 --dataset_size 6000
# python train.py --normalization time --mask --epochs 10 --batch_size 25 --dataset_size 6000 --quantile_scaler
# python train.py --normalization frequency --mask --epochs 10 --batch_size 25 --dataset_size 6000 --quantile_scaler
