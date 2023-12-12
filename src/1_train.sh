#!/bin/bash
source ./venv/bin/activate

# Unit tests
python test_audio_processing.py

# Train models for the 2 by 2 by 2 combinations of the following hyperparameters:
    # Normalization: time, frequency
    # Loss: mae, mse
    # Quantile scaler: True, False
python train.py --normalization time --epochs 20 --batch_size 64 --quantile_scaler --loss mae --augmentations     
python train.py --normalization time --epochs 20 --batch_size 64  --loss mae --augmentations
python train.py --normalization frequency --epochs 20 --batch_size 64 --quantile_scaler --loss mae --augmentations
python train.py --normalization frequency --epochs 20 --batch_size 64  --loss mae --augmentations
python train.py --normalization time --epochs 20 --batch_size 64 --quantile_scaler --loss mse --augmentations     
python train.py --normalization time --epochs 20 --batch_size 64  --loss mse --augmentations
python train.py --normalization frequency --epochs 20 --batch_size 64 --quantile_scaler --loss mse --augmentations
python train.py --normalization frequency --epochs 20 --batch_size 64  --loss mse --augmentations
