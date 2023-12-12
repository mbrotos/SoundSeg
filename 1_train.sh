#!/bin/bash
source ./venv/bin/activate

# Unit tests
python test_audio_processing.py

# Train models
python -u train.py --normalization time --epochs 20 --batch_size 64 --quantile_scaler --loss mae --augmentations     
python -u train.py --normalization time --epochs 20 --batch_size 64  --loss mae --augmentations
python -u train.py --normalization frequency --epochs 20 --batch_size 64 --quantile_scaler --loss mae --augmentations
python -u train.py --normalization frequency --epochs 20 --batch_size 64  --loss mae --augmentations
python -u train.py --normalization time --epochs 20 --batch_size 64 --quantile_scaler --loss mse --augmentations     
python -u train.py --normalization time --epochs 20 --batch_size 64  --loss mse --augmentations
python -u train.py --normalization frequency --epochs 20 --batch_size 64 --quantile_scaler --loss mse --augmentations
python -u train.py --normalization frequency --epochs 20 --batch_size 64  --loss mse --augmentations