#!/bin/bash

source ./venv/bin/activate

# Individual evaluation threads for all models in the models directory will be spawned
# Be sure you have enough compute resources to run this script 
python run_eval.py

# Alternatively, you can run the evaluation script for a single model
# python evaluate.py --model_name '{name}'

# You can download all the models with evaluation results from the following link:
# https://drive.google.com/file/d/1_myj2HVAg-g6SR44jSRB9tIkv7BhF1gU/view?usp=sharing
# gdown 1_myj2HVAg-g6SR44jSRB9tIkv7BhF1gU && unzip models.zip

# Please see ./analysis.ipynb to create the results table in the paper
