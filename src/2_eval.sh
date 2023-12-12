#!/bin/bash

source ./venv/bin/activate

# Individual evaluation threads for all models in the models directory will be spawned
# Be sure you have enough compute resources to run this script 
python run_eval.py

# Alternatively, you can run the evaluation script for a single model
# python evaluate.py --model_name '{name}'
