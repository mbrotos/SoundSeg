#!/bin/bash

# Create virtual environment
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

# Obtain dataset from Zenodo, unzip, and place in data_wav folder
# MUSDB18 - a corpus for music separation https://doi.org/10.5281/zenodo.3338373
# PLEASE NOTE: This dataset is quite large (>20GB -- zipped) and may take a while to download.

# Alternatively, you can download a subset of the dataset (~9GB) with the following command: 
# https://drive.google.com/file/d/1_kdifA4ztVXBveb9FYzmY49fvAKZmIJF/view?usp=sharing
gdown 1_kdifA4ztVXBveb9FYzmY49fvAKZmIJF && unzip data_wav.zip

# Create required directories
mkdir -p models processed_data

# Preprocess dataset
python preprocessing.py --dsType train
python preprocessing.py --dsType test

# Tensorflow dataset prep
python dataset_prep.py
