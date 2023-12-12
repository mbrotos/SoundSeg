#!/bin/bash

# Obtain dataset from Zenodo
# MUSDB18 - a corpus for music separation https://doi.org/10.5281/zenodo.1117372
# PLEASE NOTE: This dataset is quite large (>50GB) and may take a while to download.

# Alternatively, you can download a subset of the dataset (~6GB) with the following command: 
# https://drive.google.com/file/d/1_kdifA4ztVXBveb9FYzmY49fvAKZmIJF/view?usp=sharing
gdown 1_kdifA4ztVXBveb9FYzmY49fvAKZmIJF && unzip data_wav.zip && rm data_wav.zip

# Create virtual environment
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt

# Create required directories
mkdir -p models processed_data

# Preprocess dataset
python preprocessing.py --dsType train
python preprocessing.py --dsType test

# Tensorflow dataset prep
python dataset_prep.py