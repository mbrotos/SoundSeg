# Try it out: https://soundseg.com/

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbrotos/SoundSeg/blob/main/demo.ipynb)
[![arXiv](https://img.shields.io/badge/arXiv-2405.20059-b31b1b.svg)](https://arxiv.org/abs/2405.20059)

# SoundSeg

## Introduction
SoundSeg is an advanced audio processing project, focused on segmenting and analyzing sound data. The core of the project revolves around machine learning models, specifically designed for audio source seperation of songs into the vocal component.

Please see the projects preprint for more details: [Spectral Mapping of Singing Voices: U-Net-Assisted Vocal Segmentation](https://arxiv.org/abs/2405.20059)

## Example Audio and Spectrograms

### Original Audio
- Mixture: https://drive.google.com/file/d/1_o-BfOWwSTL52qE1aadqPhRNH2Top-gA/view?usp=sharing

### Predicted Output
- Vocal (Predicted): https://drive.google.com/file/d/1_ncbyUDC5GpL0-yJN7bObyk7KMjnQGuy/view?usp=sharing

### Spectrograms
- Mixture: <br/>
![Mixture Spectrogram](assets/jimi_mix_spec.png)
- Vocal (Predicted): <br/>
![Vocal Spectrogram](assets/jimi_pred_vocal_spec.png)

## System Model

![image](https://github.com/user-attachments/assets/060031b6-29cd-4cf6-a935-cc033953d578)


## Contents
- **MUSDB_README.md**: Provides details on the MUSDB 2018 Dataset, crucial for training and testing the models.
- **demo.ipynb**: A Jupyter notebook demonstrating the capabilities of SoundSeg.
- **src**: Contains the source code of the project.
- **Scripts**: Shell scripts for preprocessing (`0_preprocess.sh`), training (`1_train.sh`), and evaluation (`2_eval.sh`).
- **Python Modules**: Core modules like `model.py`, `preprocessing.py`, `train.py`, `evaluate.py`, etc., for model development and data handling.
- **requirements.txt**: Lists all the necessary dependencies.
- **analysis.ipynb**: Additional Jupyter notebook for deeper analysis.

## Usage

1. **Preprocessing**: Run the `0_preprocess.sh` script to prepare your data.
2. **Training**: Execute the `1_train.sh` script to train the models.
3. **Evaluation**: Use the `2_eval.sh` script for model evaluation.
4. **Demo**: Explore `demo.ipynb` for hands-on examples and usage demonstrations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mbrotos/SoundSeg/blob/main/demo.ipynb)

5. **Analysis**: Delve into `analysis.ipynb` for in-depth analytical insights.

## Model Weights

Download the pre-trained model weights from the following links:
- Google Drive Link [https://drive.google.com/file/d/1_myj2HVAg-g6SR44jSRB9tIkv7BhF1gU/view?usp=sharing]

## Results

| **SDR** | **SIR** | **SAR** | **Normalization** | **Scaler** | **Loss** |
|---------|---------|---------|-------------------|------------|----------|
| 7.1     | 25.2    | 7.2     | frequency         | Min/Max    | MAE      |
| 7.1     | 25.1    | 7.2     | time              | Min/Max    | MAE      |
| 6.7     | 24.8    | 6.8     | frequency         | Min/Max    | MSE      |
| 5.7     | 23.9    | 5.8     | time              | Quantile   | MAE      |
| 5.6     | 23.3    | 5.7     | time              | Min/Max    | MSE      |
| 4.8     | 22.6    | 4.9     | time              | Quantile   | MSE      |
| -0.9    | 16.6    | -0.6    | frequency         | Quantile   | MSE      |
| -2.1    | 15.8    | -1.8    | frequency         | Quantile   | MAE      |


## Acknowledgments

Special thanks to the creators of the MUSDB 2018 Dataset and all contributors to this project.
