# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: soundseg
#     language: python
#     name: python3
# ---

# %%
import musdb
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

# %%
mus_train = musdb.DB(root="data_wav", subsets="train", is_wav=True)

# %%
mus_train[0]

# %%
display.Audio(mus_train[0].audio.T, rate=41000)


# %%
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


# %%
label = "Audio"
waveform = mus_train[0].audio.T[0]
spectrogram1 = get_spectrogram(waveform)

print("Label:", label)
print("Waveform shape:", waveform.shape)
print("Spectrogram shape:", spectrogram1.shape)
print("Audio playback")
display.display(display.Audio(waveform, rate=41000))


# %%

# %%
def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


# %%
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform)
axes[0].set_title("Waveform")
axes[0].set_xlim([0, 41000])

plot_spectrogram(spectrogram1.numpy(), axes[1])
axes[1].set_title("Spectrogram")
plt.suptitle(label.title())
plt.show()

# %%
label = "Vocals"
waveform = mus_train[0].stems[4].T[0]
spectrogram2 = get_spectrogram(waveform)

print("Label:", label)
print("Waveform shape:", waveform.shape)
print("Spectrogram shape:", spectrogram2.shape)
print("Audio playback")
display.display(display.Audio(waveform, rate=41000))

# %%
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform)
axes[0].set_title("Waveform")
axes[0].set_xlim([0, 41000])

plot_spectrogram(spectrogram2.numpy(), axes[1])
axes[1].set_title("Spectrogram")
plt.suptitle(label.title())
plt.show()

# %%
label = "other"
waveform = mus_train[0].stems[3].T[0]
spectrogram3 = get_spectrogram(waveform)

print("Label:", label)
print("Waveform shape:", waveform.shape)
print("Spectrogram shape:", spectrogram3.shape)
print("Audio playback")
display.display(display.Audio(waveform, rate=41000))

# %%
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])

axes[0].set_title("Waveform")
axes[0].set_xlim([0, 41000])

plot_spectrogram(spectrogram3.numpy(), axes[1])
axes[1].set_title("Spectrogram")
plt.suptitle(label.title())
plt.show()

# %%
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])

axes[0].set_title("Waveform")
axes[0].set_xlim([0, 41000])

plot_spectrogram((spectrogram2 + spectrogram3).numpy(), axes[1])
axes[1].set_title("Spectrogram")
plt.suptitle(label.title())
plt.show()

# %%

# %%
from scipy.io import wavfile

# %%
import scipy.io

# %%
samplerate, data = wavfile.read(
    r"G:\My Drive\School\EE8223\Code\EE8223-Deep-Learning-F2023\SoundSeg\data_wav\train\A Classic Education - NightOwl\mixture.wav"
)

# %%
samplerate

# %%
data.shape

# %%
data[0, 0], data[0, 1]

# %%
(data[0, 0] + data[0, 1]) / 2

# %%
data_mono = np.mean(data, axis=1)

# %%
data_mono[0]

# %%
length = data_mono.shape[0] / samplerate

# %%
import matplotlib.pyplot as plt
import numpy as np

time = np.linspace(0.0, length, data_mono.shape[0])
plt.plot(time, data_mono, label="Mono channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# %%
display.Audio(data_mono, rate=44100)

# %%
from scipy import signal

# %%
f, t, Zxx = signal.stft(data_mono, 10e3, nperseg=256, scaling="psd")
ref = 1.0
amin = 1e-10
top_db = 80.0
magnitude = np.abs(Zxx)
ref_value = np.abs(ref)
log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
log_spec = np.maximum(log_spec, log_spec.max() - top_db)

# %%
plt.pcolormesh(t, f, log_spec, shading="gouraud")
plt.title("STFT Magnitude")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.yscale("symlog")
plt.show()

# %%
_, xrec = signal.istft(Zxx, 10e3)

# %%
xrec.shape

# %%
time = np.linspace(0.0, length, data_mono.shape[0])
plt.plot(time, xrec, label="Mono channel")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# %%
np.max(data_mono - xrec)

# %%
np.min(data_mono - xrec)

# %%
display.Audio(xrec, rate=44100)

# %%
import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

# %%
FRAME_SIZE = 2048
HOP_SIZE = 512

# %%
S_scale = librosa.stft(data_mono, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

# %%
S_scale.shape

# %%
Y_scale = np.abs(S_scale) ** 2


# %%
def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(
        Y, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis
    )
    plt.colorbar(format="%+2.f")


# %%
plot_spectrogram(S_scale, 44100, HOP_SIZE)

# %%
Y_log_scale = librosa.power_to_db(Y_scale)

# %%
from PIL import Image

# %%
Y_log_scale.shape

# %%
im = Image.fromarray(Y_log_scale)
# im.save("test.tif")
# https://stackoverflow.com/a/57204349/3600365

# %%
plot_spectrogram(Y_log_scale, 44100, HOP_SIZE, y_axis="log")

# %%
y_s = librosa.db_to_power(Y_log_scale)

# %%
y_inv = librosa.griffinlim(np.sqrt(y_s), n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

# %%
display.Audio(y_inv, rate=44100)

# %%
S = librosa.feature.melspectrogram(y=data_mono, sr=44100, n_mels=128)

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(
    S_dB, x_axis="time", y_axis="mel", sr=44100, fmax=44100 / 2.0, ax=ax
)
fig.colorbar(img, ax=ax, format="%+2.0f dB")
ax.set(title="Mel-frequency spectrogram")

# %%
audio = librosa.feature.inverse.mel_to_audio(S, sr=44100)

# %%
display.Audio(audio)

# %%
