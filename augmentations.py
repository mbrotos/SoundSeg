import numpy as np

def consecutive_oversample(mixes, vocals):
    """Creates more training data by concatenating the spectrograms with the next spectrogram in the batch. 
    I.e. the first half of the first spectrogram is concatenated with the first half of the second spectrogram,
    and the second half of the second spectrogram is concatenated with the first half of the third spectrogram, etc.
    You can ignore the half of the first and last spectrogram that is not concatenated with another spectrogram.
    Use axis 2 for halving and concatenating the spectrograms.

    Args:
        mixes (None,512,128,1): spectrograms of the mixes
        vocals (None,512,128,1): spectrograms of the vocals
    """
    def concatenate_halves(data):
        # Assuming the third dimension (axis 2) of each spectrogram has size 128
        # We take the first 64 columns of each spectrogram for concatenation
        half_size = data.shape[2] // 2
        concatenated = []

        for i in range(len(data) - 1):
            first_half = data[i, :, :half_size, :]
            second_half = data[i + 1, :, half_size:, :]
            concatenated.append(np.concatenate([first_half, second_half], axis=1))

        return np.array(concatenated)

    # Apply the concatenation logic to each of the input arrays
    concatenated_mixes = concatenate_halves(np.copy(mixes))
    concatenated_vocals = concatenate_halves(np.copy(vocals))

    return concatenated_mixes, concatenated_vocals

def blackout(mixes, vocals):
    """Choose a random segment along the time axis(2) and set all the frequencies to zero.
    The mix spectrogram and the vocal spectrogram should have the same blackout segment.

    Args:
        mixes (None,512,128,1): spectrograms of the mixes
        vocals (None,512,128,1): spectrograms of the vocals
    """
    # Assuming the third dimension (axis 2) of each spectrogram has size 128
    # We take the first 64 columns of each spectrogram for concatenation
    mix_blackout = np.copy(mixes)
    vocal_blackout = np.copy(vocals)
    blackout_size = mix_blackout.shape[2] // 2

    for i in range(len(mixes)):
        # Choose a random starting point for the blackout
        start = np.random.randint(0, mix_blackout.shape[2] - blackout_size)
        # Set the frequencies to zero
        mix_blackout[i, :, start:start + blackout_size, :] = 0
        vocal_blackout[i, :, start:start + blackout_size, :] = 0

    return mix_blackout, vocal_blackout