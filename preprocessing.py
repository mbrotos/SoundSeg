import os
import librosa
import numpy as np
import glob
import config as cfg
import argparse

def main(dsType):
    songMixturePaths = sorted(glob.glob(f"data_wav/{dsType}/*/mixture.wav"))
    songVocalPaths = sorted(glob.glob(f"data_wav/{dsType}/*/vocals.wav"))

    # check if folder exists, if not create it
    if not os.path.exists(f'processed_data/{dsType}'):
        os.makedirs(f'processed_data/{dsType}')

    for i, (mixPath, vocalPath) in enumerate(zip(songMixturePaths, songVocalPaths)):
        songName = os.path.normpath(mixPath).split(os.sep)[-2]
        
        mix_data, _ = librosa.load(mixPath, sr=cfg.SR, mono=True)
        vocal_data, _ = librosa.load(vocalPath, sr=cfg.SR, mono=True)
        
        mix_stft = librosa.stft(mix_data, n_fft=cfg.FRAME_SIZE, hop_length=cfg.HOP_SIZE, window='hann')
        vocal_stft = librosa.stft(vocal_data, n_fft=cfg.FRAME_SIZE, hop_length=cfg.HOP_SIZE, window='hann')
        
        mix_mag, mix_phase = librosa.magphase(mix_stft)
        vocal_mag, _ = librosa.magphase(vocal_stft) # We assume vocal phase information is not available
        
        # Chunk the data into square samples of size sample_sz by sample_sz
        numOfSamples = mix_mag.shape[1] // cfg.SAMPLE_SZ # This will cut off the last bit of the song
        
        # Print some info
        print(f"Song: {songName}")
        print(f"Number of samples: {numOfSamples}")
        print(f"Shape of mix_mag: {mix_mag.shape}")
        print(f"Shape of vocal_mag: {vocal_mag.shape}")
            
        mix_mag_samples = np.array(np.split(mix_mag[:512,:cfg.SAMPLE_SZ*numOfSamples], numOfSamples, axis=1))
        vocal_mag_samples = np.array(np.split(vocal_mag[:512,:cfg.SAMPLE_SZ*numOfSamples], numOfSamples, axis=1))
        
        # Trim phase information to match the shape of the magnitude
        mix_phase = mix_phase[:512,:numOfSamples*cfg.SAMPLE_SZ]
        mix_phase_samples = np.array(np.split(mix_phase, numOfSamples, axis=1))
        
        # Write the samples to a compressed numpy file
        np.savez_compressed(f'processed_data/{dsType}/{songName}.npz',
                            mix_mag=mix_mag_samples,
                            vocal_mag=vocal_mag_samples,
                            mix_phase=mix_phase_samples)
        
        # Print some info
        print(f"Shape of mix_mag_samples: {mix_mag_samples.shape}")
        print(f"Shape of vocal_mag_samples: {vocal_mag_samples.shape}")
        print(f"Shape of mix_phase_samples: {mix_phase_samples.shape}")
        print(f"Number of songs processed: {i+1}/{len(songMixturePaths)}\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dsType', type=str, default='train', help='Dataset type, either train or test')
    args = parser.parse_args()
    main(args.dsType)