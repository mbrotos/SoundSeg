import tensorflow as tf
import numpy as np
from model import get_model
import argparse
import config as cfg
import datetime
from scaler import normalize, denormalize
import pickle
import os
import json
from augmentations import consecutive_oversample, blackout
from uuid import uuid4
import librosa

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help='Number of epochs to train for')
    parser.add_argument("--batch_size", type=int, default=5, help='Batch size for training')
    parser.add_argument("--normalization", type=str, default="frequency", help='Normalization axis (time or frequency)')
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate for training')
    parser.add_argument("--mask", action="store_true", default=False, help='Experimental. Causes unstable training.')
    parser.add_argument("--quantile_scaler", action="store_true", default=False, help='Toggle quantile scaling as the normalization method')
    parser.add_argument("--q_min", type=float, default=25.0, help='Minimum quantile for quantile scaling')
    parser.add_argument("--q_max", type=float, default=75.0, help='Maximum quantile for quantile scaling')
    parser.add_argument("--loss", type=str, default="mse", help='Loss function to use (mse or mae)')
    parser.add_argument("--dataset_size", type=int, default=None, help='Number of samples to use from the dataset (None = all)')
    parser.add_argument("--augmentations", action="store_true", default=False, help='Toggle data augmentations (splicing, and blackout)')
    parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument("--mmap", action="store_true", default=True, help='Toggle memory mapping for dataset loading (helps with large datasets and limited RAM)')

    args = parser.parse_args()
    
    print('Args:')
    print(args)
    
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # load data
    mix_mags_train = np.load("./processed_data/mix_mags_train_512x128.npy", mmap_mode='r' if args.mmap else None)[:args.dataset_size]
    mix_phases_train = np.load("./processed_data/mix_phases_train_512x128.npy", mmap_mode='r' if args.mmap else None)[:args.dataset_size]
    vocal_train = np.load( f"./processed_data/vocal_mags_train_512x128.npy", mmap_mode='r' if args.mmap else None)[:args.dataset_size]

    mix_mags_train_norm, vocal_train_norm, mix_mags_train_norm_factors = normalize(
        np.copy(mix_mags_train),
        np.copy(vocal_train),
        normalization=args.normalization,
        quantile_scaler=args.quantile_scaler,
        q_min=args.q_min,
        q_max=args.q_max,
    )
    if args.augmentations:
        print('Appling augmentations...')
        
        # Remove outliers
        true_vocal = denormalize(
            vocal_train_norm,
            mix_mags_train_norm_factors,
            normalization=args.normalization,
            quantile_scaler=args.quantile_scaler,
        )

        vocal_waves = []

        for i in range(0, len(true_vocal)):
            cur_phase = np.concatenate(mix_phases_train[i : i + 1], axis=1)
            cur_true_vocal = np.concatenate(true_vocal[i : i + 1], axis=1)
            vocal_waves.append(librosa.istft(
                    cur_true_vocal[:, :, 0] * cur_phase[:, :, 0],
                    hop_length=cfg.HOP_SIZE,
                    window="hann",
                )
            )
        vocal_waves = np.array(vocal_waves)
        dist = np.abs(vocal_waves).sum(axis=1)
        indices = np.where(dist < 100)[0]

        mix_mags_train_norm = np.delete(mix_mags_train_norm, indices, axis=0)
        vocal_train_norm = np.delete(vocal_train_norm, indices, axis=0)
        mix_mags_train_norm_factors = np.delete(mix_mags_train_norm_factors, indices, axis=0)
        mix_phases_train = np.delete(mix_phases_train, indices, axis=0)
        
        # Splicing and blackout
        mix_blackout, vocal_blackout = blackout(mix_mags_train_norm, vocal_train_norm)
        mix_blackout = mix_blackout[:mix_blackout.shape[0]//4]
        vocal_blackout = vocal_blackout[:vocal_blackout.shape[0]//4]
        mix_consec, vocal_consec = consecutive_oversample(mix_mags_train_norm, vocal_train_norm)
        mix_consec = mix_consec[:mix_consec.shape[0]//2]
        vocal_consec = vocal_consec[:vocal_consec.shape[0]//2]
        
        mix_mags_train_norm = np.concatenate((mix_mags_train_norm, mix_consec, mix_blackout), axis=0)
        vocal_train_norm = np.concatenate((vocal_train_norm, vocal_consec, vocal_blackout), axis=0)
        
        
        
        
    print('Datasets:')
    print(f'Mixes: {mix_mags_train_norm.shape}')
    print(f'Vocals: {vocal_train_norm.shape}')
        
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name = f"model_{timestamp}_{uuid4().hex}"
        
    # Save normalization factors as pkl
    with open(f"./models/scaler--{model_name}.pkl", "wb") as f:
        pickle.dump(mix_mags_train_norm_factors, f)

    data_len = len(mix_mags_train_norm)
    
    # shuffle datasets before splitting
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    mix_mags_train_norm = mix_mags_train_norm[indices]
    vocal_train_norm = vocal_train_norm[indices]

    val_len = int(data_len * 0.1)
    val_data = (
        mix_mags_train_norm[-val_len:],
        vocal_train_norm[-val_len:],
    )
    train_data = (
        mix_mags_train_norm[:-val_len],
        vocal_train_norm[:-val_len],
    )

    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = (
        dataset.shuffle(args.batch_size * 2)
        .batch(args.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices(val_data)
    val_ds = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = get_model((cfg.FREQUENCY_BINS, cfg.SAMPLE_SZ), num_classes=1)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=args.loss,
    )

    os.makedirs(f"./models/{model_name}")
    os.makedirs(f"./models/{model_name}/logs")
    
    # Save args as json
    with open(f"./models/{model_name}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
        
    model.summary()

    history = model.fit(
        dataset,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"./models/{model_name}/{model_name}-val.hdf5", save_best_only=True, monitor='val_loss', mode='min', save_weights_only=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"./models/{model_name}/{model_name}-train.hdf5", save_best_only=True, monitor='loss', mode='min', save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f"./models/{model_name}/logs", histogram_freq=1
            ),
        ],
    )

