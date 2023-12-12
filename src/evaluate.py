import tensorflow as tf
import numpy as np
import argparse
from model import get_model
import config as cfg
import mir_eval.separation
from scaler import normalize, denormalize
import librosa
import json
import tqdm
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--train_loss", action="store_true", default=False)
    parser.add_argument("--mix", action="store_true", default=False)
    
    args = parser.parse_args()

    args.model_name = (
        "model_20231206-230752_6ccd61a5a81a442aaba690e772dbfdbf"
        if args.model_name == ""
        else args.model_name
    )

    # load model args from json in model folder
    with open("models/" + args.model_name + "/args.json") as f:
        model_args = json.load(f)

    print("Model args: ", model_args)
    print("Model name: ", args.model_name)
    print('Eval args:')
    print(args)
    
    weight_type = 'train' if args.train_loss else 'val'
    
    model = get_model((cfg.FREQUENCY_BINS, cfg.SAMPLE_SZ), num_classes=1)
    model.load_weights(f"models/{args.model_name}/{args.model_name}-{weight_type}.hdf5")

    # load data
    mix_mags_test = np.load("./processed_data/mix_mags_test_512x128.npy", mmap_mode="r")
    mix_phases_test = np.load(
        "./processed_data/mix_phases_test_512x128.npy", mmap_mode="r"
    )
    vocal_test = np.load(
        f"./processed_data/vocal_{'masks' if model_args['mask'] else 'mags'}_test_512x128.npy",
        mmap_mode="r",
    )

    print("Normalizing...")
    mix_mags_test_norm, vocal_test_norm, mix_mags_test_norm_factors = normalize(
        np.copy(mix_mags_test),
        np.copy(vocal_test),
        normalization=model_args["normalization"],
        quantile_scaler=model_args["quantile_scaler"],
    )

    if os.path.isfile("models/" + args.model_name + f"/pred_norm-num_samples-{args.num_samples}.npy"):
        print("Loading predictions...")
        pred_norm = np.load("models/" + args.model_name + f"/pred_norm-num_samples-{args.num_samples}.npy")[: args.num_samples]
    else:
        print("Predicting...")
        pred_norm = model.predict(mix_mags_test_norm[: args.num_samples])

        # Cache predictions
        np.save(
            "models/" + args.model_name + f"/pred_norm-num_samples-{args.num_samples}.npy",
            pred_norm,
            allow_pickle=False,
        )

    print("Denormalizing...")
    pred = denormalize(
        pred_norm,
        mix_mags_test_norm_factors[: len(pred_norm)],
        normalization=model_args["normalization"],
        quantile_scaler=model_args["quantile_scaler"],
    )
    # It is important to denormalize the true vocal as well, because the we use the mix norm factors
    true_vocal = denormalize(
        vocal_test_norm[: len(pred_norm)],
        mix_mags_test_norm_factors[: len(pred_norm)],
        normalization=model_args["normalization"],
        quantile_scaler=model_args["quantile_scaler"],
    )
    
    if args.mix:
        print("Denormalizing mix...")
        mix = denormalize(
            mix_mags_test_norm[: len(pred_norm)],
            mix_mags_test_norm_factors[: len(pred_norm)],
            normalization=model_args["normalization"],
            quantile_scaler=model_args["quantile_scaler"],
        )

    print("Transforming to wavelets using stft...")
    pred_waves = []
    vocal_waves = []
    mix_waves = []

    step=5
    for i in tqdm.tqdm(range(0, len(pred) - (len(pred) % step), step)):
        cur_pred = np.concatenate(pred[i : i + step], axis=1)
        cur_phase = np.concatenate(mix_phases_test[i : i + step], axis=1)
        cur_true_vocal = np.concatenate(true_vocal[i : i + step], axis=1)
        pred_waves.append(librosa.istft(
                cur_pred[:, :, 0] * cur_phase[:, :, 0],
                hop_length=cfg.HOP_SIZE,
                window="hann",
            )
        )
        vocal_waves.append(librosa.istft(
                cur_true_vocal[:, :, 0] * cur_phase[:, :, 0],
                hop_length=cfg.HOP_SIZE,
                window="hann",
            )
        )
        if args.mix:
            cur_mix = np.concatenate(mix[i : i + step], axis=1)
            mix_waves.append(librosa.istft(
                    cur_mix[:, :, 0] * cur_phase[:, :, 0],
                    hop_length=cfg.HOP_SIZE,
                    window="hann",
                )
            )

    print("Computing metrics...")
    pred_waves = np.array(pred_waves)
    vocal_waves = np.array(vocal_waves)
    
    
    # Remove pred_waves that are all zeros, also remove corresponding vocal_waves
    pred_waves = pred_waves[np.sum(pred_waves, axis=1) != 0]
    vocal_waves = vocal_waves[np.sum(pred_waves, axis=1) != 0]
    
    pred_waves = pred_waves[np.sum(vocal_waves, axis=1) != 0]
    vocal_waves = vocal_waves[np.sum(vocal_waves, axis=1) != 0]
    
    dist = np.abs(vocal_waves).sum(axis=1)
    indices = np.where(dist < 100)[0]
    pred_waves = np.delete(pred_waves, indices, axis=0)
    vocal_waves = np.delete(vocal_waves, indices, axis=0)
    
    if args.mix:
        mix_waves = np.array(mix_waves)
        mix_waves = mix_waves[np.sum(pred_waves, axis=1) != 0]
        mix_waves = mix_waves[np.sum(vocal_waves, axis=1) != 0]
        mix_waves = np.delete(mix_waves, indices, axis=0)
    
    # Compute metrics for every n samples
    sdr_lst, sir_lst, sar_lst = [], [], []
    n=5 # Keep  this <=100, as defined in mir_eval.separation
    for i in tqdm.tqdm(range(0, len(pred_waves) - (len(pred_waves) % n), n)):
        sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
            pred_waves[i : i + n], vocal_waves[i : i + n], compute_permutation=False
        )
        sdr_lst.append(sdr)
        sir_lst.append(sir)
        sar_lst.append(sar)

    print("SDR: ", np.mean(sdr_lst))
    print("SIR: ", np.mean(sir_lst))
    print("SAR: ", np.mean(sar_lst))

    print("Saving results...")
    # Save results as json, include args
    results = {
        "SDR": np.mean(sdr_lst),
        "SIR": np.mean(sir_lst),
        "SAR": np.mean(sar_lst),
        "args": model_args,
    }
    with open(f"models/{args.model_name}/results-{weight_type}-num_samples-{args.num_samples}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()