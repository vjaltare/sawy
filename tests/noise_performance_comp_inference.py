"""
Inference on trained noise-performance comparison models on UCI-Iris dataset.

- Models are trained with diferent noise levels injected during training.
- During inference take each model and simulate it with either zero noise or the same noise level as training.
- There are three models trained with zero noise, std=0.1, 0.01. There should be 9 permutations of inference datasets.
- Compute statistics over 45 resamples over the dataset (test).

Models are located in: /local_disk/vikrant/trident/models/noise_perf_comp_uci_iris_2026-06-05_noise_NOISE_STD.pkl

Notes: TODO
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import glob

import argparse
import csv
import fcntl
import jax
import math
import jax.numpy as jnp
import optax
import flax
from flax import nnx
from flax.nnx.nn import initializers
from typing import Callable
import json

import pickle
import numpy as np
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from datetime import date

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris



from utils import dual_sample_ternary, load_uci_iris
from models import TernaryStochasticActivation, DualSampleTernary, FFN

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

today = date.today().isoformat()

# Path for loading the models: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"

# parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Noise-performance comparison on UCI-Iris")

    # number of resamples for each noise level
    parser.add_argument("--num_resamples", type=int, default=45)

    # whether to save the metrics
    parser.add_argument("--save_metrics", action="store_true")

    # seed/key
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()
    

## prediction function
def pred_step(
        model: FFN, 
        inputs: jax.Array,
        labels: jax.Array,
        # metrics: nnx.MultiMetric
    ):

    logits = model(inputs)
    # print(logits)
    logits = nnx.softmax(logits, axis=-1)
    prediction = jnp.argmax(logits, axis=-1)
    # print(prediction)
    # accuracy = -1
    accuracy = jnp.mean(prediction == labels)

    return accuracy, prediction

def generate_data(
        noise_perf_comp_models: list,
        num_resamples: int,
        metrics_dict: dict,
        noise_std_arr: jax.Array,
        X_test: jax.Array,
        y_test: jax.Array,
        save_metrics: bool = False,
        filename: str = None,

    ):
    
    # for loop over the model
    for model_idx, model in tqdm(enumerate(noise_perf_comp_models), total=len(noise_perf_comp_models)):
        # assert model_preact_data['model_name'][model_idx] == model, f"Model name mismatch: {model_preact_data['model_name'][model_idx]} vs {model}"

        # # load the model
        # # print(f"Model under test: {model} | models tested so far: {model_idx}/{len(snr_models)}")
        # rms_preact = model_preact_data['rms_preactivation'][model_idx]
        

        payload = pickle.load(open(model, "rb"))
        model_state = payload['state']
        model_data = payload['data']
        model_configs = payload['configs']

        training_snr = float(model_data['training_snr'])

        assert type(training_snr) == float, f"Training SNR should be a float value, but got {type(training_snr)}"

        if training_snr >= 900: 
            training_snr = jnp.inf


        # for loop over noise/snr
        for noise_idx, noise_std in enumerate(noise_std_arr):

            # for loop for resampling
            for r in range(num_resamples):

                seed = model_configs['seed'] + r
                rngs = nnx.Rngs(
                    params=seed + r + noise_idx*2,
                    dropout=seed + r*10 + noise_idx*20,
                    activation=seed + r*100 + noise_idx*200,
                    next=seed + r*1000 + noise_idx*2000
                    )
        
                # load the model states
                # print(noise_std)
                model = FFN(
                    layers=model_configs['layers'],
                    noise_std=noise_std.item(),
                    threshold=model_configs['threshold'],
                    ActivationFunction=DualSampleTernary,
                    rngs=rngs
                )

                graphdef_t, _ = nnx.split(model)
                model = nnx.merge(graphdef_t, model_state)

                # compute model accuracy
                accuracy, preds = pred_step(model, X_test, y_test)

                # compute activations
                # hl_activations = model.activation(model.layers[0](X_test))
                # sparsity = jnp.mean(hl_activations == 0.0)
                hl_preact = model.layers[0](X_test)
                hl_preact = hl_preact.reshape(-1)
                rms_preact = jnp.sqrt(jnp.mean(hl_preact**2))
                inference_snr = 20*jnp.log10(rms_preact/noise_std)

                # test print statement
                # print(f"Training SNR: {training_snr} | Inference Noise: {noise_std} | Accuracy: {accuracy:.4f} | Inference SNR: {inference_snr:.4f}")


                # append everything to the dictionary
                metrics_dict['accuracy'].append(accuracy.item())
                metrics_dict['inference_snr'].append(inference_snr.item())
                metrics_dict['training_snr'].append(training_snr)
                metrics_dict['noise_std'].append(noise_std.item())
                metrics_dict['model_index'].append(model_idx)
                metrics_dict['resample_index'].append(r)

                if r == num_resamples-1:
                    print(f"TRAINING SNR: {training_snr} | INFERENCE SNR: {inference_snr:.4f} | Avg. accuracy: {sum(metrics_dict['accuracy'][-num_resamples:])/num_resamples:.4f} ")


    # at the end of each resample, print the metrics for the last noise level
    # print(f"Model Index: {model_idx} | Noise STD: {noise_std:.4f} | Accuracy: {accuracy:.4f} | Sparsity: {sparsity:.4f} | SNR: {snr:.4f}")

    # optional save the metrics dictionary as pkl
    if save_metrics:

        assert filename is not None, f"Provide a valid file name for saving the data!"

        with open(os.path.join(DATA_PATH, filename), "wb") as f:
            pickle.dump(metrics_dict, f)

    for key in metrics_dict.keys():
        print(f"Key: {key}, #Entries: {len(metrics_dict[key])}")


def main():

    # parse arguments
    args = parse_args()

    # Loading the model names
    noise_perf_comp_models = glob.glob(os.path.join(MODEL_PATH, "noise_perf_comp_uci_iris_2026-06-05_noise_*"))

    # load the data as it was during training to preserve inference on the test set only
    _, X_test, _, y_test = load_uci_iris(normalize=True, key=args.seed, train_test_split=0.7)

    # setting up sweep for noise
    noise_std_arr = jnp.append(jnp.array([0.0]), jnp.logspace(-2, -1, num=2)) # let's keep both 0.01 and 0.1

    # load the model_preactivations_rms data
    model_preact_data = pickle.load(open(os.path.join(DATA_PATH, "models_preactivations_rms_2026-05-13.pkl"), "rb"))

    # number of noise resamples
    num_resamples = args.num_resamples

    # store the metrics in a dictionary
    metrics_dict = defaultdict(list)

    # get the date
    today = date.today().isoformat()
    filename = f"noise_perf_comp_uci_iris_inference_data_{today}.pkl"

    generate_data(
        filename=filename,
        noise_perf_comp_models=noise_perf_comp_models,
        num_resamples = num_resamples,
        metrics_dict = metrics_dict,
        noise_std_arr = noise_std_arr,
        X_test = X_test,
        y_test = y_test,
        save_metrics = args.save_metrics,
    )



if __name__ == "__main__":
    main()