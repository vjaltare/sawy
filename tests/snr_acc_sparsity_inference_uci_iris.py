"""
Generate data for sweeping input noise for networks trained under varying levels of noise/SNR.

Load models from:
As of 05/13/2026 the trained models are located in /local_disk/vikrant/trident/models/snr_sweep_iris_2026-05-13_noise_*.pkl where * represents the std of injected noise.

Save data to:
/local_disk/vikrant/trident/logs/snr_sweep_uci_iris_DATE.pkl

Data fields:
accuracy, sparsity, snr (calculated for a given inference noise level), noise_std (during inference), model_index, resample_index, training_snr, training_sparsity

"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

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
import glob

import pickle
from collections import defaultdict
from functools import partial
from tqdm import tqdm
from datetime import date

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris



from utils import ternary_activation, load_cifar10_augment, dual_sample_ternary, load_uci_iris
from models import TernaryStochasticActivation, DualSampleTernary, CustomLinear, FFN 

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

today = date.today().isoformat()

# Path for loading the models: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"

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
        snr_models: list,
        model_preact_data: dict,
        num_resamples: int,
        metrics_dict: dict,
        noise_std_arr: jax.Array,
        X_test: jax.Array,
        y_test: jax.Array,
        save_metrics: bool = False,
        filename: str = None,

    ):
    
    # for loop over the model
    for model_idx, model in tqdm(enumerate(snr_models), total=len(snr_models)):
        assert model_preact_data['model_name'][model_idx] == model, f"Model name mismatch: {model_preact_data['model_name'][model_idx]} vs {model}"

        # load the model
        # print(f"Model under test: {model} | models tested so far: {model_idx}/{len(snr_models)}")
        rms_preact = model_preact_data['rms_preactivation'][model_idx]
        

        payload = pickle.load(open(model, "rb"))
        model_state = payload['state']
        model_data = payload['data']
        model_configs = payload['configs']


        # for loop over noise/snr
        for noise_idx, noise_std in enumerate(noise_std_arr):
            # print(jnp.array(noise_std))

            # for loop for resampling
            for r in range(num_resamples):

                seed = model_configs['seed'] + r
                rngs = nnx.Rngs(
                    params=seed + 0,
                    dropout=seed + 1,
                    activation=seed + 2,
                    next=seed+3
                    )
        
                # load the model states
                # print(noise_std)
                trident_model = FFN(
                    layers=model_configs['layers'],
                    noise_std=noise_std.item(),
                    threshold=model_configs['threshold'],
                    ActivationFunction=DualSampleTernary,
                    rngs=rngs
                )

                graphdef_t, _ = nnx.split(trident_model)
                trident_model = nnx.merge(graphdef_t, model_state)

                # compute model accuracy
                accuracy, preds = pred_step(trident_model, X_test, y_test)

                # compute activations
                hl_activations = trident_model.activation(trident_model.layers[0](X_test))
                sparsity = jnp.mean(hl_activations == 0.0)
                snr = 20*jnp.log10(rms_preact/noise_std)


                # append everything to the dictionary
                metrics_dict['accuracy'].append(accuracy.item())
                metrics_dict['sparsity'].append(sparsity.item())
                metrics_dict['snr'].append(snr.item())
                metrics_dict['noise_std'].append(noise_std.item())
                metrics_dict['model_index'].append(model_idx)
                metrics_dict['resample_index'].append(r)
                # TODO: Rerun this cell to include training snr and sparsity
                metrics_dict['training_snr'].append(model_data['training_snr'])
                metrics_dict['training_sparsity'].append(model_data['trianing_sparsity'])

                if r == num_resamples-1:
                    print(f"Training Noise: {model_configs['noise_std']} | Avg. accuracy: {sum(metrics_dict['accuracy'][-num_resamples:])/num_resamples:.4f} | Avg sparsity: {sum(metrics_dict['sparsity'][-num_resamples:])/num_resamples:.4f} | SNR: {snr:.4f}")


    # at the end of each resample, print the metrics for the last noise level
    # print(f"Model Index: {model_idx} | Noise STD: {noise_std:.4f} | Accuracy: {accuracy:.4f} | Sparsity: {sparsity:.4f} | SNR: {snr:.4f}")

    # optional save the metrics dictionary as pkl
    if save_metrics:

        assert filename is not None, f"Provide a valid file name for saving the data!"

        with open(os.path.join(DATA_PATH, filename), "wb") as f:
            pickle.dump(metrics_dict, f)

    print(metrics_dict)


def main():

    # Loading the model names
    snr_models = glob.glob(os.path.join(MODEL_PATH, "snr_sweep_iris_2026-05-13*"))

    ## PORTING THIS CELL TO AN INDEPENDENT SCRIPT
    # load the data as it was during training to preserve inference on the test set only
    _, X_test, _, y_test = load_uci_iris(normalize=True, key=101, train_test_split=0.7)

    # setting up sweep for noise
    noise_std_arr = jnp.logspace(-5, 5, 50)

    # load the model_preactivations_rms data
    model_preact_data = pickle.load(open(os.path.join(DATA_PATH, "models_preactivations_rms_2026-05-13.pkl"), "rb"))

    # number of noise resamples
    num_resamples = 100

    # store the metrics in a dictionary
    metrics_dict = defaultdict(list)

    # get the date
    today = date.today().isoformat()
    filename = f"snr_sweep_uci_iris_{today}.pkl"

    generate_data(
        filename=filename,
        snr_models=snr_models,
        model_preact_data = model_preact_data,
        num_resamples = num_resamples,
        metrics_dict = metrics_dict,
        noise_std_arr = noise_std_arr,
        X_test = X_test,
        y_test = y_test,
        save_metrics = True,
    )



if __name__ == "__main__":
    main()