"""
Generate inference data using the saved models with variance loss.

total_loss = ce_loss + alpha*var_loss

Load the models from:
/local_disk/vikrant/trident/models/var_loss_snr_sweep_iris_2026-05-26_alpha_ALPHA_noise_NSD.pkl

Save results to:
/local_disk/vikrant/trident/logs/var_loss_uci_iris_DATE.pkl

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
        var_loss_models: list,
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
    for model_idx, model in tqdm(enumerate(var_loss_models), total=len(var_loss_models)):
        # assert model_preact_data['model_name'][model_idx] == model, f"Model name mismatch: {model_preact_data['model_name'][model_idx]} vs {model}"

        # load the model: Computing RMS preactivations ca n be done inside the loops
        # # print(f"Model under test: {model} | models tested so far: {model_idx}/{len(snr_models)}")
        # rms_preact = model_preact_data['rms_preactivation'][model_idx]
        

        payload = pickle.load(open(model, "rb"))
        model_state = payload['state']
        model_data = payload['data']
        model_configs = payload['configs']


        # for loop over noise/snr
        for noise_idx, noise_std in enumerate(noise_std_arr):
            # print(jnp.array(noise_std))

            # for loop for resampling
            for r in range(num_resamples):

                seed = model_configs['seed'] + r + 100*noise_idx + 10000*model_idx # create a unique seed for each resample, noise level and model
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

                # compute preactivations and signal rms
                hl_preact = trident_model.layers[0](X_test)
                hl_preact = hl_preact.reshape(-1) # compute rms across all preactivations
                rms_preact = jnp.sqrt(jnp.mean(hl_preact**2))

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
                metrics_dict['training_snr'].append(model_data['training_snr'])
                # metrics_dict['training_sparsity'].append(model_data['trianing_sparsity']) 

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


# def main():

#     # Loading the model names
#     snr_models = glob.glob(os.path.join(MODEL_PATH, "snr_sweep_iris_2026-05-13*"))

#     ## PORTING THIS CELL TO AN INDEPENDENT SCRIPT
#     # load the data as it was during training to preserve inference on the test set only
#     _, X_test, _, y_test = load_uci_iris(normalize=True, key=101, train_test_split=0.7)

#     # setting up sweep for noise
#     noise_std_arr = jnp.logspace(-5, 5, 50)

#     # load the model_preactivations_rms data
#     model_preact_data = pickle.load(open(os.path.join(DATA_PATH, "models_preactivations_rms_2026-05-13.pkl"), "rb"))

#     # number of noise resamples
#     num_resamples = 100

#     # store the metrics in a dictionary
#     metrics_dict = defaultdict(list)

#     # get the date
#     today = date.today().isoformat()
#     filename = f"snr_sweep_uci_iris_{today}.pkl"

#     generate_data(
#         filename=filename,
#         snr_models=snr_models,
#         model_preact_data = model_preact_data,
#         num_resamples = num_resamples,
#         metrics_dict = metrics_dict,
#         noise_std_arr = noise_std_arr,
#         X_test = X_test,
#         y_test = y_test,
#         save_metrics = True,
#     )

def main():
    # for testing purposes
    var_loss_models = glob.glob(os.path.join(MODEL_PATH, "var_loss_snr_sweep_iris_2026-05-26_alpha_0.0100_noise*")) # start with the lowest regularizer alpha=1e-4
    print(var_loss_models)

    # load a dummy model for configs
    dummy_payload = pickle.load(open(var_loss_models[0], "rb"))
    seed = dummy_payload['configs']['seed']
    print(f"Configs: \n {dummy_payload['configs']}")
    print(f"Data: \n {dummy_payload['data']}")

    # load the data as it was during training to preserve inference on the test set only
    _, X_test, _, y_test = load_uci_iris(normalize=True, key=seed, train_test_split=0.7)

    # setting up sweep for noise
    noise_std_arr = jnp.logspace(-5, 5, 50)

    # set number of resamples
    num_resamples = 100

    # store the metrics in a dictionary
    metrics_dict = defaultdict(list)

    # get the date
    today = date.today().isoformat()
    filename = f"var_loss_uci_iris_alpha{dummy_payload['data']['alpha']:.4f}_{today}.pkl"

    # # generate the data
    generate_data(
        filename=filename,
        var_loss_models=var_loss_models,
        model_preact_data = None, # not needed for var loss models since we compute the preactivations on the fly
        num_resamples = num_resamples,
        metrics_dict = metrics_dict,
        noise_std_arr = noise_std_arr,
        X_test = X_test,
        y_test = y_test,
        save_metrics = True,
    )

    # for testing load the saved data
    loaded_data = pickle.load(open(os.path.join(DATA_PATH, filename), "rb"))
    loaded_data = pd.DataFrame(loaded_data)
    print(loaded_data.head())


if __name__ == "__main__":
    main()
