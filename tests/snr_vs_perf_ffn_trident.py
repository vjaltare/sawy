"""
Characterizing the performance of TriDENT trained on varying levels of noise.
- Train the network on a given SNR level.
- Perform inference on a range of SNR levels, and plot the performance curve.
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



from utils import ternary_activation, load_cifar10_augment, dual_sample_ternary, load_uci_iris
from models import TernaryStochasticActivation, DualSampleTernary, FFN

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

today = date.today().isoformat()

# Path for loading the models: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"

# -----------------------------
# Auxiliary functions
# ----------------------------- 
def save_payload(state, configs, filename, data = None):
    """
    Save model and parameters to a pickle.
    state: nnx.Model state
    configs: dict, configuration parameters.
    data: dict, optional data from simulations 
    """

    if data is None:
        payload = {
            'configs' : configs,
            'state': state
        }
    else:
        payload = {
            'configs' : configs,
            'data': data,
            'state': state
        }

    checkpoint_dir = "/local_disk/vikrant/trident/models"
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"Model saved to {filename_}")


# --------------------------------------------------------------
# Parsing input arguments
# --------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Trident-FFN on UCI-Iris")

    # take in the levels
    parser.add_argument("--levels", nargs="+", type=float, default=[-1.0, 0.0, 1.0])

    # take in the initial thresholds
    # parser.add_argument("--thresholds", nargs="+", type=float, default=[-1.0, 1.0])
    parser.add_argument("--threshold", type=float, default=0.0)

    # take in the noise standard deviation
    # parser.add_argument("--noise_std", type=float, default=1.0)

    # model architecture parameters
    # parser.add_argument("--ff_layer_sizes", nargs="+", type=int, default=[1000, 500])

    # hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_steps", type=int, default=int(5e4))
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--seed_next", type=int, default=0)
    parser.add_argument("--checkpoint", action='store_true')
    # parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)


    #TODO: Add arguments for saving results and the model.
    parser.add_argument("--plot_results", action='store_true')



    return parser.parse_args()

# ------------------------------------
# Training functions
# ------------------------------------
def loss_fn(
        model: FFN,
        # batch: dict,
        X: jax.Array,
        labels: jax.Array,
        loss_function: Callable = optax.softmax_cross_entropy_with_integer_labels
    ):
    # forwad pass through the model
    logits = model(X)
    # using softmax cross-entropy with integer labels
    loss = loss_function(logits, labels=labels).mean()

    return loss, logits

        
# training step
@nnx.jit
def train_step(
    model: FFN,
    optimizer: nnx.Optimizer,
    metrics: nnx.MultiMetric,
    X: jax.Array, # input data
    label: jax.Array, # labels
    loss_fn: Callable = loss_fn
    ):

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, X, label)
    metrics.update(loss=loss, logits=logits, labels=label)
    optimizer.update(model, grads)

# evaluation step
@nnx.jit
def eval_step(
    model: FFN,
    metrics: nnx.MultiMetric,
    X: jax.Array, # input data
    label: jax.Array, # labels
    loss_fn: Callable = loss_fn
 ):
    loss, logits = loss_fn(model, X, label)
    metrics.update(loss=loss, logits=logits, labels=label)

# -------------------------------------
# Custom Linear Function
# -------------------------------------
class CustomLinear(nnx.Module):

    def __init__(self,
                 rngs: nnx.Rngs = nnx.Rngs(0),
                 threshold: float = 0.0,
                 noise_std: float = 1.0,
                 noise_mean: float = 0.0,
                 ):
        self.rngs = rngs
        self.threshold = threshold
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        
    def __call__(self, x):
        return x

# ------------------------------------
# Training Pipeline
# ------------------------------------
def train(
        train_inputs: jax.Array,
        train_labels: jax.Array,
        test_inputs: jax.Array,
        test_labels: jax.Array,
        configs: dict,
        checkpoint_flag: bool = False,
        **kwargs    
    ):

    """
    Training loop to sweep over input noise levels.
    Checkpoint models at the end of training for every noise level.
    """
    
    print("--"*50)
    print(f"Noise-sweep: UCI Iris Dataset")
    print("--"*50)

    eval_every = configs['eval_every']
    train_steps = configs['train_steps']
    noise_std_arr = configs['noise_std_arr'] 

    for n_idx, noise_std in enumerate(tqdm(noise_std_arr)):
        data = defaultdict(list) # dictionary to store training data for current noise level

        # DEFINE THE MODELS
        rngs = nnx.Rngs(
            params=configs['seed'] + 0,
            dropout=configs['seed'] + 1,
            activation=configs['seed'] + 2,
        )

        model = FFN(
            layers = configs['layers'],
            noise_std = noise_std,
            threshold = configs['threshold'],
            ActivationFunction = DualSampleTernary,
            rngs = rngs
        )

        linear_model = FFN(
            layers = configs['layers'],
            noise_std = noise_std,
            threshold = configs['threshold'],
            ActivationFunction = CustomLinear,
            rngs = rngs
        )

        optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=configs['learning_rate'],
                weight_decay=1e-4
            )
        ),
        wrt=nnx.Param
        )

        metrics = nnx.MultiMetric(
            accuracy=nnx.metrics.Accuracy(),
            loss=nnx.metrics.Average('loss')
        )

        metrics_history = defaultdict(list)

        for step in range(train_steps):

            # train the model
            train_step(model, optimizer, metrics, train_inputs, train_labels)

            # evaluate and checkpoint the model
            if step > 0 and (step%eval_every==0 or step == train_steps-1):
                metrics_history['step'].append(step)

                # log the training metrics
                for metric, value in metrics.compute().items():
                    metrics_history[f"train_{metric}"].append(value.item())
                metrics.reset()

                # evaluate the model on validation set
                eval_step(model, metrics, test_inputs, test_labels)

                # log the evaluation metrics
                for metric, value in metrics.compute().items():
                    metrics_history[f"eval_{metric}"].append(value.item())
                metrics.reset()
            
                print(f"Step {step}/{train_steps} | Train Accuracy: {metrics_history['train_accuracy'][-1]:.4f} | Eval Accuracy: {metrics_history['eval_accuracy'][-1]:.4f}")
        
        best_accuracy = max(metrics_history['eval_accuracy'])
        best_acc_idx = jnp.argmax(jnp.array(metrics_history['eval_accuracy'])).astype(int)
        print(f"Best Eval Accuracy: {best_accuracy:.4f} | Step: {metrics_history['step'][best_acc_idx]}")

        ## TODO: determine checkpointing
        if checkpoint_flag:
            data['noise_std'] = noise_std
            
            ## computing SNR for current noise level
            gd_tri, state_trident = nnx.split(model)
            gd_linear, state_linear = nnx.split(linear_model)
            linear_model = nnx.merge(gd_linear, state_trident) # give the linear models weights from the trained trident model

            # compute the SNR using hidden layer pre-activations
            hidden_layer = linear_model.activation(linear_model.layers[0](train_inputs)) # should be a (150, 32) output

            # compute the rms across the dataset
            hidden_layer = hidden_layer.flatten()
            rms_pre_act = jnp.sqrt(jnp.mean(hidden_layer**2))
            snr = 20*jnp.log10(rms_pre_act/noise_std)
            training_sparsity = jnp.mean(hidden_layer == 0)
            print(f"Noise Std: {noise_std:.4f} | SNR: {snr:.2f} dB | Sparsity: {training_sparsity*100:.2f}%")

            # append SNR to data dictionary
            data['training_snr'].append(snr.item())

            # prep the configs
            data['training_sparsity'].append(training_sparsity.item())

            # save the model, data and configs
            filename = f"snr_sweep_iris_{today}_noise_{noise_std:.4f}.pkl"
            save_payload(state_trident, configs, filename, data)

            # break # for testing purposes, break after the first noise level
        os.system('clear')
            

    return model, metrics_history


# ------------------------------------
# Setting up training
# ------------------------------------
def main():

    # parse the input arguments
    args = parse_args()

    # load the data
    X_train, X_test, y_train, y_test = load_uci_iris(normalize=True, key=101, train_test_split=0.7)

    # noise std array
    noise_std_arr = jnp.logspace(-3, 3, 7)

    configs = {
        'train_steps': args.train_steps,
        'eval_every': args.eval_every,
        'learning_rate': args.learning_rate,
        'threshold': args.threshold,
        'noise_std_arr': noise_std_arr,
        'layers': [X_train.shape[1], 32, 3],
        'seed': args.seed
    }


    model, metrics_history = train(
        train_inputs=X_train,
        train_labels=y_train,
        test_inputs=X_test,
        test_labels=y_test,
        configs=configs,
        checkpoint_flag=args.checkpoint,
    )

    # as a test, print out contents of a saved model
    # fname = f"PILOT_snr_sweep_iris_{today}_noise_{noise_std_arr[0].item():.4f}.pkl"
    # with open(os.path.join(MODEL_PATH, fname), "rb") as f:
    #     loaded_state = pickle.load(f)

    # print(loaded_state['data'])
    # print(loaded_state['configs'])
    # print(loaded_state['state'])


if __name__ == "__main__":
    main()