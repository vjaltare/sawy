"""
Visualizing loss landscape of trained neural networks.
- Based on the framework of Li et al. (NeurIPS 2017) -- Filter Normalization
- Train TriDENT model (used one of the presaved models) and visualize the loss landscape and compare with that of a parameter matched sigmoid.

NOTE:
STEPS:
- Load the TriDENT model. Use noise_perf_comp_uci_iris_2026-06-05_noise_0.100.pkl
- Unpack the weights. These serve as learned parameters.
- Draw random samples (d1 and d2) from a guassian distribution of the same shape as the weights (should be (4, 32), (32, 3) for UCI-IRIS).
- Normalize the random directions "filter-wise" i.e. per-neuron (over a row of the weight matrix).
- Set up a grid (alpha-beta)
- Evalue the loss at perturbations around the learned weights as:   f(\alpha, beta) = L(w* + alpha*d1 + beta*d2)

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



from utils import dual_sample_ternary, load_uci_iris
from models import TernaryStochasticActivation, DualSampleTernary, FFN, DualSampleTernaryExact 

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

today = date.today().isoformat()

# Path for loading the models: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"

## saving teh data
def save_data(
        grid: list,
        loss_surface: jax.Array,
        filename: str,

    ):

    # make sure that the grid and loss surface have the same shape
    # assert grid.shape == loss_surface.shape, "Grid and loss surface must have the same shape"

    filename = os.path.join(DATA_PATH, filename)

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump({
            'grid': grid,
            'loss_surface': loss_surface
        }, f)


# parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Noise-performance comparison on UCI-Iris")

    # number of resamples for each noise level
    # parser.add_argument("--num_resamples", type=int, default=45)

    # whether to save the metrics
    # parser.add_argument("--save_metrics", action="store_true")

    # seed/key
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

# loss function
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

# prediction function
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

# filter-wise normalizing function
def filter_wise_norm(
        learned_weights: jax.Array,
        perturb_direction: jax.Array
        
    ):

    # if scalar, leave as is
    if learned_weights.ndim < 1:
        # print("Scalar")
        return perturb_direction

    # if there is only one dimension, normalize to the same length as the learned bias: this is the bias term
    if learned_weights.ndim == 1:
        weight_norm = jnp.linalg.norm(learned_weights)
        perturb_norm = jnp.linalg.norm(perturb_direction)

        # make sure that there is no division by zero
        perturb_norm = jnp.where(perturb_norm == 0, 1.0, perturb_norm)
        return perturb_direction * (weight_norm / perturb_norm)

    elif learned_weights.ndim == 2:
        # compute the norm of learned weights along neuron axis
        weight_norm = jnp.linalg.norm(learned_weights, axis=0, keepdims=True)

        # compute norm of perturbation direction along neuron axis
        perturb_norm = jnp.linalg.norm(perturb_direction, axis=0, keepdims=True)

        # normalize the perturbation direction
        perturb_norm = jnp.where(perturb_norm == 0, 1.0, perturb_norm)
        perturb_direction = perturb_direction * (weight_norm / perturb_norm)

        return perturb_direction
    
# function to compute the perturbed state

## guard function
def is_array(x):
    return hasattr(x, 'shape') and hasattr(x, 'dtype') and jnp.issubdtype(getattr(x, 'dtype', type(None)), jnp.floating)


def compute_perturbed_state(
        learned_weights: jax.Array,
        d1: jax.Array,
        d2: jax.Array,
        alpha: float,
        beta: float
    ):

    # only add objects that are Arrays 
    if is_array(learned_weights):
        return learned_weights + alpha*d1 + beta*d2
    else:
        return learned_weights

    

# pipeline
def generate_loss_surface(
        raw_data,
        alphas: jax.Array,
        betas: jax.Array,
        data_features: jax.Array,
        data_labels: jax.Array,
        seed: int = 101,
        **kwargs
    ):

    # load the model
    clean_model_state = raw_data['state']
    model_data = raw_data['data']
    model_configs = raw_data['configs']

    # initialize a model 
    rngs = nnx.Rngs(
        params=seed,
        dropout=seed*10,
        activation=seed*100,
        next=seed*200
    )

    model = FFN(
        layers=model_configs['layers'],
        noise_std=model_configs['noise_std'],
        threshold=model_configs['threshold'],
        activation=DualSampleTernary,
        rngs=rngs
    )

    # add the learned state to the model
    graphdef, _ = nnx.split(model)
    model = nnx.merge(graphdef, clean_model_state)


    # draw random directions to perturb
    key1, key2 = jax.random.split(jax.random.key(seed))
    raw_d1 = jax.tree.map(lambda x: jax.random.normal(key1, shape=x.shape), clean_model_state)
    raw_d2 = jax.tree.map(lambda x: jax.random.normal(key2, shape=x.shape), clean_model_state)

    # normalize the raw directions along filters
    d1 = jax.tree.map(lambda w, d: filter_wise_norm(w, d), clean_model_state, raw_d1)
    d2 = jax.tree.map(lambda w, d: filter_wise_norm(w, d), clean_model_state, raw_d2)

    def evaluate_loss_per_gridpoint(
            alpha: float,
            beta: float,
    ): 
        # perturb the state around the minima/learned weights
        perturbed_state = jax.tree.map(lambda w, d1, d2:  compute_perturbed_state(w, d1, d2, alpha, beta), clean_model_state, d1, d2)

        # evaluate the loss that the perturbed state by merging it back with graph structure 
        perturbed_model = nnx.merge(graphdef, perturbed_state)
        loss, _ = loss_fn(perturbed_model, data_features, data_labels)
        return loss
    
    # vmap across alphas (axis 0) and betas (axis 1)
    vmap_alpha = jax.vmap(evaluate_loss_per_gridpoint, in_axes=(None, 0))
    vmap_grid = jax.vmap(vmap_alpha, in_axes=(0, None))

    loss_surface = vmap_grid(alphas, betas)

    return loss_surface





def main():
    args = parse_args()

    # initialize alpha and beta grid
    alphas_ = jnp.linspace(-3.0, 3.0, 101)
    betas_ = jnp.linspace(-3.0, 3.0, 101)

    # load the UCI iris dataset
    _, X_test, _, y_test = load_uci_iris(normalize=True, key=args.seed, train_test_split=0.7)

    raw_data = pickle.load(open(os.path.join(MODEL_PATH, "noise_perf_comp_uci_iris_2026-06-05_noise_0.010.pkl"), "rb"))

    # vmap over the alpha and beta grid to evaluate the loss surface
    loss_surface = generate_loss_surface(
        raw_data=raw_data,
        alphas=alphas_,
        betas=betas_,
        data_features=X_test,
        data_labels=y_test,
        seed=args.seed
    )

    # save the data
    Alpha, Beta = jnp.meshgrid(alphas_, betas_)
    save_data(
        grid=[Alpha, Beta],
        loss_surface=loss_surface,
        filename=f"loss_vis_trident_noise_{raw_data['configs']['noise_std']:.03f}_{today}.pkl"
    )


    # visualize the loss surface: make a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Alpha, Beta, loss_surface, cmap='viridis')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('Loss')
    plt.savefig(f"../plots/tmp_loss_surface_trident_noise_{raw_data['configs']['noise_std']:.03f}.png")
    plt.show()





if __name__ == "__main__":
    main()