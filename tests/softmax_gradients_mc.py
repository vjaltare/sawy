"""
Testing gradients of dual_sample_binary_softmax and dual_sample_ce_loss.


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



from utils import dual_sample_ternary, load_uci_iris, dual_sample_binary_softmax, generate_gaussian_noise, generate_logistic_noise
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
    parser = argparse.ArgumentParser(description="Softmax monte carlo test")

    parser.add_argument("--num_resamples", type=int, default=50, help="Number of resamples for each test")
    parser.add_argument("--int_window", type=int, default=2, help="Number of samples to average over. Need at least 2")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for default rng stream") 
    parser.add_argument("--preactivation_scale", type=float, default=0.1, help="Scale of preactivation noise")

    parser.add_argument("--scale", type=float, default=0.1, help="Noise scale parameter")
    parser.add_argument("--loc", type=float, default=0.0, help="Noise location parameter")


    parser.add_argument("--gaussian_noise", action="store_true", help="whether to use gaussian noise")
    parser.add_argument("--logistic_noise", action="store_true", help="whether to use logistic noise")
    parser.add_argument("--save_results", action="store_true", help="whether to save the results")

    return parser.parse_args()


# define jacobian for softmax
def softmax_jacobian(s: jax.Array):
    """
    Compute the jacobian of the softmax function.
    Args:
        s: softmax output, shape (C,)
    Returns:
        jacobian: shape (C, C)
    """
    diag = jnp.diag(s)
    cov = jnp.einsum("i, j -> ij", s, s)
    jacobian = diag - cov
    return jacobian

# ----------------------------------------
# Cosine similarity function and norm ratio function
# ----------------------------------------
def cosine_similarity(a: jax.Array, b: jax.Array):
    """
    Compute the cosine similarity between two vectors.
    Args:
        a: vector a, shape (C,)
        b: vector b, shape (C,)
    Returns:
        cosine similarity: scalar
    """
    dot_product = jnp.dot(a, b)
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)
    return dot_product / (norm_a * norm_b + 1e-8)  # add small epsilon to avoid division by zero

def norm_ratio(a: jax.Array, b: jax.Array):
    """
    Compute the ratio of norms between two vectors.
    Args:
        a: vector a, shape (C,)
        b: vector b, shape (C,)
    Returns:
        norm ratio: scalar
    """
    norm_a = jnp.linalg.norm(a)
    norm_b = jnp.linalg.norm(b)
    return norm_a / (norm_b + 1e-8)  # add small epsilon to avoid division by zero

def resamples(z, key, nu, scale):
    """
    z: jax.Array, preactivations to softmax layer, shape (C,)
    key: jax.random.PRNGKey, random key for sampling
    nu: int, number of hardmax (one hot) samples to draw from z
    scale: float, noise standard deviation.



    """
    print(f"shape of input {z.shape}")
    noise = jax.random.normal(key, shape=(nu, z.shape[-1]))*scale
    print(f"shape of noise: {noise.shape}")
    zn = z + noise
    print(f"shape of noisy preact: {zn.shape}")
    idx_max = jnp.argmax(zn, axis=-1)
    print(f"id max shape: {idx_max.shape}")
    z_one_hot = jax.nn.one_hot(idx_max, num_classes=z.shape[-1], axis=-1)
    print(f"One hot shape: {z_one_hot.shape}")
    z_avg = jnp.average(z_one_hot, axis=0)
    print(f"Z avg shape: {z_avg.shape} \n z_avg: {z_avg}")

    # for sanity check also print softmax of z
    print(f"softmax(z) = {jax.nn.softmax(z)}")

    return z_avg


# ----------------------------------------
# Jacobian for softmax approximation
# ----------------------------------------
def jacobian_estimate(
        s: jax.Array, # dual sample softmax {-1, 0, 1}
        scale_factor: float = 2.0,
        
    ):

    # eusure that the input is a 1D array
    # assert s.ndim == 1, "Input must be a 1D array"

    # construct the jacobian
    diag = jnp.einsum("...i, ij -> ...ij", s, jnp.eye(s.shape[-1], dtype=s.dtype))
    cov = jnp.einsum("...i, ...j -> ...ij", s, s)

    jacobian = scale_factor * (diag - cov) # Refer to theory for factor 2

    return jacobian

# ----------------------------------------
# M/C Pipeline
# ----------------------------------------
def pipeline():
    args = parse_args()
    rngs1 = nnx.Rngs(default=args.seed, key=int(args.seed + 1000))
    rngs2 = nnx.Rngs(default=args.seed+2, key=int(args.seed + 2000))

    # extract input arguments
    RESAMPLES = args.num_resamples

    for r_idx in range(RESAMPLES):

        # draw a preactivation sample
        z = jax.random.normal(key=rngs1.key(), shape=args.int_window)*args.preactivation_scale

        # softmax comparison: TODO
        s = jax.nn.softmax(z, axis=-1)







## Testing
def main():
    args = parse_args()

    averaging_test = True
    if averaging_test:
        rngs = nnx.Rngs(default=0, key=345)
        x = jax.random.normal(rngs.key(), shape=(10,))*0.1
        z = resamples(z=x, key=rngs.key(), nu=5, scale=0.1)


    jacobian_test = False
    if jacobian_test:
        rngs = nnx.Rngs(default=0, key=345)
        x = jax.random.normal(rngs.key(), shape=(10,))*0.1
        n1 = generate_gaussian_noise(shape=x.shape, rngs=rngs, std=args.scale, mean=args.loc)
        n2 = generate_gaussian_noise(shape=x.shape, rngs=rngs, std=args.scale, mean=args.loc)
        s_hat = dual_sample_binary_softmax(x=x, n1=n1, n2=n2)   
        s = jax.nn.softmax(x, axis=-1)
        jacob = softmax_jacobian(s)
        jacob_est = jacobian_estimate(s_hat, scale_factor=2.0)
        print(f"Input: {x}, {jnp.argmax(x)}")
        print("Softmax output:", s)
        print(f"Softmax jac: {jacob}")
        print(f"Softmax jac estimate: {jacob_est}")

        # plotting
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(jacob, cmap='viridis', interpolation='nearest')
        axs[1].imshow(jacob_est, cmap='viridis', interpolation='nearest')

        plt.savefig("../plots/softmax_jac_comp.png")

if __name__ == "__main__":
    main()