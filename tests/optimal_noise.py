"""
Optimal noise standard deviation that maximizes mutual information between preactivations and ternary representations.

NOTE:

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
from models import TernaryStochasticActivation, DualSampleTernary

import tensorflow_datasets as tfds  # TFDS to download CIFAR-10.
import tensorflow as tf  # TensorFlow / `tf.data` operations.
tf.config.set_visible_devices([], 'GPU')

today = date.today().isoformat()

# Path for loading the models: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"


# --------------------------------------------- 
# Computing entropy
# ---------------------------------------------
def compute_marginal_entropy(
        p: jax.Array, # probability distribution,
        grid: jax.Array, # corresponding grid
        eps: float = 1e-10 # avoiding log(0)  
    ):

    H = -jnp.trapezoid(p*jnp.log(p + eps), grid)
    return H

def compute_conditional_entropy(
        p_A_given_B: jax.Array,
        p_B: jax.Array,
        grid: jax.Array,
        eps: float = 1e-10 # avoiding log(0)  
    ):

    # H = jnp.average(-p_A_given_B * jnp.log(p_A_given_B + eps) * p_B)
    H = -jnp.trapezoid(p_A_given_B * jnp.log(p_A_given_B + eps) * p_B, grid)

    return H


# ----------------------------------- 
# P(A = ai | preactivation)
# -----------------------------------
def proba_A_given_preact(
        preactivation: float,
        activation: int,
        std: float,
        threshold: float = 0.0
    ):

    # Computing for ai = -1
    if activation == -1:
        p_low = jnp.square(jax.scipy.stats.norm.cdf(x=threshold-preactivation, loc=0.0, scale=std))
        return p_low

    # computing for ai = 1
    elif activation == 1:
        p_high = jnp.square(1 - jax.scipy.stats.norm.cdf(x=threshold-preactivation, loc=0.0, scale=std))
        return p_high

    # computing for ai = 0
    elif activation == 0:
        p_zero = 1 - jnp.square(1 - jax.scipy.stats.norm.cdf(x=threshold-preactivation, loc=0.0, scale=std)) - jnp.square(jax.scipy.stats.norm.cdf(x=threshold-preactivation, loc=0.0, scale=std))
        return p_zero

    else:
        raise ValueError(f"Invalid activation value: {activation}, should be {-1, 0, 1}")


# ----------------------------------- 
# preactivation y ~ U(l1, l2)
# -----------------------------------
def proba_preact_uniform(
        y: float,
        l1: float, #lower limit
        l2: float, # upper limit
    ):

    density = jnp.where(
        (y > l1) & (y <= l2),
        1 / (l2 - l1),
        0.0
    )

    proba = jnp.where(
        y <= l1,
        0.0,
        jnp.where(
            y > l2,
            1.0,
            (y - l1)/(l2 - l1)
        )
    )

    return proba, density # temporarily adding density for checking

# --------------------------------------------- 
# P(A = ai) | Marginalizing over preactivations
# ---------------------------------------------
def proba_A(
        activation: jax.Array, # -1, 0, 1
        l1: float,
        l2: float,
        dx: float, # step size
        std: float,
        threshold: float = 0.0
    ):

    # define preactivations grid
    preactivations = jnp.arange(l1, l2, dx)

    # compute P(A | y) for all activations (shape should be (N, 3) with N being the )
    P_A_given_y = jnp.stack(
        [
            jax.vmap(proba_A_given_preact, in_axes=(0, None, None, None))(preactivations, activation[0], std, threshold),
            jax.vmap(proba_A_given_preact, in_axes=(0, None, None, None))(preactivations, activation[1], std, threshold),
            jax.vmap(proba_A_given_preact, in_axes=(0, None, None, None))(preactivations, activation[2], std, threshold)
        ]
    )

    # integrate over preactivations
    P_y, _ = proba_preact_uniform(preactivations, l1, l2)
    P_A = jnp.trapezoid(P_A_given_y * P_y, preactivations, axis=1)

    return P_A


# --------------------------------------------- 
# MI calculations
# ---------------------------------------------
def compute_mutual_information(
        P_y: jax.Array, # Marginal distribution of preactivations
        P_A: jax.Array = None, # Marginal distribution on activations
        P_A_given_y: jax.Array = None, # Conditional distribution of activations given preactivations
        grid: jax.Array = None, # corresponding grid for P_y
        l2: float = 0.3,
        l1: float = -0.3,

    ):

    # compute the entropy of marginal preactivation distribution: should be log(l2- l1) for uniform distribution
    H_y = compute_marginal_entropy(P_y, grid)

    print(H_y, jnp.log(l2 - l1))




# ----------------------------------- 
# Testing Arena
# -----------------------------------
def main():
    test_proba_A_given_preact = False
    if test_proba_A_given_preact:
        preact = jnp.linspace(-1, 1, 51)
        std = 0.1
        proba_minus1 = jax.vmap(proba_A_given_preact, in_axes=(0, None, None, None))(preact, -1, std, 0.0)
        proba_plus1 = jax.vmap(proba_A_given_preact, in_axes=(0, None, None, None))(preact, 1, std, 0.0)
        proba_zero = jax.vmap(proba_A_given_preact, in_axes=(0, None, None, None))(preact, 0, std, 0.0)

        plt.plot(preact, proba_minus1, label=r"$P(A = -1 \mid \tilde{y})$")
        plt.plot(preact, proba_plus1, label=r"$P(A = 1 \mid \tilde{y})$")
        plt.plot(preact, proba_zero, label=r"$P(A = 0 \mid \tilde{y})$")
        plt.plot(preact, proba_minus1 + proba_plus1 + proba_zero, label=r"$P(A = -1 \mid \tilde{y}) + P(A = 0 \mid \tilde{y}) + P(A = 1 \mid \tilde{y})$", c='k', ls="--")
        plt.xlabel("Preactivation")
        plt.ylabel(r"$P(A = a_i \mid \tilde{y})$")
        plt.title("Probability of Activation given Preactivation")
        plt.legend()
        plt.savefig("../plots/opt_noise_temp.png")

    test_proba_preact_uniform = False
    if test_proba_preact_uniform:
        l1 = -0.3
        l2 = 0.3
        preact = jnp.linspace(l1*2, l2*2, 51)
        proba_uniform, density = jax.vmap(proba_preact_uniform, in_axes=(0, None, None))(preact, l1, l2)
        plt.plot(preact, proba_uniform * jnp.ones_like(preact), label=r"$P(\tilde{y})$")
        plt.plot(preact, density, label=r"Density of $\tilde{y}$")
        plt.xlabel("Preactivation")
        plt.ylabel(r"$P(\tilde{y})$")
        plt.title("Probability of Preactivation (Uniform)")
        plt.legend()
        plt.savefig("../plots/opt_noise_uniform_temp.png")

    test_proba_A = False
    if test_proba_A:
        l1 = -0.3
        l2 = 0.3
        dx = 0.0001
        activation = jnp.array([-1, 0, 1])
        std = 0.1
        threshold = 0.0
        P_A = proba_A(activation, l1, l2, dx, std, threshold)
        print(P_A.shape)
        print(P_A, jnp.sum(P_A))

    test_mi = True
    if test_mi:
        l1 = -0.3
        l2 = 0.3
        preact = jnp.linspace(l1*2, l2*2, 5000)
        P_y, density_y = jax.vmap(proba_preact_uniform, in_axes=(0, None, None))(preact, l1, l2)
        compute_mutual_information(density_y, grid=preact)

if __name__ == "__main__":
    main()


