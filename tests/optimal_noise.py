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

# Path for loading the models: UPDATE BASED ON MACHINE
MODEL_PATH = "/local_disk/vikrant/trident/models"
FIGURES_PATH = "../plots/"
DATA_PATH = "/local_disk/vikrant/trident/logs"

# --------------------------------------------- 
# Auxilary functions
# ---------------------------------------------
def save_results(results, filename, DATA_PATH = "/local_disk/vikrant/trident/logs"):

    # save results in a pickle file
    filename = os.path.join(DATA_PATH, filename)

    # check if the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as f:
        pickle.dump(results, f)

    print(f"Results saved to {filename}")


def plot_results(results, **kwargs):
    fig, ax = plt.subplots(figsize=(5, 5))
    results = pd.DataFrame(results)
    ax.set_xscale("log", base=10)
    sns.lineplot(data=results, x='sigma', y='mi', hue='limit')
    ax.set_xlabel(r"$\sigma$")
    ax.set_ylabel(r"Mutual Information (nats)")
    ax.legend(frameon=False, title=r"Dynamic Range $\tilde{y}$")
    plt.savefig("../plots/sigma_mi.png", dpi=300, bbox_inches="tight")


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

def compute_H_A_given_y(
        P_A_given_y: jax.Array, # shape (3, N)
        density_y: jax.Array, # shape (N,) p(y)
        grid: jax.Array, # shape (N,) grid
        eps: float = 1e-10
        
    ):

    # sum over activations at each grid point || (3, N) -> (N,)
    p = -jnp.sum(P_A_given_y * jnp.log(P_A_given_y + eps), axis=0)

    # integrate over the grid using density
    H = jnp.trapezoid(p*density_y, grid)

    return H

def compute_H_y_given_A(
        P_A_given_y: jax.Array, # shape (3, N)
        P_A: jax.Array, # shape (3,) p(A)
        density_y: jax.Array, # shape (N,) p(y)
        grid: jax.Array, # shape (N,) grid
        eps: float = 1e-10
    ):

    # construct the posterior
    P_y_given_A = jnp.einsum("ay, y -> ay", P_A_given_y, density_y) / P_A[:, None]

    # compute the conditional entropy 
    per_activation = -jnp.trapezoid(P_y_given_A * jnp.log(P_y_given_A + eps), grid, axis=1)
    H = jnp.sum(P_A * per_activation)
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
def proba_A_given_y(
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
def proba_y_uniform(
        y: float,
        l1: float, #lower limit
        l2: float, # upper limit
    ):

    density = jnp.where(
        (y > l1) & (y <= l2),
        1 / (l2 - l1),
        0.0
    )

    proba_cdf = jnp.where(
        y <= l1,
        0.0,
        jnp.where(
            y > l2,
            1.0,
            (y - l1)/(l2 - l1)
        )
    )

    return proba_cdf, density # temporarily adding density for checking

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
            jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(preactivations, activation[0], std, threshold),
            jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(preactivations, activation[1], std, threshold),
            jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(preactivations, activation[2], std, threshold)
        ]
    )

    # integrate over preactivations
    P_y, density_y = proba_y_uniform(preactivations, l1, l2)
    P_A = jnp.trapezoid(P_A_given_y * density_y, preactivations, axis=1)

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

def test_conditional_entropy():
    l1, l2 = -2, 2
    std = 0.1
    activation_vals = jnp.array([-1, 0, 1])
    grid = jnp.linspace(l1, l2, 5000)  # stay within support only

    # Density (uniform)
    _, density_y = jax.vmap(proba_y_uniform, in_axes=(0, None, None))(grid, l1, l2)

    # P(A=a | ỹ): shape (3, N)
    P_A_given_y = jnp.stack([
        jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(grid, a, std, 0.0)
        for a in [-1, 0, 1]
    ])

    # P(A=a): shape (3,) — marginalize over ỹ
    P_A = jnp.trapezoid(P_A_given_y * density_y[None, :], grid, axis=1)

    # --- Sanity checks ---
    # 1. P(A=a|ỹ) sums to 1 at every grid point
    row_sums = jnp.sum(P_A_given_y, axis=0)
    print(f"P(A|ỹ) row sums — min: {row_sums.min():.4f}, max: {row_sums.max():.4f}  (expect all 1.0)")

    ## also check if analytical H(y) is same as computed one
    H_y_computed = compute_marginal_entropy(density_y, grid)
    print(f"Analytical H(y) = {jnp.log(l2 - l1):.4f}, Computed H(y) = {H_y_computed:.4f}")

    # 2. P(A=a) sums to 1
    print(f"P(A) = {P_A}, sum = {jnp.sum(P_A):.4f}  (expect 1.0)")

    # 3. Compute both conditional entropies
    H_A_given_y = compute_H_A_given_y(P_A_given_y=P_A_given_y, density_y=density_y, grid=grid)
    H_y_given_A = compute_H_y_given_A(P_A_given_y=P_A_given_y, density_y=density_y, P_A=P_A, grid=grid)
    H_y = jnp.log(l2 - l1)

    print(f"H(A | ỹ)  = {H_A_given_y:.4f}  (expect >= 0, <= log(3) = {jnp.log(3):.4f})")
    print(f"H(ỹ | A)  = {H_y_given_A:.4f}  (expect <= H(ỹ) = {H_y:.4f})")

    # 4. Both decompositions of MI should agree
    MI_v1 = H_y - H_y_given_A
    H_A = -jnp.sum(P_A * jnp.log(P_A + 1e-10))
    MI_v2 = H_A - H_A_given_y
    print(f"MI via H(ỹ) - H(ỹ|A) = {MI_v1:.4f}")
    print(f"MI via H(A) - H(A|ỹ) = {MI_v2:.4f}  (both should match)")
    print(f"MI >= 0: {MI_v1 >= 0} | MI <= H(ỹ): {MI_v1 <= H_y}")

    # 5. Extreme sigma checks
    print("\n--- Sigma sweep sanity ---")
    for s in [1e-5, 0.1, 1e5]:
        P_Agy_s = jnp.stack([
            jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(grid, a, s, 0.0)
            for a in [-1, 0, 1]
        ])
        P_A_s = jnp.trapezoid(P_Agy_s * density_y[None, :], grid, axis=1)
        mi = jnp.log(l2-l1) - compute_H_y_given_A(P_A_given_y=P_Agy_s, density_y=density_y, P_A=P_A_s, grid=grid)
        print(f"  sigma={s:.3f} -> MI={mi:.4f}")

# ----------------------------------- 
# Sweeping Over sigma 
# -----------------------------------
def parameter_sweeps():

    # define the range for sigma sweep
    sigmas = jnp.logspace(-4, 4, 100)

    # define threshold
    threshold = 0.0

    # end points of uniform distribution
    # limits_uniform = jnp.arange(0.1, 1.5, 0.2)
    limits_uniform = jnp.array([1e-2, 0.25, 0.5, 0.75, 1.0])

    print(f"Sigma sweep range: {sigmas}")
    print(f"Uniform limits: {limits_uniform}")

    # activations
    activation_vals = jnp.array([-1., 0., 1.])

    # grid
    grid = jnp.linspace(-2.0, 2.0, 50001)

    # storing the results
    results = defaultdict(list)

    # looping over sigma and limits of the uniform distribution
    for s_idx, s in enumerate(sigmas):
        for l_idx, l in enumerate(limits_uniform):
            [l1, l2] = [-l, l]

            # uniform distribution of ỹ
            # Density (uniform)
            cdf_y, density_y = jax.vmap(proba_y_uniform, in_axes=(0, None, None))(grid, l1, l2)

            # P(A=a | ỹ): shape (3, N)
            P_A_given_y = jnp.stack([
                    jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(grid, a, s, threshold)
                    for a in activation_vals
                ])
            
            # P(A=a): shape (3,) — marginalize over ỹ
            P_A = jnp.trapezoid(P_A_given_y * density_y[None, :], grid, axis=1)

            # compute marginal entropy of y
            H_y = jnp.log(l2 - l1)

            # compute marginal entropy (H(y | A))
            H_y_given_A = compute_H_y_given_A(P_A_given_y=P_A_given_y, density_y=density_y, P_A=P_A, grid=grid)

            # compute MI
            mi = H_y - H_y_given_A

            # append results
            results['sigma'].append(s.item())
            results['limit'].append(l.item())
            results['mi'].append(mi.item())
            results['H_y'].append(H_y.item())
            results['H_y_given_A'].append(H_y_given_A.item())

            # print out results: sigma, mi
            print(f"STD = {s:.3f}, Limit = {l:.3f} -> MI = {mi:.4f} (nats)")

    return results





def main():
    test_proba_A_given_preact = False
    if test_proba_A_given_preact:
        preact = jnp.linspace(-1, 1, 51)
        std = 0.1
        proba_minus1 = jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(preact, -1, std, 0.0)
        proba_plus1 = jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(preact, 1, std, 0.0)
        proba_zero = jax.vmap(proba_A_given_y, in_axes=(0, None, None, None))(preact, 0, std, 0.0)

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
        proba_uniform, density = jax.vmap(proba_y_uniform, in_axes=(0, None, None))(preact, l1, l2)
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

    test_mi = False
    if test_mi:
        l1 = -0.3
        l2 = 0.3
        preact = jnp.linspace(l1*2, l2*2, 5000)
        P_y, density_y = jax.vmap(proba_y_uniform, in_axes=(0, None, None))(preact, l1, l2)
        compute_mutual_information(density_y, grid=preact)

    test_conditional_entropy_ = False
    if test_conditional_entropy_:
        test_conditional_entropy()

    run_sweeps = True
    if run_sweeps:
        today = date.today().isoformat()
        results = parameter_sweeps()
        filename = f"optimal_noise_data_{today}.pkl"
        save_results(results, filename)
        plot_results(results)


if __name__ == "__main__":
    main()


