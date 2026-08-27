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
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes or output neurons.")
    parser.add_argument("--max_token_size", type=int, default=1024, help="Max length to sweep classes over.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for default rng stream") 
    parser.add_argument("--preactivation_scale", type=float, default=1.0, help="Scale of preactivation noise") #  z/scale before passing it to softmax 

    parser.add_argument("--scale", type=float, default=1.0, help="Noise scale parameter")
    parser.add_argument("--loc", type=float, default=0.0, help="Noise location parameter")


    parser.add_argument("--gaussian_noise", action="store_true", help="whether to use gaussian noise")
    parser.add_argument("--logistic_noise", action="store_true", help="whether to use logistic noise")
    parser.add_argument("--save_results", action="store_true", help="whether to save the results")

    parser.add_argument("--run_softmax_pipeline", action="store_true", help="run softmax pipeline")
    parser.add_argument("--run_jacobian_pipeline", action="store_true", help="run jacobian pipeline")

    return parser.parse_args()

## saving a file
def save_payload(data, configs, filename, **kwargs):
    """
    Save model and parameters to a pickle.
    state: nnx.Model state
    configs: dict, configuration parameters.
    data: dict, optional data from simulations 
    """

    # if data is None:
    #     payload = {
    #         'configs' : configs,
    #         'state': state
    #     }
    # else:
    #     payload = {
    #         'configs' : configs,
    #         'data': data,
    #         'state': state
    #     }

    payload = {
        'configs': configs,
        'data': data
    }

    checkpoint_dir = "/local_disk/vikrant/trident/logs"
    filename_ = os.path.join(checkpoint_dir, filename)

    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    
    print(f"Model saved to {filename_}")


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

def softmax_resamples(z, key, nu, scale, gauss_noise = True, logistic_noise = False):
    """
    z: jax.Array, preactivations to softmax layer, shape (C,)
    key: jax.random.PRNGKey, random key for sampling
    nu: int, number of hardmax (one hot) samples to draw from z
    scale: float, noise standard deviation.



    """
    # print(f"shape of input {z.shape}")
    if gauss_noise:
        noise = jax.random.normal(key, shape=(nu, z.shape[-1]))*scale
    elif logistic_noise:
        noise = jax.random.logistic(key, shape=(nu, z.shape[-1]))*scale
    else:
        raise ValueError("Noise can be gaussian or logitic")
    
    # print(f"shape of noise: {noise.shape}")
    zn = z + noise
    # print(f"shape of noisy preact: {zn.shape}")
    idx_max = jnp.argmax(zn, axis=-1)
    # print(f"id max shape: {idx_max.shape}")
    z_one_hot = jax.nn.one_hot(idx_max, num_classes=z.shape[-1], axis=-1)
    # print(f"One hot shape: {z_one_hot.shape}")
    z_avg = jnp.average(z_one_hot, axis=0)
    # print(f"Z avg shape: {z_avg.shape} \n z_avg: {z_avg}")

    # for sanity check also print softmax of z
    # print(f"softmax(z) = {jax.nn.softmax(z)}")

    return z_avg

def probit_infty(z, key, scale, nu_infty=None, gauss_noise = True, logistic_noise = False):
    """
    Generating probits for a large (infinite) averaging window.
    """

    nu_infty = nu_infty or max(int(5e4), 500*z.shape[-1])

    # print(f"shape of input {z.shape}")
    if gauss_noise:
        noise = jax.random.normal(key, shape=(nu_infty, z.shape[-1]))*scale
    elif logistic_noise:
        noise = jax.random.logistic(key, shape=(nu_infty, z.shape[-1]))*scale
    else:
        raise ValueError("Noise can be gaussian or logitic")

    
    zn = z + noise
    # print(f"shape of noisy preact: {zn.shape}")
    idx_max = jnp.argmax(zn, axis=-1)
    # print(f"id max shape: {idx_max.shape}")
    z_one_hot = jax.nn.one_hot(idx_max, num_classes=z.shape[-1], axis=-1)
    # print(f"One hot shape: {z_one_hot.shape}")
    z_inf = jnp.average(z_one_hot, axis=0)
    # print(f"Z avg shape: {z_avg.shape} \n z_avg: {z_avg}")

    # for sanity check also print softmax of z
    # print(f"softmax(z) = {jax.nn.softmax(z)}")

    return z_inf

    



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
def softmax_pipeline():
    print("**"*50)
    print("SOFTMAX PIPELINE")
    print("**"*50)


    args = parse_args()
    rngs1 = nnx.Rngs(default=args.seed, key=int(args.seed + 1000)) # keep rng streams consistent across softmax and jacobian pipelines
    rngs2 = nnx.Rngs(default=args.seed+2, key=int(args.seed + 2000))
    rngs3 = nnx.Rngs(default=args.seed+3, key=int(args.seed + 3000))

    data = defaultdict(list)

    # pick the noise
    noise_kws = dict(
        gauss_noise = args.gaussian_noise,
        logistic_noise = not args.gaussian_noise
    )

    # extract input arguments
    RESAMPLES = args.num_resamples

    # construct the array of number of classes
    num_classes = jnp.logspace(1, jnp.log2(args.max_token_size), base=2, num=10, dtype=int)

    # integration window 
    int_window_list = jnp.logspace(1, jnp.log2(args.int_window), base=2, num=10, dtype=int)

    for c in tqdm(num_classes, total=len(num_classes)):
        print(f"Progress {c}/{len(num_classes)}")

        for r_idx in range(RESAMPLES):
            print(f"Resample# {r_idx}")

            # draw a preactivation sample
            z = jax.random.normal(key=rngs1.key(), shape=(c,))*args.preactivation_scale

            # softmax comparison: noise scaled
            s = jax.nn.softmax(z/args.scale, axis=-1)

            # compute probits
            q = probit_infty(z=z, key=rngs3.key(), scale=args.scale, **noise_kws)

            for nu in int_window_list:
                print(f"Window: {nu}")
                z_avg = softmax_resamples(z=z, key=rngs2.key(), nu=nu, scale=args.scale, **noise_kws)
                cos_sim = cosine_similarity(a=z_avg, b=s)
                cos_sim_q = cosine_similarity(a=z_avg, b=q)
                cos_sim_q_s = cosine_similarity(a=q, b=s)
                norm_r = norm_ratio(a=z_avg, b=s) #Z/S
                norm_r_q = norm_ratio(a=z_avg, b=q)
                print(f"cosine_sim: {cos_sim.item():.2f}, norm_ratio: {norm_r.item():.2f}")
                print(f"cosine_sim q-z: {cos_sim_q.item():.2f}")

                # append to the data file
                data['cos_sim_z_s'].append(cos_sim.item())
                data['cos_sim_z_q'].append(cos_sim_q.item())
                data['cos_sim_q_s'].append(cos_sim_q_s.item())
                data['norm_ratio_z_s'].append(norm_r.item())
                data['norm_ratio_z_q'].append(norm_r_q.item())
                data['num_classes'].append(c.item())
                data['int_window'].append(nu.item())
                data['resample'].append(r_idx)


    if args.save_results:
        configs = {
            'resamples': args.num_resamples,
            'num_classes': num_classes.tolist(),
            'preact_scale': args.preactivation_scale,
            'noise_scale': args.scale,
        }

        filename = f"softmax_mc_analysis_gauss_{today}.pkl"
        save_payload(data=data, configs=configs, filename=filename)

        print(data)


def soft_jacobian_pipeline():
    """
    Pipeline for analyzing the estimated and true softmax jacobian.
    """
    print("**"*50)
    print("SOFTMAX-JACOBIAN PIPELINE")
    print("**"*50)

    args = parse_args()
    rngs1 = nnx.Rngs(default=args.seed, key=int(args.seed + 1000)) # keep rng streams consistent across softmax and jacobian pipelines
    rngs2 = nnx.Rngs(default=args.seed+2, key=int(args.seed + 2000))
    rngs3 = nnx.Rngs(default=args.seed+3, key=int(args.seed + 3000))

    data = defaultdict(list)

    # pick the noise
    noise_kws = dict(
        gauss_noise = args.gaussian_noise,
        logistic_noise = not args.gaussian_noise
    )

    # extract input arguments
    RESAMPLES = args.num_resamples

    # construct the array of number of classes
    num_classes = jnp.logspace(1, jnp.log2(args.max_token_size), base=2, num=10, dtype=int)

    # integration window 
    int_window_list = jnp.logspace(1, jnp.log2(args.int_window), base=2, num=10, dtype=int)

    # scale factor for jacobian estimate
    scale_factor_list = jnp.arange(0.5, 3, 0.5)

    for c_idx, c in tqdm(enumerate(num_classes), total=len(num_classes)):
        print(f"Progress {c_idx}/{len(num_classes)}")

        for r_idx in range(RESAMPLES):
            print(f"Resample# {r_idx}")

            # draw a preactivation sample
            z = jax.random.normal(key=rngs1.key(), shape=(c,))*args.preactivation_scale

            # compute probits
            q = probit_infty(z=z, key=rngs3.key(), scale=args.scale, **noise_kws)

            # softmax comparison: noise scaled
            s = jax.nn.softmax(z/args.scale, axis=-1)

            # compute softmax jacobian
            jacobian_exact = softmax_jacobian(s)
            jacobian_exact_flat = jacobian_exact.reshape(-1,)

            for nu in int_window_list:
                print(f"Window: {nu}")
                sf_theory = nu/(nu - 1)

                # comparing z_avg with the same key across scale factors
                z_avg = softmax_resamples(z=z, key=rngs2.key(), nu=nu, scale=args.scale, **noise_kws)

                for scale_idx, sf in enumerate(scale_factor_list):
                    print(f"Scale progress: {scale_idx}/{len(scale_factor_list)}")

                    # compute the estimated jacobian
                    jacobian_est = jacobian_estimate(s=z_avg, scale_factor=sf)

                    # flatten the estimated jacobian
                    jacobian_est = jacobian_est.reshape(-1,)

                    # compute estimated jacobian for probits
                    jacobian_q_est = jacobian_estimate(s=q, scale_factor=sf)
                    jacobian_q_est = jacobian_q_est.reshape(-1,)

                    # compute cosine sim
                    cos_sim = cosine_similarity(a=jacobian_est, b=jacobian_exact_flat)

                    # compute cosine sim against probits etimate
                    cos_sim_q = cosine_similarity(a=jacobian_q_est, b=jacobian_exact_flat)

                    # compute cosine sim between estimate and probits
                    cos_sim_z_q = cosine_similarity(a=jacobian_est, b=jacobian_q_est)

                    # compute norm ratio
                    norm_r = norm_ratio(a=jacobian_est, b=jacobian_exact_flat) #J_hat/J

                    norm_r_q = norm_ratio(a=jacobian_est, b=jacobian_q_est)

                    print(f"cosine_sim: {cos_sim.item():.2f}, norm_ratio: {norm_r.item():.2f}")
                    print(f"cosine_sim q-z: {cos_sim_z_q.item():.2f}")

                    # append to the data file
                    data['cos_sim_z_s'].append(cos_sim.item())
                    data['cos_sim_q_s'].append(cos_sim_q.item())
                    data['cos_sim_z_q'].append(cos_sim_z_q.item())
                    data['norm_ratio_z_s'].append(norm_r.item())
                    data['norm_ratio_z_q'].append(norm_r_q.item())
                    data['num_classes'].append(c.item())
                    data['int_window'].append(nu.item())
                    data['resample'].append(r_idx)
                    data['scale_factor'].append(sf.item())
                    data['sf_theory'].append(sf_theory.item())


    if args.save_results:
        configs = {
                    'resamples': args.num_resamples,
                    'num_classes': num_classes.tolist(),
                    'preact_scale': args.preactivation_scale,
                    'noise_scale': args.scale,
                }

        filename = f"softmax_jacobian_mc_analysis_gauss_{today}.pkl"
        save_payload(data=data, configs=configs, filename=filename)

        print(data)





## Testing
def main():
    args = parse_args()

    jacobian_pipeline_test = args.run_jacobian_pipeline
    if jacobian_pipeline_test:
        soft_jacobian_pipeline()

    softmax_pipeline_test = args.run_softmax_pipeline
    if softmax_pipeline_test:
        softmax_pipeline()



    averaging_test = False
    if averaging_test:
        rngs = nnx.Rngs(default=0, key=345)
        x = jax.random.normal(rngs.key(), shape=(10,))*0.1
        z = softmax_resamples(z=x, key=rngs.key(), nu=5, scale=0.1)


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