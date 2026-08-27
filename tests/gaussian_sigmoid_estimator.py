"""
Testing the gaussian sigmoid-estimator

- Sample a range of scalar preactivations y = U(-a, a)
- For every sample iterate over multiple averaging windows (nu > 2)
- Compute the averaged values, analytical expected values.
- Also compute analytical (gaussian, logistic) and estimated gradients (cdf(1- cdf)).

For running:
- Manually change the run_pipeline flag between gaussian and logistic noise.
- 
"""

import os
import sys
import argparse
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from utils import generate_gaussian_noise, generate_logistic_noise
from tqdm import tqdm
import pickle

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import date
today = date.today().isoformat()

# parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train a parameter-matched FFN with sigmoidal activation on UCI Dataset for loss visualization.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inputs", type=int, default=100)

    return parser.parse_args()

def save_payload(data, configs, filename, **kwargs):
    payload = {
        'data': data,
        'configs': configs
    }

    checkpoint_dir = "/local_disk/vikrant/trident/logs"
    filename_ = os.path.join(checkpoint_dir, filename)
    
    os.makedirs(os.path.dirname(filename_), exist_ok=True)  # Ensure the directory exists.

    with open(filename_, 'wb') as f:
        pickle.dump(payload, f)
    
    print("**"*50)
    print(f"Payload saved to {filename_}")
    print("**"*50)

# Functions ...
def bipolar_heaviside(x):
    return 2*jnp.heaviside(x, 0) - 1

def sampled_gradient(x):
    """
    x: Bipolar Array. (num_points, nu). Averaging dimension should be the last one. 
    Should be used for all noise distributions.
    """

    assert x.shape[-1] >= 2

    g_samples = jnp.abs(x[:, :-1] - x[:, 1:])/2 # pairwise average
    # print(f"pre avg samples: {g_samples}")
    g_samples = jnp.mean(g_samples, axis=-1)
    # print(f"Sampled gs: {g_samples}")

    # assert nu <= len(g_samples), "averaging window (nu) cannot be greater than size of samples"

    # avg_arr = jnp.ones((nu,))*(1/nu)
    # g_avg = jnp.convolve(g_samples, avg_arr, mode="valid")
    return g_samples

def sigmoid(x, loc, scale):
    return 1/(1 + jnp.exp(-(x - loc)/scale))


def expected_state_gauss(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float,
    ):

    Ey = 2*jax.scipy.stats.norm.cdf(x=x-threshold, loc=mean, scale=std) - 1

    return Ey

# TODO: expected state logistic
def expected_state_logistic(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float
        
    ):

    Ey = 2*sigmoid(x=x-threshold, scale=std, loc=mean) - 1
    return Ey


def exact_gauss_gradient(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float,
        
    ):

    grad_ = 2*jax.scipy.stats.norm.pdf(x=threshold-x, loc=mean, scale=std)

    return grad_

# TODO: exact gradient logistic
def exact_logistic_gradient(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float,
    ):

    grad = 2*sigmoid(x=x-threshold, scale=std, loc=mean)*(1 - sigmoid(x=x-threshold, scale=std, loc=mean))

    return grad


def estimated_gauss_gradient(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float
    ):

    grad_ = 2*(jax.scipy.stats.norm.cdf(x=x-threshold, loc=mean, scale=std)*(1 - jax.scipy.stats.norm.cdf(x=x-threshold, loc=mean, scale=std)))
    return grad_



def int_samples(
        x: jax.Array, #(N, ), bipolar entries needed!
        noise_arr: jax.Array, #(N, nu)
        nu: int, # integration window
    ):

    x_noise = x[:, None] + noise_arr
    x_noise = jnp.mean(x_noise, axis=-1)
    return x_noise


def single_window_gauss(
        rngs: nnx.Rngs,
        num_inputs: int, # no. input points
        mean: float,
        std: float,
        nu: int, # integration window

    ):
    # generate inputs
    # x = jax.random.uniform(rngs.key(), shape=(resamples,), minval=-1.0, maxval=1.0)
    x = jnp.linspace(-5, 5, num_inputs)

    # map x through the expected state: gauss
    exp_states_gauss = expected_state_gauss(x=x, threshold=0.0, std=std, mean=mean)

    # compute exact expected gradient: gauss
    exact_grad_gauss = exact_gauss_gradient(x=x, threshold=0.0, std=std, mean=mean)

    # compute the expected gradient
    exp_grad_gauss = estimated_gauss_gradient(x=x, threshold=0.0, std=std, mean=mean)

    ## integration analysis
    # add noise
    noise_arr = generate_gaussian_noise(shape=x.shape + (int(nu),), rngs=rngs, std=std, mean=mean)
    x_noise = x[:, None] + noise_arr
    x_noise = bipolar_heaviside(x_noise)

    # computed expected state from samples
    calc_exp_state = jnp.mean(x_noise, axis=-1)  # average along the nu-axis

    # computed gradient from samples
    calc_grad = sampled_gradient(x_noise)

    # compose all the results into a dictionary
    data = defaultdict(list)
    
    data['input'] = x.tolist()
    data['exp_state_gauss'] = exp_states_gauss.tolist()
    data['exact_grad_gauss'] = exact_grad_gauss.tolist()
    data['exp_grad_gauss']= exp_grad_gauss.tolist()
    data['computed_exp_state'] = calc_exp_state.tolist()
    data['computed_grad'] = calc_grad.tolist()

    return data

def single_window_logistic(
        rngs: nnx.Rngs,
        num_inputs: int, # no. input points
        mean: float,
        std: float,
        nu: int, # integration window

    ):
    # generate inputs
    # x = jax.random.uniform(rngs.key(), shape=(resamples,), minval=-1.0, maxval=1.0)
    x = jnp.linspace(-5, 5, num_inputs)

    # map x through the expected state: gauss
    exp_states_logistic = expected_state_logistic(x=x, threshold=0.0, std=std, mean=mean)

    # exact/estimated gradient for logistic: both are equal!
    exact_grad_logistic = exact_logistic_gradient(x=x, threshold=0.0, std=std, mean=mean)


    ## integration analysis
    # add noise: logistic
    noise_arr = generate_logistic_noise(shape=x.shape + (int(nu),), rngs=rngs, std=std, loc=mean)
    x_noise = x[:, None] + noise_arr
    x_noise = bipolar_heaviside(x_noise)

    # computed expected state from samples
    calc_exp_state = jnp.mean(x_noise, axis=-1)  # average along the nu-axis

    # computed gradient from samples
    calc_grad = sampled_gradient(x_noise)

    # compose all the results into a dictionary
    data = defaultdict(list)
    
    data['input'] = x.tolist()
    data['exp_state_logistic'] = exp_states_logistic.tolist()
    data['exact_grad_logistic'] = exact_grad_logistic.tolist()
    # data['exp_grad_gauss']= exp_grad_gauss.tolist()
    data['computed_exp_state'] = calc_exp_state.tolist()
    data['computed_grad'] = calc_grad.tolist()

    return data

def pipeline_gauss(
        num_inputs: int, # no. input points
        nu_arr: jax.Array,
        std_arr: jax.Array,
        seed: int, # initial seed for keys
    ):

    nu_arr = nu_arr.tolist()
    std_arr = std_arr.tolist()

    key = jax.random.key(seed)
    records = []

    for std_idx, std in tqdm(enumerate(std_arr), total=len(std_arr)):
        print(f"STD: {std}, PROGRESS {std_idx+1}/{len(std_arr)}")
        for nu_idx, nu in tqdm(enumerate(nu_arr), total=len(nu_arr)):
            print(f"NU: {nu}, PROGRESS {nu_idx+1}/{len(nu_arr)}")

            key, subkey = jax.random.split(key)
            rngs = nnx.Rngs(key=key, default=0)
            data = single_window_gauss(rngs=rngs, num_inputs=num_inputs, std=std, mean=0.0, nu=nu)
            data = pd.DataFrame(data)
            data['nu'] = nu
            data['std'] = std
            records.append(data)

    payload = pd.concat(records, ignore_index=True)

    return payload

def pipeline_logistic(
        num_inputs: int, # no. input points
        nu_arr: jax.Array,
        std_arr: jax.Array,
        seed: int, # initial seed for keys
    ):

    nu_arr = nu_arr.tolist()
    std_arr = std_arr.tolist()

    key = jax.random.key(seed)
    records = []

    for std_idx, std in tqdm(enumerate(std_arr), total=len(std_arr)):
        print(f"STD: {std}, PROGRESS {std_idx+1}/{len(std_arr)}")
        for nu_idx, nu in tqdm(enumerate(nu_arr), total=len(nu_arr)):
            print(f"NU: {nu}, PROGRESS {nu_idx+1}/{len(nu_arr)}")

            key, subkey = jax.random.split(key)
            rngs = nnx.Rngs(key=key, default=0)
            data = single_window_logistic(rngs=rngs, num_inputs=num_inputs, std=std, mean=0.0, nu=nu)
            data = pd.DataFrame(data)
            data['nu'] = nu
            data['std'] = std
            records.append(data)

    payload = pd.concat(records, ignore_index=True)

    return payload


# Testing
def main():

    test_activations_ = False
    if test_activations_:
        x = jnp.arange(-2, 2, 0.01)
        y_bi = bipolar_heaviside(x)
        y_exp = expected_state_gauss(x=x, std=0.1, mean=0, threshold=0)
        g_ex = exact_gauss_gradient(x=x, std=0.1, mean=0, threshold=0)
        g_es = estimated_gauss_gradient(x=x, std=0.1, mean=0, threshold=0)

        plt.subplot(121)
        plt.plot(x, y_bi)
        plt.plot(x, y_exp)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.subplot(122)
        plt.plot(x, g_ex)
        plt.plot(x, g_es)
        plt.xlabel("x")
        plt.ylabel("f'(x)")
        plt.savefig("../plots/gauss_sigmoid_test.png")

    test_sampled_gradient = False
    if test_sampled_gradient:
        rngs = nnx.Rngs(key=1, default=0)
        x = jax.random.normal(rngs.key(), (10, 200))
        y_bi = bipolar_heaviside(x)
        print(f"inputs: {x}")
        print(f"Bipolar samples: {y_bi}")
        gs = sampled_gradient(x=y_bi)
        # print(f"Expect gradients to be binary for")
        print(f"Sampled gradient: {gs}")

    run_pipeline_gauss = False
    if run_pipeline_gauss:
        print("**"*50)
        print("GAUSS PIPELINE")
        print("**"*50)

        args = parse_args()

        nu_arr = jnp.logspace(0, 4, 5)*2
        nu_arr = jnp.round(nu_arr)
        print(f"Nu arr: {nu_arr}")
        std_arr = jnp.logspace(-3, 0, 4)
        print(f"std_arr: {std_arr}")

        data = pipeline_gauss(num_inputs=args.num_inputs, nu_arr=nu_arr, std_arr=std_arr, seed=args.seed)
        configs = {"seed": args.seed, "num_inputs": args.num_inputs}
        filename = f"trident_gauss_noise_{today}.pkl"
        save_payload(data=data, configs=configs, filename=filename)
        print(data.head())

        fig, ax = plt.subplots(1, 2)
        df = data.loc[data['std'] == 1.0]
        sns.lineplot(data=df, x="input", y="exp_grad_gauss", hue="nu", palette='colorblind', ax=ax[0], alpha=0.3)
        sns.lineplot(data=df, x="input", y="computed_grad", hue="nu", palette='colorblind', ax=ax[1], alpha=0.2)
        fig.savefig("../plots/sanity_check_gauss_pipeline.png")


    run_pipeline_logistic = True
    if run_pipeline_logistic:
        print("**"*50)
        print("LOGISTIC PIPELINE")
        print("**"*50)

        args = parse_args()

        nu_arr = jnp.logspace(0, 4, 5)*2
        nu_arr = jnp.round(nu_arr)
        print(f"Nu arr: {nu_arr}")
        std_arr = jnp.logspace(-3, 0, 4)
        print(f"std_arr: {std_arr}")

        data = pipeline_logistic(num_inputs=args.num_inputs, nu_arr=nu_arr, std_arr=std_arr, seed=args.seed)
        configs = {"seed": args.seed, "num_inputs": args.num_inputs}
        filename = f"trident_logisitc_noise_{today}.pkl"
        save_payload(data=data, configs=configs, filename=filename)
        print(data.head())

        fig, ax = plt.subplots(1, 2)
        df = data.loc[data['std'] == 1.0]
        sns.lineplot(data=df, x="input", y="exact_grad_logistic", hue="nu", palette='colorblind', ax=ax[0], alpha=0.3)
        sns.lineplot(data=df, x="input", y="computed_grad", hue="nu", palette='colorblind', ax=ax[1], alpha=0.2)
        fig.savefig("../plots/sanity_check_logistic_pipeline.png")




    # test_pipeline_gauss = True
    # if test_pipeline_gauss:
    #     rngs = nnx.Rngs(key=1, default=0)
    #     data = pipeline_gauss(
    #         rngs=rngs,
    #         num_inputs=100,
    #         gauss_kws={'mean': 0.0, 'std': 1.0},
    #         nu=100

    #     )

    #     print(data)

    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].plot(data['input'], data['exp_state_gauss'], label="expected state")
    #     ax[0].plot(data['input'], data['computed_exp_state'], label="computed expected state")
    #     ax[0].set_xlabel("Input (x)")
    #     ax[0].set_ylabel("Activation")
    #     ax[0].legend()

    #     ax[1].plot(data['input'], data['exact_grad_gauss'], label="exact grad")
    #     ax[1].plot(data['input'], data['exp_grad_gauss'], label="expected grad")
    #     ax[1].plot(data['input'], data['computed_grad'], label="computed grad")
    #     ax[1].set_xlabel("Input (x)")
    #     ax[1].set_ylabel("Gradients")
    #     ax[1].legend()

    #     plt.tight_layout()
    #     fig.savefig("../plots/gauss_activation_pipeline.png")
    #     plt.show()


    


if __name__ == "__main__":
    main()





