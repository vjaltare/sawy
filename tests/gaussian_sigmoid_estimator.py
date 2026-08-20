"""
Testing the gaussian sigmoid-estimator

- Sample a range of scalar preactivations y = U(-a, a)
- For every sample iterate over multiple averaging windows (nu > 2)
- Compute the averaged values, analytical expected values.
- Also compute analytical (gaussian) and estimated gradients (cdf(1- cdf))
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from utils import generate_gaussian_noise, generate_logistic_noise

import matplotlib.pyplot as plt
import seaborn as sns

# parser
# def parse_args():
#     parser = 

# Functions ...
def bipolar_heaviside(x):
    return 2*jnp.heaviside(x, 0) - 1


def expected_state_gauss(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float,
    ):

    Ey = 2*jax.scipy.stats.norm.cdf(x=x-threshold, loc=mean, scale=std) - 1

    return Ey


def exact_gauss_gradient(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float,
        
    ):

    grad_ = 2*jax.scipy.stats.norm.pdf(x=threshold-x, loc=mean, scale=std)

    return grad_

def estimated_gauss_gradient(
        x: jax.Array,
        threshold: float,
        std: float,
        mean: float
    ):

    grad_ = 2*(jax.scipy.stats.norm.cdf(x=threshold-x, loc=mean, scale=std)* (1 - jax.scipy.stats.norm.cdf(x=threshold-x, loc=mean, scale=std)))
    return grad_

# def generate_bipolar_samples(
#         x: jax.Array,
#         rngs: nnx.Rngs,
#         std: float,
#         mean: float,
#         noise_fn: Callable,
#     ):

#     noise_arr = 



def int_samples(
        x: jax.Array, #(N, ), bipolar entries needed!
        noise_arr: jax.Array, #(N, nu)
        nu: int, # integration window
    ):

    x_noise = x[:, None] + noise_arr
    x_noise = jnp.mean(x_noise, axis=-1)
    return x_noise


def pipeline():
    # TODO

    return None


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


if __name__ == "__main__":
    main()





