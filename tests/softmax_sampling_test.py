"""
Testing the sampling approximation of softmax function
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
import matplotlib.pyplot as plt

# ----------------------------------------
# logistic noise
# ----------------------------------------
def logistic_noise(
        shape: tuple,
        key: jax.random.key,
        s: float, # scale of the noise
        loc: float = 0.0, # location of the noise
    ):

    n = jax.random.logistic(key, shape=shape, dtype=jnp.float32) * s + loc
    return n

# --------------------------------------------------
# Defining the two sample softmax function: logistic
# --------------------------------------------------
def dual_sample_sigmoid_bipolar_softmax(
        x: jax.Array, # input/preactivation
        key: jax.random.key,
        s: float, # scale of the noise
    ):

    # draw two noisy samples from the input
    key, subkey = jax.random.split(key)
    n1 = logistic_noise(shape=x.shape, key=key, s=s)
    key, subkey = jax.random.split(key)
    n2 = logistic_noise(shape=x.shape, key=key, s=s)

    # add noise to the input (independent)
    [x1, x2] = [x + n1, x + n2]

    # hardmax across the inputs
    idx1 = jnp.argmax(x1, axis=-1)
    idx2 = jnp.argmax(x2, axis=-1)

    # generate bipolar ohe hot vectors
    s1 = -jnp.ones_like(x1)
    s1 = s1.at[idx1].set(1.0)
    s2 = -jnp.ones_like(x2)
    s2 = s2.at[idx2].set(1.0)

    s_hat = (s1+s2)/2

    return s_hat

# --------------------------------------------------
# Defining the two sample softmax function: gaussian
# --------------------------------------------------
def dual_sample_gaussian_bipolar_softmax(
        x: jax.Array, # input/preactivation
        key: jax.random.key,
        s: float, # scale of the noise
        mean: float = 0.0
    ):

    # draw two noisy samples from the input
    key, subkey = jax.random.split(key)
    n1 = jax.random.normal(key=key, shape=x.shape)*s + mean
    key, subkey = jax.random.split(key)
    n2 = jax.random.normal(key=key, shape=x.shape)*s + mean

    # add noise to the input (independent)
    [x1, x2] = [x + n1, x + n2]

    # hardmax across the inputs
    idx1 = jnp.argmax(x1, axis=-1)
    idx2 = jnp.argmax(x2, axis=-1)

    # generate bipolar ohe hot vectors
    s1 = -jnp.ones_like(x1)
    s1 = s1.at[idx1].set(1.0)
    s2 = -jnp.ones_like(x2)
    s2 = s2.at[idx2].set(1.0)

    s_hat = (s1+s2)/2

    return s_hat

# ----------------------------------------
# Test
# ----------------------------------------
def main():
    testing_logistic_noise_ = False
    if testing_logistic_noise_:
        rngs = nnx.Rngs(key=0)
        noise_arr = logistic_noise(shape=(1000,), key=rngs.key(), s=1.0, loc=0.0)
        noise_arr2 = logistic_noise(shape=(1000,), key=rngs.key(), s=0.5, loc=0.0)

    

        plt.hist(noise_arr, bins=50, density=True, alpha=0.3)
        plt.hist(noise_arr2, bins=50, density=True, alpha=0.3)
        plt.savefig("../plots/sigmoid_noise_hist.png", bbox_inches="tight")
        plt.show()

    testing_dual_sample_sigmoid = False
    if testing_dual_sample_sigmoid:
        x_far = jnp.array([0.9, 0.03, 0.02])
        x_close = jnp.array([0.5, 0.49, 0.01])

        rngs = nnx.Rngs(key=0)
        s_far_list = []
        s_close_list = []

        for _ in range(1000):
            s_far = dual_sample_sigmoid_bipolar_softmax(x_far, rngs.key(), s=0.01)
            s_close = dual_sample_sigmoid_bipolar_softmax(x_close, rngs.key(), s=0.01)
            s_far_list.append(s_far.tolist())
            s_close_list.append(s_close.tolist())

        s_far_arr = jnp.array(s_far_list)
        s_far_arr = s_far_arr.flatten()
        s_close_arr = jnp.array(s_close_list)
        s_close_arr = s_close_arr.flatten()

        # plot histograms
        plt.hist(s_far_arr, bins=50, alpha=0.3, label="far labels")
        plt.hist(s_close_arr, bins=50, alpha=0.3, label="close labels")
        plt.legend()
        plt.savefig("../plots/sigmoid_softmax_hist.png", bbox_inches="tight")
        plt.show()

    testing_dual_sample_gaussian = True
    if testing_dual_sample_gaussian:
        x_far = jnp.array([0.9, 0.03, 0.02])
        x_close = jnp.array([0.5, 0.49, 0.01])

        rngs = nnx.Rngs(key=0)
        s_far_list = []
        s_close_list = []

        for _ in range(1000):
            s_far = dual_sample_gaussian_bipolar_softmax(x_far, rngs.key(), s=0.1)
            s_close = dual_sample_gaussian_bipolar_softmax(x_close, rngs.key(), s=0.1)
            s_far_list.append(s_far.tolist())
            s_close_list.append(s_close.tolist())

        s_far_arr = jnp.array(s_far_list)
        s_far_arr = s_far_arr.flatten()
        s_close_arr = jnp.array(s_close_list)
        s_close_arr = s_close_arr.flatten()

        # plot histograms
        plt.hist(s_far_arr, bins=50, alpha=0.3, label="far labels")
        plt.hist(s_close_arr, bins=50, alpha=0.3, label="close labels")
        plt.legend()
        plt.savefig("../plots/gaussian_softmax_hist.png", bbox_inches="tight")
        plt.show()            





if __name__ == "__main__":
    main()
    



    
