"""
Dual sample binary approximation for softmax.

- We add gaussian noise here.
- For sigmoidal noise, refer to adjacent file: dual_sample_binary_sigmoid_softmax.py
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable


# --------------------------------------------------
# Defining the two BINARY sample softmax function: gaussian
# --------------------------------------------------
def dual_sample_gaussian_binary_softmax(
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

    # binary one-hot vectors
    s1 = jax.nn.one_hot(idx1, num_classes=x.shape[-1], dtype=x.dtype)
    s2 = jax.nn.one_hot(idx2, num_classes=x.shape[-1], dtype=x.dtype)
    # s1 = -jnp.ones_like(x1)
    # s1 = s1.at[idx1].set(1.0)
    # s2 = -jnp.ones_like(x2)
    # s2 = s2.at[idx2].set(1.0)

    s_hat = (s1+s2)/2

    return s_hat

# ----------------------------------------
# Jacobian for softmax approximation
# ----------------------------------------
def jacobian_estimate(
        s: jax.Array, # dual sample softmax {-1, 0, 1}
        **kwargs
        
    ):

    # eusure that the input is a 1D array
    assert s.ndim == 1, "Input must be a 1D array"

    # construct the jacobian
    diag = jnp.diag(s)
    cov = jnp.einsum("i, j -> ij", s, s)

    jacobian = diag - cov

    return jacobian

# --------------------------------------------------
# Defining custom vjp for softmax: TODO: implement custom vjp for softmax
# --------------------------------------------------
