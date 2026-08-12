"""
Dual sample binary approximation for sigmoidal softmax.
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable

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
def dual_sample_sigmoid_binary_softmax(
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
    # s1 = -jnp.ones_like(x1)
    # s1 = s1.at[idx1].set(1.0)
    # s2 = -jnp.ones_like(x2)
    # s2 = s2.at[idx2].set(1.0)
    s1 = jax.nn.one_hot(idx1, num_classes=x.shape[-1], dtype=jnp.float32)
    s2 = jax.nn.one_hot(idx2, num_classes=x.shape[-1], dtype=jnp.float32)

    s_hat = (s1+s2)/2

    return s_hat