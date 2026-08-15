import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable

# Noise generators
def generate_gaussian_noise(
        shape: tuple,
        rngs: nnx.Rngs,
        std: float,
        mean: float = 0.0
        
    ):

    n = jax.random.normal(key=rngs.key(), shape=shape)*std + mean

    return n

def generate_logistic_noise(
        shape: tuple,
        rngs: nnx.Rngs,
        std: float,
        loc: float = 0.0
        
    ):

    n = jax.random.logistic(key=rngs.key(), shape=shape)*std + loc

    return n


