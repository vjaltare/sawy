"""
Wrapper for dual_sample_binary_softmax to be used as a nnx.Module
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable

from .dual_sample_binary_softmax import dual_sample_binary_softmax, generate_gaussian_noise, generate_logistic_noise

class NoisyHardmax(nnx.Module):
    def __init__(
            self
    ):
    # TODO
        return
