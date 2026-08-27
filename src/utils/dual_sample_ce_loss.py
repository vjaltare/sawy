"""
NOTE: REFER TO dual_sample_binary_softmax.py for the original implementation of dual_sample_ce_loss. This file is a duplicate of that file and is used to avoid circular imports.


Softmax-cross entropy loss with dual sampling approach
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from .dual_sample_auxilary_functions import generate_gaussian_noise, generate_logistic_noise
from .dual_sample_binary_softmax import dual_sample_binary_softmax

