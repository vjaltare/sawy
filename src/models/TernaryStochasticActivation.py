"""
Flax nnx module for ternary stochastic activation.
This module creates a wrapper arounf ternery activation function to make thresholds trainable parameters of the model.

Updates:
02/23/2026: Learnable thresholds.
TODO: Learnable noise standard deviation which controls the amplitude of additive noise.
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from utils import ternary_activation

class TernaryStochasticActivation(nnx.Module):
    """
    nnx Module for ternary activation with learnable parameters.
    """

    def __init__(self,
                 thresholds: list[float],
                 levels: list[float],
                 noise_std: float,
                 rngs: nnx.Rngs,
                 noise_mean: float = 0.0,
                 ):
        
        self.rngs = rngs
        self.levels = jnp.array(levels)
        self.noise_std = noise_std
        self.noise_mean = noise_mean

        # define thresholds as learnable parameters
        self.thresholds = nnx.Param(jnp.array(thresholds))

    def __call__(self, x):
        key = self.rngs.activation()
        y = ternary_activation(
            x,
            thresholds=self.thresholds,
            levels=self.levels,
            noise_std=self.noise_std,
            noise_mean=self.noise_mean,
            key=key
        )

        return y

