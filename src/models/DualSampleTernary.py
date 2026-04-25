"""
Ternary Activation using Dual Sampling
Created on: 03/13/2026

TODO:
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from utils import dual_sample_ternary

class DualSampleTernary(nnx.Module):

    def __init__(self,
                 rngs: nnx.Rngs,
                 threshold: float = 0.0,
                 noise_std: float = 1.0,
                 noise_mean: float = 0.0,
                 ):

        self.rngs = rngs
        self.threshold = threshold
        self.noise_std = noise_std
        self.noise_mean = noise_mean

        # here we can optionally define learnable parameters

        # learnable offsets
        # self.loc_offset = nnx.Param(jnp.array(0.0))
        # self.scale_offset = nnx.Param(jnp.array(1.0))

    def __call__(self, x):
        key = self.rngs.activation()

        y = dual_sample_ternary(
            x,
            # (x - self.loc_offset) / (self.scale_offset + 1e-8),
            key,
            threshold=self.threshold,
            noise_std=self.noise_std,
            noise_mean=self.noise_mean
        )

        return y

                 