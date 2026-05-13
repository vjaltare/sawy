"""
Custom Linear layer that provides syntax like TriDENT but essentially only applies identity transformation.
"""
import jax
import jax.numpy as jnp
import flax
from flax import nnx

class CustomLinear(nnx.Module):

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
        
    def __call__(self, x):
        return x