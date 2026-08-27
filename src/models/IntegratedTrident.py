"""
nnx wrapper for integrated_trident for managing rng streams
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from utils.dual_sample_auxilary_functions import generate_gaussian_noise, generate_logistic_noise
from utils.integrated_trident import integrated_trident

class IntegratedTrident(nnx.Module):
    def __init__(
            self,
            rngs: nnx.Rngs,
            nu: int, # itegration window
            noise_mean: float, # noise mean
            noise_std: float, # noise std
            gauss_noise_flag: bool, # whether to use gaussian noise

        ):

        self.rngs = rngs
        self.nu = nu
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_fn = generate_gaussian_noise if gauss_noise_flag else generate_logistic_noise


    def __call__(self, x):
        noise_array = self.noise_fn(shape=x.shape + (self.nu,), rngs=self.rngs, std=self.noise_std, mean=self.noise_mean)
        x_mean = integrated_trident(x=x, noise_arr=noise_array)

        return x_mean


# testing
def main():
    rngs = nnx.Rngs(key=3452, default=0)
    int_trident = IntegratedTrident(rngs=rngs, nu=2, noise_mean=0.0, noise_std=0.1, gauss_noise_flag=True)
    x = jax.random.normal(rngs.key(), (5, 3))
    out = int_trident(x)
    print(f"Outputs: {out}")

# if __name__ == "__main__":
#     main()



        