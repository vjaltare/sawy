"""
nnx wrapper for integrated_ce_loss for managing rng streams
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from utils.dual_sample_auxilary_functions import generate_gaussian_noise, generate_logistic_noise
from utils.integrated_softmax import integrated_ce_loss

class IntegratedCELoss(nnx.Module):
    def __init__(
                self,
                rngs: nnx.Rngs,
                nu: int, # itegration window
                noise_mean: float, # noise mean
                noise_std: float, # noise std
                gauss_noise_flag: bool, # whether to use gaussian noise
                labels: jax.Array, # array of labels shape: (Batch, Classes)
            ):

        self.rngs = rngs
        self.nu = nu
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_fn = generate_gaussian_noise if gauss_noise_flag else generate_logistic_noise
        self.labels = labels

    def __call__(self, x):
        noise_array = self.noise_fn(shape=x.shape + (self.nu,), rngs=self.rngs, std=self.noise_std, mean=self.noise_mean)
        loss = integrated_ce_loss(x=x, noise_array=noise_array, labels=self.labels)
        return loss


# testing
def main():
    rngs = nnx.Rngs(key=3452, default=0)
    x = jax.random.normal(rngs.key(), (5, ))
    labels = jax.nn.one_hot(3, num_classes=x.shape[0])
    int_loss = IntegratedCELoss(rngs=rngs, nu=2, noise_mean=0.0, noise_std=0.1, gauss_noise_flag=True, labels=labels)

    out = int_loss(x)
    print(f"Outputs: {out}")

# if __name__ == "__main__":
#     main()
