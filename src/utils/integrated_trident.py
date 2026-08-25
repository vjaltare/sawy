"""
Trident module with built-in integration window.

FORWARD MODE:
y = Wx + b + n (or similar linear transformation)
Draw bipolar samples from the distribution of y (y_k)

T(y; loc, scale) = 1/nu ∑_k y_k 
- For 2 samples (k = 2) T = {-1, 0, 1}

REVERSE MODE:
g_2(y; loc, scale) = {1, if y_1 + y_2 = 0}; 0, otherwise}

For averaging over a window nu (nu = 2n, n is an integer)

g_mean = 1/(len(g_nu))g_nu(y; loc, scale)
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from utils.dual_sample_auxilary_functions import generate_gaussian_noise, generate_logistic_noise

# -----------------------------------------
# Bipolar activation using heaviside
# -----------------------------------------
def bipolar_heaviside(x):
    return 2*jnp.heaviside(x, 0) - 1


def noisy_bipolar_heaviside(
        x: jax.Array, # (B, ...)
        noise_arr: jax.Array, # (x.shape, int_window)
    ):

    # add noise
    x_noisy_preact = x[..., None] + noise_arr

    # pass through non-linearity
    x_noise = bipolar_heaviside(x_noisy_preact)
    

    # average over the last dimension (nu)
    x_mean = jnp.mean(x_noise, axis=-1)

    return x_mean, x_noise

def sampled_gradient(x):
    """
    x: Bipolar Array. (num_points, nu). Averaging dimension should be the last one. 
    Should be used for all noise distributions.
    """

    assert x.shape[-1] >= 2

    g_samples = jnp.abs(x[..., :-1] - x[..., 1:])/2 # pairwise average
    g_samples = jnp.mean(g_samples, axis=-1)

    return g_samples

def sigmoid(x, loc, scale):
    return 1/(1 + jnp.exp(-(x - loc)/scale))


# -----------------------------------------
# Custom backward pass
# -----------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=()) # integrating window is nondiff
def integrated_trident(
    x: jax.Array,
    noise_arr: jax.Array,
    ):

    x_mean, x_noise = noisy_bipolar_heaviside(x=x, noise_arr=noise_arr) 

    return x_mean # return a single differentiable output

def _fwd(
    x: jax.Array,
    noise_arr: jax.Array,
    ):

    x_mean, x_noise = noisy_bipolar_heaviside(x=x, noise_arr=noise_arr) 
    
    return x_mean, x_noise # (primal_out, cotangents)

def _bwd(cotangents, gradients):
    x_noise = cotangents

    dx = sampled_gradient(x_noise)*gradients
    zeros_ = jnp.zeros_like(x_noise)

    return (dx, zeros_)

integrated_trident.defvjp(_fwd, _bwd)

# TODO: testing!




# -----------------------------------------
# Testing
# -----------------------------------------
def main():
    rngs = nnx.Rngs(key=234, default=0)
    x = jax.random.normal(rngs.key(), (10, 5, 5, 3))
    noise_arr = jax.random.normal(rngs.key(), x.shape + (2,))

    out_x = integrated_trident(x, noise_arr)
    print(f"output: {out_x}")

    def scalar_fn(x_i, noise_i):
        return jnp.sum(integrated_trident(x_i, noise_i))

    grad_fn = jax.grad(scalar_fn, argnums=(0, 1))
    dx, dnoise = jax.vmap(grad_fn, in_axes=(0, 0))(x, noise_arr)
    print(f"dx: {dx}")
    print(f"dnoise: {dnoise}")




if __name__ == "__main__":
    main()