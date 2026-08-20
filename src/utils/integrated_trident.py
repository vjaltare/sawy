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

g_mean = 1/(len(g_nu))g_nu(y; loc, scale)=
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from dual_sample_auxilary_functions import generate_gaussian_noise, generate_logistic_noise

# -----------------------------------------
# Bipolar activation using heaviside
# -----------------------------------------
def bipolar_heaviside(x):
    return 2*jnp.heaviside(x, 0) - 1


def noisy_bipolar_heaviside(
        x: jax.Array, # (B, ...)
        rngs: nnx.Rngs,
        scale: float,
        loc: float,
        int_window: int,
        noise_fn: Callable = generate_gaussian_noise
    ):

    shape_x = x.shape
    x = x.reshape(-1,)

    # generate independent nose samples
    noise_arr = noise_fn(shape=(x.shape + (int_window,)), rngs=rngs, std=scale, mean=loc)

    print(f"noise arr shape: {noise_arr.shape}")

    # add the noise to inputs
    x = x[:, None] + noise_arr

    print(f"noisy x shape: {x.shape}")

    bipolar_heaviside(x)

    # average over the integration window
    x = jnp.mean(x, axis=-1)

    print(f"Averaged x shape: {x.shape}")

    x = x.reshape(shape_x)

    print(f"final x shape: {x.shape}")

    return x


# -----------------------------------------
# Custom backward pass
# -----------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=()) # integrating window is nondiff
def integrated_trident(
    x: jax.Array,
    noise_arr,
    ):
    # TODO:

    return 

# -----------------------------------------
# Testing
# -----------------------------------------
def main():

    heaviside_test_flag = False
    if heaviside_test_flag:
        x = jnp.arange(-2, 2, 0.1)
        y = bipolar_heaviside(x)

        print(y)

    noisy_h_test_ = True
    if noisy_h_test_:
        rngs = nnx.Rngs(key=22, default=0)
        x = jax.random.normal(rngs.key(), (10, 5, 5, 3))

        x_n = noisy_bipolar_heaviside(x, rngs, 0.1, 0, 2)
        print(x_n[0, ...])


if __name__ == "__main__":
    main()