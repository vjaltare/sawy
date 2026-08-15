"""
Dual sample binary approximation for softmax.

- We add gaussian noise here.
- For sigmoidal noise, refer to adjacent file: dual_sample_binary_sigmoid_softmax.py
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from .dual_sample_auxilary_functions import generate_gaussian_noise, generate_logistic_noise

# --------------------------------------------------
# Auxiliary functions
# --------------------------------------------------
def noisy_hardmax(
        x: jax.Array,
        rngs: nnx.Rngs,
        scale: float,
        loc: float,
        noise_fn: Callable = generate_gaussian_noise,
        ):

    # generate independent noise samples
    n1 = noise_fn(shape=x.shape, rngs=rngs, std=scale, mean=loc)
    n2 = noise_fn(shape=x.shape, rngs=rngs, std=scale, mean=loc)

    return dual_sample_binary_softmax(x=x, n1=n1, n2=n2)


# ----------------------------------------
# Jacobian for softmax approximation
# ----------------------------------------
def jacobian_estimate(
        s: jax.Array, # dual sample softmax {-1, 0, 1}
        **kwargs
        
    ):

    # eusure that the input is a 1D array
    # assert s.ndim == 1, "Input must be a 1D array"

    # construct the jacobian
    diag = jnp.einsum("...i, ij -> ...ij", s, jnp.eye(s.shape[-1], dtype=s.dtype))
    cov = jnp.einsum("...i, ...j -> ...ij", s, s)

    jacobian = 2.0 * (diag - cov) # Refer to theory for factor 2

    return jacobian

# --------------------------------------------------
# Shorthand implementation of noisy hardmax
# --------------------------------------------------
def _impl(x, n1, n2):
    """
    Implementing noisy hardmax
    """
    x1, x2 = x + n1, x + n2
    C = x.shape[-1]
    s1 = jax.nn.one_hot(jnp.argmax(x1, axis=-1), num_classes=C, dtype=x.dtype)
    s2 = jax.nn.one_hot(jnp.argmax(x2, axis=-1), num_classes=C, dtype=x.dtype)
    return (s1 + s2) / 2


# --------------------------------------------------
# Defining custom vjp for softmax
# --------------------------------------------------
@jax.custom_vjp
def dual_sample_binary_softmax(
        x: jax.Array, # input/preactivation. shape -> (B, C)
        n1: jax.Array, # noise array 1
        n2: jax.Array, # noise array 2
    ):


    return _impl(x, n1, n2)

def _fwd(
        x: jax.Array,
        n1: jax.Array,
        n2: jax.Array,
        
    ):

    s_hat = _impl(x, n1, n2)

    return s_hat, (s_hat,)

def _bwd(cotangents, gradients):
    (s_hat,) = cotangents

    # compute the jacobian estimate
    jacobian = jacobian_estimate(s_hat)

    # compute the gradient w.r.t. the input
    grad_x = jnp.einsum("...ij, ...j -> ...i", jacobian, gradients)

    zeros_ = jnp.zeros_like(grad_x)

    return grad_x, zeros_, zeros_


dual_sample_binary_softmax.defvjp(_fwd, _bwd)

# --------------------------------------------------
# Defining custom vjp for ce loss
# --------------------------------------------------
def _ce_impl(
    x: jax.Array, # preactivations
    labels: jax.Array, # one-hot labels
    n1: jax.Array, # noise array 1
    n2: jax.Array, # noise array 2
    ):

    s_hat = dual_sample_binary_softmax(x, n1, n2)
    loss = -jnp.mean(jnp.sum(labels * jnp.log(s_hat + 1e-8), axis=-1)) # cross-entropy loss
    return loss, s_hat


@jax.custom_vjp
def dual_sample_ce_loss(
    x: jax.Array, # preactivations
    labels: jax.Array, # one-hot labels
    n1: jax.Array, # noise array 1
    n2: jax.Array, # noise array 2
    ):

    loss, logits = _ce_impl(x, labels, n1, n2)

    return loss, logits

def _ce_fwd(x, labels, n1, n2):
    loss, logits = _ce_impl(x, labels, n1, n2)
    return loss, (x, labels, logits)

def _ce_bwd(cotangents, gradients):
    x, labels, logits = cotangents

    # gradient w.r.t x 
    dx = logits - labels
    zeros_ = jnp.zeros_like(dx)

    return dx, zeros_, zeros_, zeros_

dual_sample_ce_loss.defvjp(_ce_fwd, _ce_bwd)



# --------------------------------------------------
# Comparing against rank 1 vjp
# --------------------------------------------------
def rank1_vjp(x1, x2, grads):
    d = x1 - x2
    return 0.5 * d * jnp.sum(d*grads, axis=-1, keepdims=True)



# --------------------------------------------------
# Testing....
# --------------------------------------------------
def main():

    rngs = nnx.Rngs(key=2341, default = 0)
    (B, C) = (5, 10) # batch x classes

    # "far"   = one class clearly ahead   -> draws usually AGREE  -> J_hat ~ 0
    # "close" = top classes nearly tied   -> draws often DISAGREE -> J_hat live
    base = jax.random.normal(rngs.key(), shape=(B, C))
    x_far = 3.0 * base                       # large logit spread
    x_close = 0.05 * base                    # nearly flat logits
    SCALE = 1.0

    print("\n" + "=" * 68)
    print("1. FORWARD: structure of s_hat")
    print("=" * 68)
    s_hat = noisy_hardmax(x_close, rngs, SCALE, 0.0)
    print("s_hat[0] =", s_hat[0])
    print("row sums  =", s_hat.sum(-1), " (must be exactly 1)")
    print("values    =", jnp.unique(s_hat), " (must be a subset of {0, .5, 1})")
 
    print("\n" + "=" * 68)
    print("2. DISAGREEMENT RATE in the two regimes")
    print("=" * 68)
    for name, xx in (("far  ", x_far), ("close", x_close)):
        sh = noisy_hardmax(xx, rngs, SCALE, 0.0)
        disagree = jnp.mean(jnp.any(sh == 0.5, axis=-1))
        print(f"  {name}: disagreement c = {float(disagree):.3f}"
              f"   (J_hat is exactly 0 on the agreeing rows)")
 
    y = jax.nn.one_hot(jnp.arange(B) % C, C)

    n1 = generate_gaussian_noise(shape=x_close.shape, rngs=rngs, std=SCALE, mean=0.0)
    n2 = generate_gaussian_noise(shape=x_close.shape, rngs=rngs, std=SCALE, mean=0.0)
 
    def loss(z):
        return -jnp.mean(jnp.sum(y * dual_sample_binary_softmax(z, n1, n2), -1))
 
    gx = jax.grad(loss)(x_close)
    print(f"dL/dx = {gx}")

    # test the ce loss. TODO: move the samping to _ce_fwd function
    loss_val, logits = dual_sample_ce_loss(x_close, y, n1, n2)
    print(f"CE loss = {loss_val}")
    gx = jax.grad(dual_sample_ce_loss)
    gx_eval = gx(x_close, y, n1, n2)
    print(f"dL/dx = {gx_eval[0]}")


# if __name__ == "__main__":
#     main()