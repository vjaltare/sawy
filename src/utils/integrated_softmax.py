"""
Softmax trident function

FORWARD MODE:
Softmax layer
y = Wx + b + n
Take k-one hot samples from the distribution of y (vector)
s_hat = mean(y1, y2, ..., yk)

REVERSE MODE:
J = diag(s_hat) - outer(s_hat, s_hat)
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
from einops import rearrange
from optax.losses import softmax_cross_entropy
from utils.dual_sample_auxilary_functions import generate_gaussian_noise, generate_logistic_noise


# -----------------------------------------
# One hot encoding a vector
# -----------------------------------------
def one_hot(x):
    """
    x: jax.Array. Shape (B, C, nu)
    """
    num_classes = x.shape[1]
    # compute the argmax
    idx_max = jnp.argmax(x, axis=1)
    s = jax.nn.one_hot(x=idx_max, num_classes=num_classes, axis=1)
    return s



# -----------------------------------------
# Noisy Hardmax
# -----------------------------------------
def noisy_hardmax(
        x: jax.Array, # (B, C)
        noise_array: jax.Array, # (x.shape, nu)
    ):

    # add noise
    x_noise_preact = x[..., None] + noise_array

    # compute one hot
    one_hots = one_hot(x_noise_preact)

    # compute the mean
    s_hat = jnp.mean(one_hots, axis=-1)

    return s_hat

# -----------------------------------------
# s_hat - crossentropy loss
# -----------------------------------------
def s_hat_ce_loss(x: jax.Array, # shape (B, C)
                  noise_array: jax.Array, # shape (B, C, nu)
                  labels: jax.Array, # shape (B, C)
                  eps = 1e-8,
                  ):
    """
    Implementing cross entropy loss with estimator of softmax s_hat
    Computes: Li = sum(label(ij) * log s_hat(ij), dim=j) where j is the dimension of class
    """

    # compute the estimated softmax
    s_hat = noisy_hardmax(x=x, noise_array=noise_array)

    L = -jnp.sum(labels * jnp.log(s_hat + eps), axis=-1)
    return L

# -----------------------------------------
# Estimated Jacobian
# -----------------------------------------
def estimated_jacobian(s_hat: jax.Array):

    C = s_hat.shape[-1]
    diag_part = s_hat[..., :, None] * jnp.eye(C)             # (..., C, C)
    outer_part = jnp.einsum("...j,...k->...jk", s_hat, s_hat)
    return diag_part - outer_part

# -----------------------------------------
# Estimated Cross-entropy loss
# -----------------------------------------
def estimated_ce_loss(
        preactivations: jax.Array, # (B, C)
        labels: jax.Array, # (B, C), must be one hot!
        noise_array: jax.Array,
        # logsumexp_factor: float = 1.0, # use this instead of the log term
    ):

    # L =  logsumexp_factor - jnp.einsum("...i, ...i -> ...", preactivations, labels) # analytical upper bound: jnp.log(jnp.exp(jnp.sum(preactivations, axis=-1)))
    # L = softmax_cross_entropy(logits=preactivations, labels=labels)
    L = s_hat_ce_loss(x=preactivations, labels=labels, noise_array=noise_array)

    return L

def estimated_ce_jacobian(
        s_hat: jax.Array, #(B, C)
        labels: jax.Array, #(B, C)
    ):

    J = s_hat - labels
    return J



# -----------------------------------------
# Custom backward pass: softmax
# -----------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=())
def integrated_softmax(
    x: jax.Array,
    noise_array: jax.Array
    ):

    s_hat = noisy_hardmax(x, noise_array)

    return s_hat

def _fwd(x, noise_array):
    s_hat = noisy_hardmax(x, noise_array)
    return s_hat, (s_hat, noise_array) #(primal, cotangent)

def _bwd(cotangents, gradients):
    s_hat, noise_array = cotangents
    J = estimated_jacobian(s_hat)

    ds = jnp.einsum("...ij, ...j -> ...i", J, gradients)

    d_noise = jnp.zeros_like(noise_array) # no gradients w.r.t noise
    return (ds, d_noise)

integrated_softmax.defvjp(_fwd, _bwd)

# -----------------------------------------
# Custom backward pass: softmax-ce loss
# -----------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=())
def integrated_ce_loss(
    x: jax.Array,
    noise_array: jax.Array,
    labels: jax.Array, # must be one hot!
    ):

    loss = estimated_ce_loss(preactivations=x, labels=labels, noise_array=noise_array)
    return loss

def _fwd_ce(
    x: jax.Array,
    noise_array: jax.Array,
    labels: jax.Array, 
    ):

    loss = estimated_ce_loss(preactivations=x, labels=labels, noise_array=noise_array)
    return loss, (x, noise_array, labels)

def _bwd_ce(cotangents, gradients):
    x, noise_array, labels = cotangents
    s_hat = noisy_hardmax(x=x, noise_array=noise_array)
    dx = gradients[..., None]*(s_hat - labels)
    d_noise = jnp.zeros_like(noise_array)
    d_labels = jnp.zeros_like(labels)

    return (dx, d_noise, d_labels)

integrated_ce_loss.defvjp(_fwd_ce, _bwd_ce)




# -----------------------------------------
# Testing: TODO finish this section!
# -----------------------------------------
def make_labels(rngs, B, C):
    idx = jax.random.randint(rngs.key(), (B,), 0, C)
    return jax.nn.one_hot(idx, C)

def test_shapes_and_zero_grad_for_noise_and_labels():
    """
    FROM CLAUDE
    """
    rngs = nnx.Rngs(key=1, default=0, noise=1)
    B, C, nu = 5, 10, 2
    x = jax.random.normal(rngs.key(), (B, C))
    noise_array = generate_gaussian_noise((B, C, nu), rngs, std=1.0, mean=0.0)
    labels = make_labels(rngs, B, C)
 
    def loss_fn(x, noise_array, labels):
        return jnp.sum(integrated_ce_loss(x, noise_array, labels))
 
    dx, dnoise, dlabels = jax.grad(loss_fn, argnums=(0, 1, 2))(x, noise_array, labels)

    print(f"GRADIENTS, batch 1, 10 elements: {dx}")
 
    assert dx.shape == x.shape
    assert dnoise.shape == noise_array.shape
    assert dlabels.shape == labels.shape
    assert jnp.all(dnoise == 0.0), "noise_array must receive zero gradient"
    assert jnp.all(dlabels == 0.0), "labels must receive zero gradient"
    assert jnp.all(jnp.isfinite(dx))
    print("[PASS] shapes correct; noise_array & labels get exactly zero gradient")

def test_softmax_layer():
    rngs = nnx.Rngs(key=1, default=0, noise=1)
    B, C, nu = 5, 10, 2
    x = jax.random.normal(rngs.key(), (B, C))
    noise_array = generate_gaussian_noise((B, C, nu), rngs, std=1.0, mean=0.0)
    # labels = make_labels(rngs, B, C)

    def loss_fn(x, noise_array):
            return jnp.sum(integrated_softmax(x, noise_array))

    dx, dnoise = jax.grad(loss_fn, argnums=(0, 1))(x, noise_array)

    print(f"GRADIENTS, batch 1, 10 elements: {dx[0, :10]}")
    assert dx.shape == x.shape
    assert dnoise.shape == noise_array.shape
    assert jnp.all(jnp.isfinite(dx))

    


def main():
    testing_softmax = True
    if testing_softmax:
        test_softmax_layer()
        

    testing_ce_loss = False
    if testing_ce_loss:
        test_shapes_and_zero_grad_for_noise_and_labels()



# if __name__=="__main__":
#     main()


