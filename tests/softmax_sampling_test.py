"""
Testing the sampling approximation of softmax function
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx
from functools import partial
from typing import Callable
import matplotlib.pyplot as plt

# ----------------------------------------
# logistic noise
# ----------------------------------------
def logistic_noise(
        shape: tuple,
        key: jax.random.key,
        s: float, # scale of the noise
        loc: float = 0.0, # location of the noise
    ):

    n = jax.random.logistic(key, shape=shape, dtype=jnp.float32) * s + loc
    return n

# --------------------------------------------------
# Defining the two sample softmax function: logistic
# --------------------------------------------------
def dual_sample_sigmoid_bipolar_softmax(
        x: jax.Array, # input/preactivation
        key: jax.random.key,
        s: float, # scale of the noise
    ):

    # draw two noisy samples from the input
    key, subkey = jax.random.split(key)
    n1 = logistic_noise(shape=x.shape, key=key, s=s)
    key, subkey = jax.random.split(key)
    n2 = logistic_noise(shape=x.shape, key=key, s=s)

    # add noise to the input (independent)
    [x1, x2] = [x + n1, x + n2]

    # hardmax across the inputs
    idx1 = jnp.argmax(x1, axis=-1)
    idx2 = jnp.argmax(x2, axis=-1)

    # generate bipolar ohe hot vectors
    s1 = -jnp.ones_like(x1)
    s1 = s1.at[idx1].set(1.0)
    s2 = -jnp.ones_like(x2)
    s2 = s2.at[idx2].set(1.0)

    s_hat = (s1+s2)/2

    return s_hat

# --------------------------------------------------
# Defining the two sample softmax function: gaussian
# --------------------------------------------------
def dual_sample_gaussian_bipolar_softmax(
        x: jax.Array, # input/preactivation
        key: jax.random.key,
        s: float, # scale of the noise
        mean: float = 0.0
    ):

    # draw two noisy samples from the input
    key, subkey = jax.random.split(key)
    n1 = jax.random.normal(key=key, shape=x.shape)*s + mean
    key, subkey = jax.random.split(key)
    n2 = jax.random.normal(key=key, shape=x.shape)*s + mean

    # add noise to the input (independent)
    [x1, x2] = [x + n1, x + n2]

    # hardmax across the inputs
    idx1 = jnp.argmax(x1, axis=-1)
    idx2 = jnp.argmax(x2, axis=-1)

    # generate bipolar ohe hot vectors
    s1 = -jnp.ones_like(x1)
    s1 = s1.at[idx1].set(1.0)
    s2 = -jnp.ones_like(x2)
    s2 = s2.at[idx2].set(1.0)

    s_hat = (s1+s2)/2

    return s_hat

# --------------------------------------------------
# Defining the two BINARY sample softmax function: gaussian
# --------------------------------------------------
def dual_sample_gaussian_binary_softmax(
        x: jax.Array, # input/preactivation
        key: jax.random.key,
        s: float, # scale of the noise
        mean: float = 0.0
    ):

    # draw two noisy samples from the input
    key, subkey = jax.random.split(key)
    n1 = jax.random.normal(key=key, shape=x.shape)*s + mean
    key, subkey = jax.random.split(key)
    n2 = jax.random.normal(key=key, shape=x.shape)*s + mean

    # add noise to the input (independent)
    [x1, x2] = [x + n1, x + n2]

    # hardmax across the inputs
    idx1 = jnp.argmax(x1, axis=-1)
    idx2 = jnp.argmax(x2, axis=-1)

    # binary one-hot vectors
    s1 = jax.nn.one_hot(idx1, num_classes=x.shape[-1], dtype=jnp.float32)
    s2 = jax.nn.one_hot(idx2, num_classes=x.shape[-1], dtype=jnp.float32)
    # s1 = -jnp.ones_like(x1)
    # s1 = s1.at[idx1].set(1.0)
    # s2 = -jnp.ones_like(x2)
    # s2 = s2.at[idx2].set(1.0)

    s_hat = (s1+s2)/2

    return s_hat

# ----------------------------------------
# Jacobian for softmax approximation
# ----------------------------------------
def softmax_jacobian_approximation(
        s: jax.Array, # dual sample softmax {-1, 0, 1}
        **kwargs
        
    ):

    # eusure that the input is a 1D array
    assert s.ndim == 1, "Input must be a 1D array"

    # construct the jacobian
    diag = jnp.diag(s)
    cov = jnp.einsum("i, j -> ij", s, s)

    jacobian = diag - cov

    return jacobian

# -------------------------------------------------------
# Jacobian for softmax approximation - cross entropy loss
# -------------------------------------------------------
def softmax_jacobian_approximation(
        s: jax.Array, # dual sample softmax {-1, 0, 1}
        label: int, # true label
        **kwargs
        
    ):

    # eusure that the input is a 1D array
    assert s.ndim == 1, "Input must be a 1D array"

    assert label >=0 and type(label) == int and label < s.shape[0], "Label must be a valid index"

    # construct one-hot vector for the label
    q = jax.nn.one_hot(label, num_classes=s.shape[0], dtype=jnp.float32)

    # construct the jacobian
    # diag = jnp.diag(s)
    # cov = jnp.einsum("i, j -> ij", s, s)

    jacobian = s - q

    return jacobian


# ----------------------------------------
# Test
# ----------------------------------------
def main():
    testing_logistic_noise_ = False
    if testing_logistic_noise_:
        rngs = nnx.Rngs(key=0)
        noise_arr = logistic_noise(shape=(1000,), key=rngs.key(), s=1.0, loc=0.0)
        noise_arr2 = logistic_noise(shape=(1000,), key=rngs.key(), s=0.5, loc=0.0)

    

        plt.hist(noise_arr, bins=50, density=True, alpha=0.3)
        plt.hist(noise_arr2, bins=50, density=True, alpha=0.3)
        plt.savefig("../plots/sigmoid_noise_hist.png", bbox_inches="tight")
        plt.show()

    testing_dual_sample_sigmoid = False
    if testing_dual_sample_sigmoid:
        x_far = jnp.array([0.9, 0.03, 0.02])
        x_close = jnp.array([0.5, 0.49, 0.01])

        rngs = nnx.Rngs(key=0)
        s_far_list = []
        s_close_list = []

        for _ in range(1000):
            s_far = dual_sample_sigmoid_bipolar_softmax(x_far, rngs.key(), s=0.01)
            s_close = dual_sample_sigmoid_bipolar_softmax(x_close, rngs.key(), s=0.01)
            s_far_list.append(s_far.tolist())
            s_close_list.append(s_close.tolist())

        s_far_arr = jnp.array(s_far_list)
        s_far_arr = s_far_arr.flatten()
        s_close_arr = jnp.array(s_close_list)
        s_close_arr = s_close_arr.flatten()

        # plot histograms
        plt.hist(s_far_arr, bins=50, alpha=0.3, label="far labels")
        plt.hist(s_close_arr, bins=50, alpha=0.3, label="close labels")
        plt.legend()
        plt.savefig("../plots/sigmoid_softmax_hist.png", bbox_inches="tight")
        plt.show()

    testing_dual_sample_gaussian = False
    if testing_dual_sample_gaussian:
        x_far = jnp.array([0.9, 0.03, 0.02])
        x_close = jnp.array([0.5, 0.49, 0.01])

        rngs = nnx.Rngs(key=0)
        s_far_list = []
        s_close_list = []

        for _ in range(1000):
            s_far = dual_sample_gaussian_bipolar_softmax(x_far, rngs.key(), s=0.1)
            s_close = dual_sample_gaussian_bipolar_softmax(x_close, rngs.key(), s=0.1)
            s_far_list.append(s_far.tolist())
            s_close_list.append(s_close.tolist())

        s_far_arr = jnp.array(s_far_list)
        s_far_arr = s_far_arr.flatten()
        s_close_arr = jnp.array(s_close_list)
        s_close_arr = s_close_arr.flatten()

        # plot histograms
        plt.hist(s_far_arr, bins=50, alpha=0.3, label="far labels")
        plt.hist(s_close_arr, bins=50, alpha=0.3, label="close labels")
        plt.legend()
        plt.savefig("../plots/gaussian_softmax_hist.png", bbox_inches="tight")
        plt.show()

    testing_dual_sample_binary_gaussian = False
    if testing_dual_sample_binary_gaussian:
        x_far = jnp.array([0.9, 0.03, 0.02])
        x_close = jnp.array([0.5, 0.49, 0.01])

        rngs = nnx.Rngs(key=0)
        s_far_list = []
        s_close_list = []

        for _ in range(1000):
            s_far = dual_sample_gaussian_binary_softmax(x_far, rngs.key(), s=0.1)
            s_close = dual_sample_gaussian_binary_softmax(x_close, rngs.key(), s=0.1)
            s_far_list.append(s_far.tolist())
            s_close_list.append(s_close.tolist())

        s_far_arr = jnp.array(s_far_list)
        s_far_arr = s_far_arr.flatten()
        s_close_arr = jnp.array(s_close_list)
        s_close_arr = s_close_arr.flatten()

        # plot histograms
        plt.hist(s_far_arr, bins=50, alpha=0.3, label="far labels")
        plt.hist(s_close_arr, bins=50, alpha=0.3, label="close labels")
        plt.legend()
        plt.savefig("../plots/gaussian_softmax_binary_hist.png", bbox_inches="tight")
        plt.show()   

    testing_jacobian_ = False
    if testing_jacobian_:
        x_far = jnp.array([0.9, 0.03, 0.02])
        x_close = jnp.array([0.5, 0.49, 0.01])
        rngs = nnx.Rngs(key=9)

        s_close = dual_sample_gaussian_bipolar_softmax(x_close, rngs.key(), s=0.1)
        j_close = softmax_jacobian_approximation(s_close)
        s_far = dual_sample_gaussian_bipolar_softmax(x_far, rngs.key(), s=0.1)
        j_far = softmax_jacobian_approximation(s_far)

        print("Close labels: ", s_close)
        print("Far labels: ", s_far)

        print("Jacobian close: ", j_close)
        print("Jacobian far: ", j_far)

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        # plot s on the left
        ax[0].bar(range(len(s_close)), s_close, alpha=0.3, label="close labels")
        ax[0].bar(range(len(s_far)), s_far, alpha=0.3, label="far labels")

        # plot close jacobian in the middle
        im = ax[1].imshow(j_close, cmap="viridis", interpolation="nearest")
        ax[1].set_title("Jacobian (close)")
        # add colorbar
        plt.colorbar(im, ax=ax[1])


        # plot far jacobian on the right
        im2 = ax[2].imshow(j_far, cmap="viridis", interpolation="nearest")
        ax[2].set_title("Jacobian (far)")
        # add colorbar
        plt.colorbar(im2, ax=ax[2])

        plt.savefig("../plots/softmax_jacobian.png", bbox_inches="tight")

    testing_jacobian_binary_ = True
    if testing_jacobian_binary_:
        x_far = jnp.array([0.9, 0.03, 0.02])
        x_close = jnp.array([0.5, 0.49, 0.01])
        rngs = nnx.Rngs(key=9)

        s_close = dual_sample_gaussian_binary_softmax(x_close, rngs.key(), s=0.1)
        j_close = softmax_jacobian_approximation(s_close)
        s_far = dual_sample_gaussian_binary_softmax(x_far, rngs.key(), s=0.1)
        j_far = softmax_jacobian_approximation(s_far)

        print("Close labels: ", s_close)
        print("Far labels: ", s_far)

        print("Jacobian close: ", j_close)
        print("Jacobian far: ", j_far)

        fig, ax = plt.subplots(1, 3, figsize=(10, 5))

        # plot s on the left
        ax[0].bar(range(len(s_close)), s_close, alpha=0.3, label="close labels")
        ax[0].bar(range(len(s_far)), s_far, alpha=0.3, label="far labels")

        # plot close jacobian in the middle
        im = ax[1].imshow(j_close, cmap="viridis", interpolation="nearest")
        ax[1].set_title("Jacobian (close)")
        # add colorbar
        plt.colorbar(im, ax=ax[1])


        # plot far jacobian on the right
        im2 = ax[2].imshow(j_far, cmap="viridis", interpolation="nearest")
        ax[2].set_title("Jacobian (far)")
        # add colorbar
        plt.colorbar(im2, ax=ax[2])

        plt.savefig("../plots/softmax_jacobian_binary.png", bbox_inches="tight")
    




if __name__ == "__main__":
    main()
    



    
