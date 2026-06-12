"""
Ternary Sigmoid activation function.

Written in the call signature of dual_sample_ternary for two reasons:
1. For consistent call signature while running scripts.
2. Can later add noise!

TODO:
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx

# comment later
# import matplotlib.pyplot as plt

class TernarySigmoid(nnx.Module):

    def __init__(
            self,
            rngs: nnx.Rngs,
            threshold: float = None,
            noise_std: float = None,
            noise_mean: float = None
    ):
        self.rngs = rngs
        # self.threshold = threshold
        # self.noise_std = noise_std
        # self.noise_mean = noise_mean

    def __call__(self, x):
        y = 2*nnx.sigmoid(x) - 1
        return y
    
# testing code
# def main():
#     x = jnp.linspace(-5, 5, 100)
#     act = TernarySigmoid(nnx.Rngs(params=0, dropout=0, activation=0, next=0))
#     y = act(x)

#     plt.plot(x, y)
#     plt.xlabel("Input")
#     plt.ylabel("Output")
#     plt.savefig("ternary_sigmoid.png")
#     plt.show()

# if __name__ == "__main__":
#     main()


