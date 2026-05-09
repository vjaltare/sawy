"""
A simple 1-hidden layer FFN for characterizing TriDENT's performance on smaller datasets.
"""

import jax
import jax.numpy as jnp
import flax
from flax import nnx

from models import DualSampleTernary


class FFN(nnx.Module):

    def __init__(
            self,
            rngs: nnx.Rngs,
            layers: list[int],
            noise_std: float = 0.5,
            threshold: float = 0.0,
            ActivationFunction: nnx.Module = DualSampleTernary,
            **kwargs):
        
        self.activation = ActivationFunction(threshold=threshold, noise_std=noise_std, rngs=rngs)

        self.layers = nnx.List([
            nnx.Linear(in_features=li, out_features=lo, rngs=rngs) for li, lo in zip(layers[:-1], layers[1:])
        ])
    
    def __call__(self, x):

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x


## testing
# if __name__ == "__main__":
#     model = FFN(rngs=nnx.Rngs(0), layers=[4, 10, 3])
#     nnx.display(model)
#     x = jnp.empty((1, 4))
#     print(model(x))

