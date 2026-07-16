"""
MLP Mixer Architecture
"""
import jax
import jax.numpy as jnp
import flax
from flax import nnx
import DualSampleTernary, FFN
import einops
from typing import Callable
from functools import partial

# -------------------------------------------------------
# MLP Block
# -------------------------------------------------------
class MLPBlock(nnx.Module):
    """
    MLP Block for mixing across tokens or channels
    """

    # NOTE: implemented in FFN

    # def __init__(self, 
    #              in_features: int, 
    #              out_features: int,
    #              ActivationFunction: nnx.Module,
    #              threshold: float = 0.0,
    #              noise_std: float = 0.1,
    #              ):
        
    #     self.in_features = in_features
    #     self.out_features = out_features
    #     self.activation_function = ActivationFunction() # here use DualSampleTernary or relu 



# -------------------------------------------------------
# Mixer Block
# -------------------------------------------------------
class MixerBlock(nnx.Module):
    """
    Mixer block in the MLP architecture
    """

    def __init__(self,
                 channel_dim: int, # C
                 channel_mix_hidden_dim: int, # Dc
                 token_dim: int, # S
                 token_mix_hidden_dim: int, # Ds
                 ActivationFunction: nnx.Module,
                 rngs: nnx.Rngs,
                 threshold: float = 0.0,
                 noise_std: float = 0.1,
                 ):

        # construct the activation function
        self.activation_function = ActivationFunction(threshold=threshold, noise_std=noise_std, rngs=rngs) # here use DualSampleTernary or relu

        # Define channel mixing MLP
        self.mlp_c = FFN(
            rngs=rngs,
            layers=[channel_dim, channel_mix_hidden_dim, channel_dim],
            noise_std=noise_std,
            threshold=threshold,
            ActivationFunction=ActivationFunction
        )

        # Define token mixing MLP
        self.mlp_s = FFN(
            rngs=rngs,
            layers=[token_dim, token_mix_hidden_dim, token_dim],
            noise_std=noise_std,
            threshold=threshold,
            ActivationFunction=ActivationFunction
        )

        # layer norms
        self.layer_norm_C = nnx.LayerNorm(num_features=channel_dim, rngs=rngs)
        self.layer_norm_S = nnx.LayerNorm(num_features=token_dim, rngs=rngs)

    def __call__(self, x):
        """
        x: (batch, S, C) jax.Array
        """

        # apply layer norm across channels
        y = self.layer_norm_C(x) # applies layer norm across final dimention by default

        # apply token mixing MLP
        y = einops.rearrange(y, 'b s c -> b c s')
        y = self.mlp_s(y)

        # apply channel mixing
        y = einops.rearrange(y, 'b c s -> b s c')
        y = y + x # first residual connection
        y = self.layer_norm_S(y)
        y = self.mlp_c(y) + y # second residual connection

        return y



# -------------------------------------------------------
# MLPMixer
# -------------------------------------------------------
class MLPMixer(nnx.Module):
    """
    MLP Mixer Architecture
    """



