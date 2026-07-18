"""
MLP Mixer Architecture
"""
import jax
import jax.numpy as jnp
import flax
from flax import nnx
from FFN import FFN
from DualSampleTernary import DualSampleTernary
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
                 num_patches: int, # S
                 patch_hidden_dim: int, # C
                 mlp_c_hidden_size: int, # Dc
                 mlp_t_hidden_size: int, # Ds
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
            layers=[patch_hidden_dim, mlp_c_hidden_size, patch_hidden_dim],
            noise_std=noise_std,
            threshold=threshold,
            ActivationFunction=ActivationFunction
        )

        # Define token mixing MLP
        self.mlp_s = FFN(
            rngs=rngs,
            layers=[num_patches, mlp_t_hidden_size, num_patches],
            noise_std=noise_std,
            threshold=threshold,
            ActivationFunction=ActivationFunction
        )

        # layer norms
        self.layer_norm_C = nnx.LayerNorm(num_features=patch_hidden_dim, rngs=rngs)
        # self.layer_norm_S = nnx.LayerNorm(num_features=num_patches, rngs=rngs)

    def __call__(self, x):
        """
        x: (batch, S, C) jax.Array
        """

        # apply layer norm across channels
        y = self.layer_norm_C(x) # applies layer norm across final dimension by default

        # apply token mixing MLP
        y = einops.rearrange(y, 'b s c -> b c s')
        y = self.mlp_s(y)
        # print(f"MLP S output shape: {y.shape}")

        # apply channel mixing
        y = einops.rearrange(y, 'b c s -> b s c')
        x = y + x # first residual connection
        y = self.layer_norm_C(x)
        y = self.mlp_c(y) + y # second residual connection
        # print(f"MLP C output shape: {y.shape}")

        return y



# -------------------------------------------------------
# MLPMixer
# -------------------------------------------------------
class MLPMixer(nnx.Module):
    """
    MLP Mixer Architecture
    """

    def __init__(self,
                 num_mlp_mix_blocks: int,
                 num_input_channels: int, # typically 3 for RGB
                 patch_size: int, # size of the patch. S = HW/P^2
                 patch_hidden_dim: int, # C
                 mlp_c_hidden_size: int, # Dc
                 mlp_t_hidden_size: int, # Ds
                 ActivationFunction: nnx.Module,
                 rngs: nnx.Rngs,
                 threshold: float = 0.0,
                 noise_std: float = 0.1,
                 height_pixels: int = 32, # CIFAR -> change as needed
                 width_pixels: int = 32, # CIFAR -> change as needed
                 ):

        # initialize the per-patch feedforward network using conv
        self.conv1 = nnx.Conv(in_features=num_input_channels, out_features=patch_hidden_dim, kernel_size=(patch_size, patch_size), strides=(patch_size, patch_size), rngs=rngs)

        # compute number of patches
        num_patches = int(height_pixels*width_pixels//(patch_size**2))

        # initialize list of MNPMixer blocks
        self.mlp_mixers = nnx.List([
            MixerBlock(
                num_patches=num_patches,
                patch_hidden_dim=patch_hidden_dim,
                mlp_c_hidden_size=mlp_c_hidden_size,
                mlp_t_hidden_size=mlp_t_hidden_size,
                ActivationFunction=ActivationFunction,
                rngs=rngs,
                threshold=threshold,
                noise_std=noise_std
            ) for _ in range(num_mlp_mix_blocks)
        ])

    def __call__(self, x):
        """
        x: (H, W, C) -> for CIFAR (32, 32, 3)
        """
        
        # Per-patch processing
        x = self.conv1(x)
        # print(f"Conv1 output shape: {x.shape}")

        # contract h,w dimensions into a S dimension
        x = einops.rearrange(x, 'b h w c -> b (h w) c')
        # print(f"Rearranged output shape: {x.shape}")

        # apply the mixer blocks
        for i, mixer in enumerate(self.mlp_mixers):
            x = mixer(x)
            # print(f"Output shape after mixer block {i}: {x.shape}")

        return x
        

# -------------------------------------------------------
# TESTING
# -------------------------------------------------------
def main():
    key = jax.random.key(0)
    x_test = jax.random.normal(key, (10, 32, 32, 3)) # (B, H, W, C)
    print(f"Input shape: {x_test.shape}")

    mixer = MLPMixer(
        num_mlp_mix_blocks=2,
        num_input_channels=x_test.shape[-1],
        patch_size=4,
        patch_hidden_dim=512,
        mlp_c_hidden_size=2048,
        mlp_t_hidden_size=256,
        height_pixels=x_test.shape[1],
        width_pixels=x_test.shape[2],
        ActivationFunction=DualSampleTernary,
        rngs=nnx.Rngs(0)
    )

    nnx.display(mixer)

    y = mixer(x_test)
    print(f"Output shape: {y.shape}")


if __name__ == "__main__":
    main()