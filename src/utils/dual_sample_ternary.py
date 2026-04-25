import jax
import jax.numpy as jnp

## defining the activation function
@jax.custom_vjp
def dual_sample_ternary(
        x: float,
        key: jax.random.key,
        threshold: float = 0.0,
        noise_std: float = 1.0,
        noise_mean: float = 0.0,
    ):

    # generate independent noise samples
    key1, key2, subkey = jax.random.split(key, 3)
    noise1 = jax.random.normal(key1)*noise_std + noise_mean
    noise2 = jax.random.normal(key2)*noise_std + noise_mean

    # perturb the inputs with noise
    s1 = x + noise1
    s2 = x + noise2

    # find the threshold crossing for both samples
    s1 = s1 > threshold
    s2 = s2 > threshold

    # determine the sign
    sign = jnp.where(
        jnp.logical_and(s1, s2), 1.0,
        jnp.where(
            jnp.logical_and(jnp.logical_not(s1), jnp.logical_not(s2)), -1.0,
            0.0
        )
    )

    # determine the output magnitude
    mag = jnp.logical_not(jnp.logical_xor(s1, s2)).astype(jnp.float32)

    return sign * mag

def dual_sample_ternary_fwd(
    x: float,
    key: jax.random.key,
    threshold: float = 0.0,
    noise_std: float = 1.0,
    noise_mean: float = 0.0,
    **kwargs
):
    y = dual_sample_ternary(
        x,
        key,
        threshold=threshold,
        noise_std=noise_std,
        noise_mean=noise_mean,
        **kwargs
    )
    return y, (y, x, key, threshold, noise_std, noise_mean)

def dual_sample_ternary_bwd(residuals, gradients):
    y, x, key, threshold, noise_std, noise_mean = residuals

    dx = gradients * (1 - jnp.square(y)) # gradient is 0 when output is 1 or -1, else 1

    return (dx, None, None, None, None)
    
# bind the forward and backward functions
dual_sample_ternary.defvjp(dual_sample_ternary_fwd, dual_sample_ternary_bwd)

    

# defining the expected state
def expected_state(
        x: float,
        threshold: float = 0.0,
        noise_std: float = 1.0,
        noise_mean: float = 0.0,
        **kwargs
    ):

    p = jax.scipy.stats.norm.cdf(x=threshold, loc=x, scale=noise_std)

    return 1 - 2*p
