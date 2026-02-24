"""
Forward and backward pass for ternary stochastic activation.

"""

import jax
import jax.numpy as jnp

# define gaussian pdf and cdf functions
def gaussian_pdf(x: float,
                 mean: float,
                 std: float) -> float:
    """Gaussian Probability density function."""

    pd = jax.scipy.stats.norm.pdf(x=x, loc=mean, scale=std)
    return pd

def gaussian_cdf(
        x: float,
        mean: float,
        std: float
):
    "Gaussian CDF"
    cdf = jax.scipy.stats.norm.cdf(x=x, loc=mean, scale=std)
    return cdf


# define the ternary activation
@jax.custom_vjp
def ternary_activation(
        x: float,
        thresholds: jnp.ndarray,
        levels: jnp.ndarray,
        key: jax.random.key,
        noise_std: float, 
        noise_mean: float = 0.0
):
    # generate noise
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(key) * noise_std + noise_mean

    # add noise to the input
    x = x + noise # mimics input referreed noise of a comparator

    # ternary activation
    s = jnp.where(
        x < thresholds[0], levels[0],
        jnp.where(
            x > thresholds[1], levels[2],
            levels[1]
        )
    )

    return s

def ternary_activation_fwd(
    x: float,
    thresholds: list[float],
    levels: list[float],
    key: jax.random.key,
    noise_std: float,
    noise_mean: float = 0.0
):
    # return the primal function in the forward mode
    y = ternary_activation(x, thresholds, levels, key, noise_std, noise_mean)
    return y, (x, thresholds, levels, key, noise_std, noise_mean) # return the primal function and the auxiliary data for the backward pass

def ternary_activation_bwd(residuals, gradients):
    # unpack the residuals
    x, thresholds, levels, key, noise_std, noise_mean = residuals

    # gradient w.r.t. x
    t_low, t_high = thresholds
    l_low, l_mid, l_high = levels # assuming that the mid-level is zero

    dx = gradients * (l_high * gaussian_pdf(x=t_high-x, mean=0, std=noise_std) - l_low * gaussian_pdf(x=t_low-x, mean=0, std=noise_std))

    # gradient w.r.t. thresholds
    dt_high = - gradients * l_high * gaussian_pdf(x=t_high-x, mean=0, std=noise_std)
    dt_low = gradients * l_low * gaussian_pdf(x=t_low-x, mean=0, std=noise_std)
    d_thresholds = jnp.array([jnp.sum(dt_low), jnp.sum(dt_high)])

    return (
        dx, # gradient w.r.t. input
        d_thresholds, # gradient w.r.t thresholds
        None, # no gradient w.r.t. levels
        None, # no gradient w,r.t. key
        None, # no gradient w.r.t. noise_std
        None # no gradient w.r.t. noise_mean
    )

# bind the forward and backward passes together
ternary_activation.defvjp(ternary_activation_fwd, ternary_activation_bwd)

def expected_state(
        x: float,
        thresholds: list[float],
        levels: list[float],
        noise_std: float,
        noise_mean: float = 0.0
):
    """Ternary expected state for gaussian input-referred noise.
    Caveat: levels[1] = 0
    """

    # compute probability for low and high states
    p_low = gaussian_cdf(x=thresholds[0] - x, mean=0, std=noise_std)
    p_high = 1 - gaussian_cdf(x=thresholds[1]-x, mean=0, std=noise_std)
    #ADD a p_mid if mid-level is not zero

    exp_state = levels[2]*p_high + levels[0]*p_low # + levels[1]*(1-p_low-p_high) if there is a non-zero mid-level
    return exp_state

    