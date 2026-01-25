"""Standalone weight functions for training losses."""

from functools import partial
from typing import Callable

from jax import numpy as jnp

from diffusionlab.typing import AuxInfo, Scalar

WeightFn = Callable[[AuxInfo], Scalar]


def uniform_weight(aux: AuxInfo) -> Scalar:
    """Uniform (constant 1) weighting."""
    return jnp.ones_like(aux["t"])


def snr_weight(aux: AuxInfo) -> Scalar:
    """SNR weighting: exp(logsnr)."""
    return jnp.exp(aux["logsnr"])


def truncated_snr_weight(aux: AuxInfo, max_weight: float = 5.0) -> Scalar:
    """Truncated SNR weighting: min(exp(logsnr), max_weight)."""
    return jnp.minimum(jnp.exp(aux["logsnr"]), max_weight)


def min_snr_weight(aux: AuxInfo, gamma: float = 5.0) -> Scalar:
    """Min-SNR weighting (Hang et al. 2023): min(SNR, gamma) / SNR."""
    snr = jnp.exp(aux["logsnr"])
    return jnp.minimum(snr, gamma) / snr


def inverse_sigma_squared_weight(aux: AuxInfo) -> Scalar:
    """Inverse sigma squared weighting: 1/sigma^2."""
    return 1.0 / (aux["sigma"] ** 2)


def p2_weight(aux: AuxInfo, k: float = 1.0, gamma: float = 1.0) -> Scalar:
    """P2 weighting (Choi et al. 2022): 1 / (k + SNR)^gamma."""
    snr = jnp.exp(aux["logsnr"])
    return 1.0 / (k + snr) ** gamma


def make_weight_fn(fn: Callable, **kwargs) -> WeightFn:
    """Create a weight function with partially applied parameters.

    Args:
        fn: A weight function that takes (aux, **kwargs).
        **kwargs: Parameters to partially apply.

    Returns:
        A ``WeightFn`` with the given parameters baked in.
    """
    return partial(fn, **kwargs)
