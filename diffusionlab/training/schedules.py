"""Stochastic time schedules for training."""

from typing import Callable

import jax
from jax import numpy as jnp

from diffusionlab.typing import PRNGKey, Time


def uniform_time(key: PRNGKey, batch_size: int) -> Time:
    """Sample times uniformly from U[0, 1].

    Args:
        key: PRNG key.
        batch_size: Number of times to sample.

    Returns:
        Array of shape ``(batch_size,)`` with values in [0, 1].
    """
    return jax.random.uniform(key, (batch_size,))


def uniform_time_with_eps(key: PRNGKey, batch_size: int, eps: float = 1e-5) -> Time:
    """Sample times uniformly from U[eps, 1-eps].

    Args:
        key: PRNG key.
        batch_size: Number of times to sample.
        eps: Small offset from boundaries. Default 1e-5.

    Returns:
        Array of shape ``(batch_size,)`` with values in [eps, 1-eps].
    """
    return jax.random.uniform(key, (batch_size,), minval=eps, maxval=1.0 - eps)


def logit_normal_time(
    key: PRNGKey,
    batch_size: int,
    mean: float = 0.0,
    std: float = 1.0,
) -> Time:
    """Sample times via logit-normal distribution (SD3 / rectified flow).

    Draws z ~ N(mean, std^2) and returns sigmoid(z).

    Args:
        key: PRNG key.
        batch_size: Number of times to sample.
        mean: Mean of the underlying normal distribution.
        std: Standard deviation of the underlying normal distribution.

    Returns:
        Array of shape ``(batch_size,)`` with values in (0, 1).
    """
    z = jax.random.normal(key, (batch_size,)) * std + mean
    return jax.nn.sigmoid(z)


def logsnr_uniform_time(
    key: PRNGKey,
    batch_size: int,
    logsnr_inverse: Callable,
    logsnr_min: float = -10.0,
    logsnr_max: float = 10.0,
) -> Time:
    """Sample times uniformly in log-SNR space.

    Draws uniform samples in [logsnr_min, logsnr_max] and maps back to
    the time domain via ``logsnr_inverse``.

    Args:
        key: PRNG key.
        batch_size: Number of times to sample.
        logsnr_inverse: Callable mapping log-SNR values to times.
        logsnr_min: Minimum log-SNR value.
        logsnr_max: Maximum log-SNR value.

    Returns:
        Array of shape ``(batch_size,)`` with valid time values.
    """
    lam = jax.random.uniform(key, (batch_size,), minval=logsnr_min, maxval=logsnr_max)
    return logsnr_inverse(lam)


def stratified_time(key: PRNGKey, batch_size: int) -> Time:
    """Stratified time sampling for variance reduction.

    Divides [0, 1] into ``batch_size`` equal strata and samples one
    point uniformly from each stratum.

    Args:
        key: PRNG key.
        batch_size: Number of times to sample.

    Returns:
        Array of shape ``(batch_size,)`` with stratified values in [0, 1].
    """
    offsets = jax.random.uniform(key, (batch_size,))
    bins = jnp.arange(batch_size, dtype=jnp.float32)
    return (bins + offsets) / batch_size


__all__ = [
    "uniform_time",
    "uniform_time_with_eps",
    "logit_normal_time",
    "logsnr_uniform_time",
    "stratified_time",
]
