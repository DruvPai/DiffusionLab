"""Deterministic time schedules for sampling."""

from jax import numpy as jnp

from diffusionlab.typing import Time


def uniform_schedule(num_steps: int, batch_size: int) -> list[Time]:
    """Uniform time schedule from t~=1 to t~=0.

    Args:
        num_steps: Number of sampling steps.
        batch_size: Batch dimension for each time array.

    Returns:
        List of ``num_steps + 1`` arrays of shape ``(batch_size,)``,
        linearly spaced from t~=1 to t~=0.
    """
    times = jnp.linspace(1.0, 0.0, num_steps + 1)
    return [jnp.broadcast_to(times[i], (batch_size,)) for i in range(num_steps + 1)]


def edm_schedule(
    num_steps: int,
    batch_size: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
) -> list[Time]:
    """EDM power-law time schedule (Karras et al. 2022).

    Produces a schedule of ``num_steps + 1`` time points from t=1 to t=0,
    spaced according to a power-law interpolation in sigma space::

        sigma_i = (sigma_max^(1/rho) + i/N * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

    which is then converted back to the normalised time axis via
    ``t = (sigma^(1/rho) - sigma_min^(1/rho)) / (sigma_max^(1/rho) - sigma_min^(1/rho))``.

    Args:
        num_steps: Number of sampling steps.
        batch_size: Batch dimension for each time array.
        sigma_min: Minimum noise level.
        sigma_max: Maximum noise level.
        rho: Power-law exponent (default 7.0).

    Returns:
        List of ``num_steps + 1`` arrays of shape ``(batch_size,)``
        from t~=1 to t~=0.
    """
    inv_rho = 1.0 / rho
    s_min_inv = sigma_min**inv_rho
    s_max_inv = sigma_max**inv_rho

    # Power-law spaced sigmas from sigma_max to sigma_min
    sigmas = jnp.linspace(s_max_inv, s_min_inv, num_steps + 1) ** rho

    # Convert to normalized time
    times = (sigmas**inv_rho - s_min_inv) / (s_max_inv - s_min_inv)
    return [jnp.broadcast_to(times[i], (batch_size,)) for i in range(num_steps + 1)]


__all__ = ["uniform_schedule", "edm_schedule"]
