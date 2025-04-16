from dataclasses import dataclass, field
from typing import Callable

import jax
from jax import Array, numpy as jnp


@dataclass(frozen=True)
class DiffusionProcess:
    """
    Base class for implementing various diffusion processes.

    A diffusion process defines how data evolves over time when noise is added according to
    specific dynamics operating on scalar time inputs. This class provides a framework to
    implement diffusion processes based on a schedule defined by alpha(t) and sigma(t).

    The diffusion is parameterized by two scalar functions of scalar time `t`:
    - alpha(t): Controls how much of the original signal is preserved at time `t`.
    - sigma(t): Controls how much noise is added at time `t`.

    The forward process for a single data point `x_0` is defined as:
    `x_t = alpha(t) * x_0 + sigma(t) * eps`, where:
    - `x_0` is the original data (Array[*data_dims])
    - `x_t` is the noised data at time `t` (Array[*data_dims])
    - `eps` is random noise sampled from a standard Gaussian distribution (Array[*data_dims])
    - `t` is the scalar diffusion time parameter (Array[])

    Attributes:
        alpha (Callable[[Array], Array]): Function mapping scalar time `t` (Array[]) -> scalar signal coefficient alpha(t) (Array[]).
        sigma (Callable[[Array], Array]): Function mapping scalar time `t` (Array[]) -> scalar noise coefficient sigma(t) (Array[]).
        alpha_prime (Callable[[Array], Array]): Derivative of alpha w.r.t. scalar time `t`. Maps `t` (Array[]) -> alpha'(t) (Array[]).
        sigma_prime (Callable[[Array], Array]): Derivative of sigma w.r.t. scalar time `t`. Maps `t` (Array[]) -> sigma'(t) (Array[]).
    """

    alpha: Callable[[Array], Array]
    sigma: Callable[[Array], Array]
    alpha_prime: Callable[[Array], Array] = field(init=False)
    sigma_prime: Callable[[Array], Array] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "alpha_prime", jax.grad(self.alpha))
        object.__setattr__(self, "sigma_prime", jax.grad(self.sigma))

    def forward(self, x: Array, t: Array, eps: Array) -> Array:
        """
        Applies the forward diffusion process to a data tensor `x` at time `t` using noise `eps`.

        Computes `x_t = alpha(t) * x + sigma(t) * eps`.

        Args:
            x (Array[*data_dims]): The input data tensor `x_0`.
            t (Array[]): The scalar time parameter `t`.
            eps (Array[*data_dims]): The Gaussian noise tensor `eps`, matching the shape of `x`.

        Returns:
            Array[*data_dims]: The noised data tensor `x_t` at time `t`.
        """
        alpha_t = self.alpha(t)  # Shape: []
        sigma_t = self.sigma(t)  # Shape: []
        return alpha_t * x + sigma_t * eps


@dataclass(frozen=True)
class VarianceExplodingProcess(DiffusionProcess):
    """
    Implements a Variance Exploding (VE) diffusion process.

    In this process, the signal component is constant (alpha(t) = 1), while the noise component
    increases over time according to the provided sigma(t) function. The variance of the
    noised data `x_t` explodes as `t` increases.

    Forward process: `x_t = x_0 + sigma(t) * eps`.

    This process uses:
    - alpha(t) = 1
    - sigma(t) = Provided by the user

    Attributes:
        Inherits `alpha`, `sigma`, `alpha_prime`, `sigma_prime` from `DiffusionProcess`.
        `alpha` is fixed to `lambda t: jnp.ones_like(t)`.
        `alpha_prime` is fixed to `lambda t: jnp.zeros_like(t)`.
    """

    def __init__(self, sigma: Callable[[Array], Array]):
        """
        Initialize a Variance Exploding diffusion process.

        Args:
            sigma (Callable[[Array], Array]): Function mapping scalar time `t` (Array[]) -> scalar noise coefficient sigma(t) (Array[]).
        """
        super().__init__(alpha=lambda t: jnp.ones_like(t), sigma=sigma)


@dataclass(frozen=True)
class VariancePreservingProcess(DiffusionProcess):
    """
    Implements a Variance Preserving (VP) diffusion process, often used in DDPMs.

    This process maintains the variance of the noised data `x_t` close to 1 (assuming `x_0`
    and `eps` have unit variance) throughout the diffusion by scaling the signal and noise
    components appropriately.

    Uses the following scalar dynamics:
    - alpha(t) = sqrt(1 - t²)
    - sigma(t) = t

    Forward process: `x_t = sqrt(1 - t²) * x_0 + t * eps`.

    Attributes:
        Inherits `alpha`, `sigma`, `alpha_prime`, `sigma_prime` from `DiffusionProcess`.
        `alpha` and `sigma` are set according to the VP schedule.
    """

    def __init__(self):
        """
        Initialize a Variance Preserving process with predefined scalar dynamics.
        """
        super().__init__(
            alpha=lambda t: jnp.sqrt(jnp.ones_like(t) - t**2), sigma=lambda t: t
        )


@dataclass(frozen=True)
class FlowMatchingProcess(DiffusionProcess):
    """
    Implements a diffusion process based on Flow Matching principles.

    This process defines dynamics that linearly interpolate between the data distribution
    at t=0 and a noise distribution (standard Gaussian) at t=1.

    Uses the following scalar dynamics:
    - alpha(t) = 1 - t
    - sigma(t) = t

    Forward process: `x_t = (1 - t) * x_0 + t * eps`.

    Attributes:
        Inherits `alpha`, `sigma`, `alpha_prime`, `sigma_prime` from `DiffusionProcess`.
        `alpha` and `sigma` are set to `1-t` and `t` respectively.
    """

    def __init__(self):
        """
        Initialize a Flow Matching process with predefined linear interpolation dynamics.
        """
        super().__init__(alpha=lambda t: jnp.ones_like(t) - t, sigma=lambda t: t)
