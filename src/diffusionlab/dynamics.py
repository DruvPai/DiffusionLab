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
    implement diffusion processes based on a schedule defined by ``α(t)`` and ``σ(t)``.

    The diffusion is parameterized by two scalar functions of scalar time ``t``:

    - ``α(t)``: Controls how much of the original signal is preserved at time ``t``.
    - ``σ(t)``: Controls how much noise is added at time ``t``.

    The forward process for a single data point ``x_0`` is defined as:

    ``x_t = α(t) * x_0 + σ(t) * ε``

    where:

    - ``x_0`` is the original data (``Array[*data_dims]``)
    - ``x_t`` is the noised data at time ``t`` (``Array[*data_dims]``)
    - ``ε`` is random noise sampled from a standard Gaussian distribution (``Array[*data_dims]``)
    - ``t`` is the scalar diffusion time parameter (``Array[]``)

    Attributes:
        alpha (``Callable[[Array[]], Array[]]``): Function mapping scalar time ``t`` -> scalar signal coefficient ``α(t)``.
        sigma (``Callable[[Array[]], Array[]]``): Function mapping scalar time ``t`` -> scalar noise coefficient ``σ(t)``.
        alpha_prime (``Callable[[Array[]], Array[]]``): Derivative of ``α`` w.r.t. scalar time ``t``.
        sigma_prime (``Callable[[Array[]], Array[]]``): Derivative of ``σ`` w.r.t. scalar time ``t``.
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
        Applies the forward diffusion process to a data tensor ``x`` at time ``t`` using noise ``ε``.

        Computes ``x_t = α(t) * x + σ(t) * ε``.

        Args:
            x (``Array[*data_dims]``): The input data tensor ``x_0``.
            t (``Array[]``): The scalar time parameter ``t``.
            eps (``Array[*data_dims]``): The Gaussian noise tensor ``ε``, matching the shape of ``x``.

        Returns:
            ``Array[*data_dims]``: The noised data tensor ``x_t`` at time ``t``.
        """
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        return alpha_t * x + sigma_t * eps


@dataclass(frozen=True)
class VarianceExplodingProcess(DiffusionProcess):
    """
    Implements a Variance Exploding (VE) diffusion process.

    In this process, the signal component is constant (``α(t) = 1``), while the noise component
    increases over time according to the provided ``σ(t)`` function. The variance of the
    noised data ``x_t`` explodes as ``t`` increases.

    Forward process:

    ``x_t = x_0 + σ(t) * ε``.

    This process uses:

    - ``α(t) = 1``
    - ``σ(t) =`` Provided by the user

    Attributes:
        alpha (``Callable[[Array[]], Array[]]``): Function mapping scalar time ``t`` -> scalar signal coefficient ``α(t)``. Set to 1.
        sigma (``Callable[[Array[]], Array[]]``): Function mapping scalar time ``t`` -> scalar noise coefficient ``σ(t)``. Provided by the user.
        alpha_prime (``Callable[[Array[]], Array[]]``): Derivative of ``α`` w.r.t. scalar time ``t``. Set to 0.
        sigma_prime (``Callable[[Array[]], Array[]]``): Derivative of ``σ`` w.r.t. scalar time ``t``.
    """

    def __init__(self, sigma: Callable[[Array], Array]):
        """
        Initialize a Variance Exploding diffusion process.

        Args:
            sigma (``Callable[[Array], Array]``): Function mapping scalar time ``t`` -> scalar noise coefficient ``σ(t)``.
        """
        super().__init__(alpha=lambda t: jnp.ones_like(t), sigma=sigma)


@dataclass(frozen=True)
class VariancePreservingProcess(DiffusionProcess):
    """
    Implements a Variance Preserving (VP) diffusion process, often used in DDPMs.

    This process maintains the variance of the noised data ``x_t`` close to 1 (assuming ``x_0``
    and ``ε`` have unit variance) throughout the diffusion by scaling the signal and noise
    components appropriately.

    Uses the following scalar dynamics:

    - ``α(t) = sqrt(1 - t²)``
    - ``σ(t) = t``

    Forward process:

    ``x_t = sqrt(1 - t²) * x_0 + t * ε``.

    Attributes:
        alpha (``Callable[[Array[]], Array[]]``): Function mapping scalar time ``t`` -> scalar signal coefficient ``α(t)``. Set to ``sqrt(1 - t²)``.
        sigma (``Callable[[Array[]], Array[]]``): Function mapping scalar time ``t`` -> scalar noise coefficient ``σ(t)``. Set to ``t``.
        alpha_prime (``Callable[[Array[]], Array[]]``): Derivative of ``α`` w.r.t. scalar time ``t``. Set to ``-t / sqrt(1 - t²)``.
        sigma_prime (``Callable[[Array[]], Array[]]``): Derivative of ``σ`` w.r.t. scalar time ``t``. Set to ``1``.
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
    at ``t=0`` and a noise distribution (standard Gaussian) at ``t=1``.

    Uses the following scalar dynamics:

    - ``α(t) = 1 - t``
    - ``σ(t) = t``

    Forward process:

    ``x_t = (1 - t) * x_0 + t * ε``.

    Attributes:
        alpha (``Callable[[Array[]], Array[]]``): Function mapping scalar time ``t`` -> scalar signal coefficient ``α(t)``. Set to ``1 - t``.
        sigma (``Callable[[Array[]], Array[]]``): Function mapping scalar time ``t`` -> scalar noise coefficient ``σ(t)``. Set to ``t``.
        alpha_prime (``Callable[[Array[]], Array[]]``): Derivative of ``α`` w.r.t. scalar time ``t``. Set to ``-1``.
        sigma_prime (``Callable[[Array[]], Array[]]``): Derivative of ``σ`` w.r.t. scalar time ``t``. Set to ``1``.
    """

    def __init__(self):
        """
        Initialize a Flow Matching process with predefined linear interpolation dynamics.
        """
        super().__init__(alpha=lambda t: jnp.ones_like(t) - t, sigma=lambda t: t)
