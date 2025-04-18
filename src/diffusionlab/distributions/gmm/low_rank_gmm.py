from typing import Tuple, cast
from dataclasses import dataclass
from jax import numpy as jnp, Array
import jax
from diffusionlab.distributions.base import Distribution
from diffusionlab.distributions.gmm.gmm import GMM
from diffusionlab.distributions.gmm.utils import (
    _logdeth,
    _lstsq,
    create_gmm_vector_field_fns,
)
from diffusionlab.dynamics import DiffusionProcess


@dataclass(frozen=True)
class LowRankGMM(Distribution):
    """
    Implements a low-rank Gaussian Mixture Model (GMM) distribution.

    The probability measure is given by:

    ``μ(A) = sum_{i=1}^{num_components} priors[i] * N(A; means[i], cov_factors[i] @ cov_factors[i].T)``

    This class provides methods for sampling from the GMM and computing various vector fields (``VectorFieldType.SCORE``, ``VectorFieldType.X0``, ``VectorFieldType.EPS``, ``VectorFieldType.V``) related to the distribution under a given diffusion process.

    Attributes:
        dist_params (``Dict[str, Array]``): Dictionary containing the core low-rank GMM parameters.

            - ``means`` (``Array[num_components, data_dim]``): The means of the GMM components.
            - ``cov_factors`` (``Array[num_components, data_dim, rank]``): The low-rank covariance matrix factors of the GMM components.
            - ``priors`` (``Array[num_components]``): The prior probabilities (mixture weights) of the GMM components.

        dist_hparams (``Dict[str, Any]``): Dictionary for storing hyperparameters (currently unused).
    """

    def __init__(self, means: Array, cov_factors: Array, priors: Array):
        """
        Initializes the low-rank GMM distribution.

        Args:
            means (``Array[num_components, data_dim]``): Means for each Gaussian component.
            cov_factors (``Array[num_components, data_dim, rank]``): Low-rank covariance matrices for each Gaussian component.
            priors (``Array[num_components]``): Mixture weights for each component. Must sum to 1.
        """
        eps = cast(float, jnp.finfo(cov_factors.dtype).eps)
        assert means.ndim == 2
        num_components, data_dim, rank = cov_factors.shape
        assert means.shape == (num_components, data_dim)
        assert priors.shape == (num_components,)
        assert jnp.isclose(jnp.sum(priors), 1.0, atol=eps)

        super().__init__(
            dist_params={
                "means": means,
                "cov_factors": cov_factors,
                "priors": priors,
            },
            dist_hparams={},
        )

    def sample(self, key: Array, num_samples: int) -> Tuple[Array, Array]:
        """
        Draws samples from the low-rank GMM distribution.

        Args:
            key (``Array``): JAX PRNG key for random sampling.
            num_samples (``int``): The total number of samples to generate.

        Returns:
            ``Tuple[Array[num_samples, data_dim], Array[num_samples]]``: A tuple ``(samples, component_indices)`` containing the drawn samples and the index of the GMM component from which each sample was drawn.
        """
        covs = jax.vmap(
            lambda low_rank_cov_factor: low_rank_cov_factor @ low_rank_cov_factor.T
        )(self.dist_params["cov_factors"])
        return GMM(self.dist_params["means"], covs, self.dist_params["priors"]).sample(
            key, num_samples
        )

    def score(self, x_t: Array, t: Array, diffusion_process: DiffusionProcess) -> Array:
        """
        Computes the score vector field ``∇_x log p_t(x_t)`` for the low-rank GMM distribution.

        This is calculated with respect to the perturbed distribution ``p_t`` induced by the
        ``diffusion_process`` at time ``t``.

        Args:
            x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step (scalar).
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[data_dim]``: The score vector field evaluated at ``x_t`` and ``t``.
        """
        return low_rank_gmm_score(
            x_t,
            t,
            diffusion_process,
            self.dist_params["means"],
            self.dist_params["cov_factors"],
            self.dist_params["priors"],
        )

    def x0(self, x_t: Array, t: Array, diffusion_process: DiffusionProcess) -> Array:
        """
        Computes the denoised prediction x0 = E[x_0 | x_t] for the low-rank GMM distribution.

        This represents the expected original sample ``x_0`` given the noisy observation ``x_t``
        at time ``t`` under the ``diffusion_process``.

        Args:
            x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step (scalar).
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[data_dim]``: The denoised prediction vector field ``x0`` evaluated at ``x_t`` and ``t``.
        """
        return low_rank_gmm_x0(
            x_t,
            t,
            diffusion_process,
            self.dist_params["means"],
            self.dist_params["cov_factors"],
            self.dist_params["priors"],
        )

    def eps(self, x_t: Array, t: Array, diffusion_process: DiffusionProcess) -> Array:
        """
        Computes the noise prediction ε for the low-rank GMM distribution.

        This predicts the noise that was added to the original sample ``x_0`` to obtain ``x_t``
        at time ``t`` under the ``diffusion_process``.

        Args:
            x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step (scalar).
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[data_dim]``: The noise prediction vector field ``ε`` evaluated at ``x_t`` and ``t``.
        """
        return low_rank_gmm_eps(
            x_t,
            t,
            diffusion_process,
            self.dist_params["means"],
            self.dist_params["cov_factors"],
            self.dist_params["priors"],
        )

    def v(self, x_t: Array, t: Array, diffusion_process: DiffusionProcess) -> Array:
        """
        Computes the velocity vector field ``v`` for the low-rank GMM distribution.

        This is the conditional velocity ``E[dx_t/dt | x_t]`` under the ``diffusion_process``.

        Args:
            x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step (scalar).
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[data_dim]``: The velocity vector field ``v`` evaluated at ``x_t`` and ``t``.
        """
        return low_rank_gmm_v(
            x_t,
            t,
            diffusion_process,
            self.dist_params["means"],
            self.dist_params["cov_factors"],
            self.dist_params["priors"],
        )


def low_rank_gmm_x0(
    x_t: Array,
    t: Array,
    diffusion_process: DiffusionProcess,
    means: Array,
    cov_factors: Array,
    priors: Array,
) -> Array:
    """
    Computes the denoised prediction ``x0 = E[x_0 | x_t]`` for a low-rank GMM.

    This implements the closed-form solution for the conditional expectation
    ``E[x_0 | x_t]`` where ``x_t ~ N(α_t x_0, σ_t^2 I)`` and ``x_0`` follows the low-rank GMM distribution
    defined by ``means``, ``cov_factors``, and ``priors``.

    Args:
        x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
        t (``Array[]``): The time step (scalar).
        diffusion_process (``DiffusionProcess``): Provides ``α(t)`` and ``σ(t)``.
        means (``Array[num_components, data_dim]``): Means of the GMM components.
        cov_factors (``Array[num_components, data_dim, rank]``): Low-rank covariance matrices of the GMM components.
        priors (``Array[num_components]``): Mixture weights of the GMM components.

    Returns:
        ``Array[data_dim]``: The denoised prediction ``x0`` evaluated at ``x_t`` and ``t``.
    """
    num_components, data_dim, rank = cov_factors.shape
    alpha_t = diffusion_process.alpha(t)
    sigma_t = diffusion_process.sigma(t)

    means_t = jax.vmap(lambda mean: alpha_t * mean)(means)  # (num_components, data_dim)
    inner_covs = jax.vmap(lambda cov_factor: cov_factor.T @ cov_factor)(
        cov_factors
    )  # (num_components, rank, rank)

    xbars_t = jax.vmap(lambda mean_t: x_t - mean_t)(
        means_t
    )  # (num_components, data_dim)
    covs_t_inverse_xbars_t = jax.vmap(
        lambda cov_factor, inner_cov, xbar_t: (1 / sigma_t**2)
        * (
            xbar_t
            - cov_factor
            @ _lstsq(
                inner_cov + (sigma_t / alpha_t) ** 2 * jnp.eye(rank),
                cov_factor.T @ xbar_t,
            )
        )
    )(cov_factors, inner_covs, xbars_t)  # (num_components, data_dim)

    logdets_covs_t = jax.vmap(
        lambda inner_cov: _logdeth((alpha_t / sigma_t) ** 2 * inner_cov + jnp.eye(rank))
    )(inner_covs) + 2 * data_dim * jnp.log(sigma_t)  # (num_components,)

    log_likelihoods_unnormalized = jax.vmap(
        lambda xbar_t, covs_t_inverse_xbar_t, logdet_covs_t: -(1 / 2)
        * (jnp.sum(xbar_t * covs_t_inverse_xbar_t) + logdet_covs_t)
    )(xbars_t, covs_t_inverse_xbars_t, logdets_covs_t)  # (num_components,)

    log_posterior_unnormalized = (
        jnp.log(priors) + log_likelihoods_unnormalized
    )  # (num_components,)

    posterior_probs = jax.nn.softmax(
        log_posterior_unnormalized, axis=0
    )  # (num_components,)

    posterior_means = jax.vmap(
        lambda mean, cov_factor, covs_t_inverse_xbar_t: mean
        + alpha_t * cov_factor @ (cov_factor.T @ covs_t_inverse_xbar_t)
    )(means, cov_factors, covs_t_inverse_xbars_t)  # (num_components, data_dim)

    x0_pred = jnp.sum(posterior_probs[:, None] * posterior_means, axis=0)  # (data_dim,)

    return x0_pred


# Generate eps, score, v functions from low_rank_gmm_x0
low_rank_gmm_eps, low_rank_gmm_score, low_rank_gmm_v = create_gmm_vector_field_fns(
    low_rank_gmm_x0
)
