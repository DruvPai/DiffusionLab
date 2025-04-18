from dataclasses import dataclass
from typing import Tuple, cast
from jax import numpy as jnp, Array
import jax
from diffusionlab.distributions.base import Distribution
from diffusionlab.distributions.gmm.gmm import GMM
from diffusionlab.distributions.gmm.utils import create_gmm_vector_field_fns
from diffusionlab.dynamics import DiffusionProcess


@dataclass(frozen=True)
class IsoGMM(Distribution):
    """
    Implements an isotropic Gaussian Mixture Model (GMM) distribution.

    The probability measure is given by:

    ``μ(A) = sum_{i=1}^{num_components} priors[i] * N(A; means[i], variances[i] * I)``

    This class provides methods for sampling from the GMM and computing various vector fields (``VectorFieldType.SCORE``, ``VectorFieldType.X0``, ``VectorFieldType.EPS``, ``VectorFieldType.V``) related to the distribution under a given diffusion process.

    Attributes:
        dist_params (``Dict[str, Array]``): Dictionary containing the core GMM parameters.

            - ``means`` (``Array[num_components, data_dim]``):  The means of the GMM components.
            - ``variances`` (``Array[num_components]``): The variances of the GMM components.
            - ``priors`` (``Array[num_components]``): The prior probabilities (mixture weights) of the GMM components.

        dist_hparams (``Dict[str, Any]``): Dictionary for storing hyperparameters (currently unused).
    """

    def __init__(self, means: Array, variances: Array, priors: Array):
        """
        Initializes the isotropic GMM distribution.

        Args:
            means (``Array[num_components, data_dim]``): Means for each Gaussian component.
            variances (``Array[num_components]``): Variance for each Gaussian component.
            priors (``Array[num_components]``): Mixture weights for each component. Must sum to 1.
        """
        eps = cast(float, jnp.finfo(variances.dtype).eps)
        assert means.ndim == 2
        num_components, data_dim = means.shape
        assert variances.shape == (num_components,)
        assert priors.shape == (num_components,)
        assert jnp.isclose(jnp.sum(priors), 1.0, atol=eps)
        assert jnp.all(variances >= -eps)

        super().__init__(
            dist_params={
                "means": means,
                "variances": variances,
                "priors": priors,
            },
            dist_hparams={},
        )

    def sample(self, key: Array, num_samples: int) -> Tuple[Array, Array]:
        """
        Draws samples from the isotropic GMM distribution.

        Args:
            key (``Array``): JAX PRNG key for random sampling.
            num_samples (``int``): The total number of samples to generate.

        Returns:
            ``Tuple[Array[num_samples, data_dim], Array[num_samples]]``: A tuple ``(samples, component_indices)`` containing the drawn samples and the index of the GMM component from which each sample was drawn.
        """
        data_dim = self.dist_params["means"].shape[1]
        covs = jax.vmap(lambda variance: variance * jnp.eye(data_dim))(
            self.dist_params["variances"]
        )
        base_gmm = GMM(self.dist_params["means"], covs, self.dist_params["priors"])
        return base_gmm.sample(key, num_samples)

    def score(self, x_t: Array, t: Array, diffusion_process: DiffusionProcess) -> Array:
        """
        Computes the score vector field ``∇_x log p_t(x_t)`` for the isotropic GMM distribution.

        This is calculated with respect to the perturbed distribution ``p_t`` induced by the
        ``diffusion_process`` at time ``t``.

        Args:
            x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step (scalar).
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[data_dim]``: The score vector field evaluated at ``x_t`` and ``t``.
        """
        return iso_gmm_score(
            x_t,
            t,
            diffusion_process,
            self.dist_params["means"],
            self.dist_params["variances"],
            self.dist_params["priors"],
        )

    def x0(self, x_t: Array, t: Array, diffusion_process: DiffusionProcess) -> Array:
        """
        Computes the denoised prediction ``x0 = E[x_0 | x_t]`` for the isotropic GMM distribution.

        This represents the expected original sample ``x_0`` given the noisy observation ``x_t``
        at time ``t`` under the ``diffusion_process``.

        Args:
            x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step (scalar).
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[data_dim]``: The denoised prediction vector field ``x0`` evaluated at ``x_t`` and ``t``.
        """
        return iso_gmm_x0(
            x_t,
            t,
            diffusion_process,
            self.dist_params["means"],
            self.dist_params["variances"],
            self.dist_params["priors"],
        )

    def eps(self, x_t: Array, t: Array, diffusion_process: DiffusionProcess) -> Array:
        """
        Computes the noise prediction ``ε`` for the isotropic GMM distribution.

        This predicts the noise that was added to the original sample ``x_0`` to obtain ``x_t``
        at time ``t`` under the ``diffusion_process``.

        Args:
            x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step (scalar).
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[data_dim]``: The noise prediction vector field ``ε`` evaluated at ``x_t`` and ``t``.
        """
        return iso_gmm_eps(
            x_t,
            t,
            diffusion_process,
            self.dist_params["means"],
            self.dist_params["variances"],
            self.dist_params["priors"],
        )

    def v(self, x_t: Array, t: Array, diffusion_process: DiffusionProcess) -> Array:
        """
        Computes the velocity vector field ``v`` for the isotropic GMM distribution.

        This is conditional velocity ``E[dx_t/dt | x_t]`` under the ``diffusion_process``.

        Args:
            x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step (scalar).
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[data_dim]``: The velocity vector field ``v`` evaluated at ``x_t`` and ``t``.
        """
        return iso_gmm_v(
            x_t,
            t,
            diffusion_process,
            self.dist_params["means"],
            self.dist_params["variances"],
            self.dist_params["priors"],
        )


def iso_gmm_x0(
    x_t: Array,
    t: Array,
    diffusion_process: DiffusionProcess,
    means: Array,
    variances: Array,
    priors: Array,
) -> Array:
    """
    Computes the denoised prediction ``x0 = E[x_0 | x_t]`` for a GMM.

    This implements the closed-form solution for the conditional expectation
    ``E[x_0 | x_t]`` where ``x_t ~ N(α_t x_0, σ_t^2 I)`` and ``x_0`` follows the GMM distribution
    defined by ``means``, ``covs``, and ``priors``.

    Args:
        x_t (``Array[data_dim]``): The noisy state tensor at time ``t``.
        t (``Array[]``): The time step (scalar).
        diffusion_process (``DiffusionProcess``): Provides ``α(t)`` and ``σ(t)``.
        means (``Array[num_components, data_dim]``): Means of the GMM components.
        variances (``Array[num_components]``): Covariances of the GMM components.
        priors (``Array[num_components]``): Mixture weights of the GMM components.

    Returns:
        ``Array[data_dim]``: The denoised prediction ``x0`` evaluated at ``x_t`` and ``t``.
    """
    num_components, data_dim = means.shape
    alpha_t = diffusion_process.alpha(t)
    sigma_t = diffusion_process.sigma(t)

    means_t = jax.vmap(lambda mean: alpha_t * mean)(means)  # (num_components, data_dim)
    variances_t = jax.vmap(lambda variance: alpha_t**2 * variance + sigma_t**2)(
        variances
    )  # (num_components,)

    xbars_t = jax.vmap(lambda mean_t: x_t - mean_t)(
        means_t
    )  # (num_components, data_dim)
    variances_t_inv_xbars_t = jax.vmap(lambda variance_t, xbar_t: xbar_t / variance_t)(
        variances_t, xbars_t
    )  # (num_components, data_dim)

    log_likelihoods_unnormalized = jax.vmap(
        lambda xbar_t, variance_t, variance_t_inv_xbar_t: -0.5
        * (jnp.sum(xbar_t * variance_t_inv_xbar_t) + data_dim * jnp.log(variance_t))
    )(xbars_t, variances_t, variances_t_inv_xbars_t)  # (num_components,)
    log_posterior_unnormalized = (
        jnp.log(priors) + log_likelihoods_unnormalized
    )  # (num_components,)
    posterior_probs = jax.nn.softmax(
        log_posterior_unnormalized, axis=0
    )  # (num_components,) sum to 1

    posterior_means = jax.vmap(
        lambda mean, variance, variance_t_inv_xbar_t: mean
        + alpha_t * variance * variance_t_inv_xbar_t
    )(means, variances, variances_t_inv_xbars_t)  # (num_components, data_dim)

    x0_pred = jnp.sum(posterior_probs[:, None] * posterior_means, axis=0)  # (data_dim,)

    return x0_pred


# Generate eps, score, v functions from iso_gmm_x0
iso_gmm_eps, iso_gmm_score, iso_gmm_v = create_gmm_vector_field_fns(iso_gmm_x0)
