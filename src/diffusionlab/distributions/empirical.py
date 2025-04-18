from dataclasses import dataclass
from typing import Iterable, Tuple, cast

import jax
from jax import Array, numpy as jnp

from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.distributions.base import Distribution
from diffusionlab.vector_fields import VectorFieldType, convert_vector_field_type


@dataclass(frozen=True)
class EmpiricalDistribution(Distribution):
    """
    An empirical distribution, i.e., the uniform distribution over a dataset.
    The probability measure is defined as:

    ``μ(A) = (1/N) * sum_{i=1}^{num_samples} delta(x_i in A)``

    where ``x_i`` is the ith data point in the dataset, and ``N`` is the number of data points.

    This class provides methods for sampling from the empirical distribution and computing various vector fields (``VectorFieldType.SCORE``, ``VectorFieldType.X0``, ``VectorFieldType.EPS``, ``VectorFieldType.V``) related to the distribution under a given diffusion process.

    Attributes:
        dist_params (``Dict[str, Array]``): Dictionary containing distribution parameters (currently unused).
        dist_hparams (``Dict[str, Any]``): Dictionary for storing hyperparameters. It may contain the following keys:

            - ``labeled_data`` (``Iterable[Tuple[Array, Array]] | Iterable[Tuple[Array, None]]``): An iterable of data whose elements (samples) are tuples of (data batch, label batch). The label batch can be ``None`` if the data is unlabelled.
    """

    def __init__(
        self, labeled_data: Iterable[Tuple[Array, Array]] | Iterable[Tuple[Array, None]]
    ):
        super().__init__(
            dist_params={},
            dist_hparams={"labeled_data": labeled_data},
        )

    def sample(
        self, key: Array, num_samples: int
    ) -> Tuple[Array, Array] | Tuple[Array, None]:
        """
        Sample from the empirical distribution using reservoir sampling.
        Assumes all batches in ``labeled_data`` are consistent: either all have labels (``Array``)
        or none have labels (``None``).

        Args:
            key (``Array``): The JAX PRNG key to use for sampling.
            num_samples (``int``): The number of samples to draw.

        Returns:
            ``Tuple[Array[num_samples, *data_dims], Array[num_samples, *label_dims]] | Tuple[Array[num_samples, *data_dims], None]``: A tuple ``(samples, labels)`` containing the samples and corresponding labels (stacked into an ``Array``), or ``(samples, None)`` if the data is unlabelled.
        """
        data_iterator = iter(self.dist_hparams["labeled_data"])  # Get an iterator

        # Initialize reservoir
        reservoir_samples = []
        reservoir_labels = []  # Will store labels if present, otherwise remains empty
        items_seen = 0
        is_labeled = None  # Determine based on first batch

        for X_batch, y_batch in data_iterator:
            # Determine if data is labeled based on the first batch encountered
            if is_labeled is None:
                is_labeled = y_batch is not None
                if is_labeled:
                    # Basic validation for the first labeled batch
                    if (
                        not isinstance(y_batch, jnp.ndarray)
                        or y_batch.shape[0] != X_batch.shape[0]
                    ):
                        raise ValueError(
                            f"First labeled batch has inconsistent shape. X shape: {X_batch.shape}, Y shape: {getattr(y_batch, 'shape', 'N/A')}"
                        )
                # else: y_batch is None, is_labeled remains False

            current_batch_size = X_batch.shape[0]

            # Reservoir sampling
            for i in range(current_batch_size):
                x = X_batch[i]
                y = y_batch[i] if is_labeled else None

                if items_seen < num_samples:
                    reservoir_samples.append(x)
                    if is_labeled:
                        reservoir_labels.append(y)
                else:
                    key, subkey = jax.random.split(key)
                    j = jax.random.randint(
                        subkey, shape=(), minval=0, maxval=items_seen + 1
                    )
                    if j < num_samples:
                        reservoir_samples[j] = x
                        if is_labeled:
                            reservoir_labels[j] = y

                items_seen += 1

        # Final checks and return
        if items_seen < num_samples:
            raise ValueError(
                f"Requested {num_samples} samples, but only {items_seen} items are available in the dataset."
            )

        # Stack samples into a single array
        stacked_samples = jnp.stack(reservoir_samples)

        # Stack labels if data was labeled, otherwise return None
        stacked_labels = None
        if is_labeled:
            stacked_labels = jnp.stack(reservoir_labels)
            return stacked_samples, stacked_labels
        else:
            return stacked_samples, None

    def score(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Computes the score function (``∇_x log p_t(x)``) of the empirical distribution at time ``t``,
        given the noisy state ``x_t`` and the diffusion process.

        Args:
            x_t (``Array[*data_dims]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time tensor.
            diffusion_process (``DiffusionProcess``): The diffusion process.

        Returns:
            ``Array[*data_dims]``: The score of the empirical distribution at ``(x_t, t)``.
        """
        x0_x_t = self.x0(x_t, t, diffusion_process)
        alpha_t = diffusion_process.alpha(t)
        sigma_t = diffusion_process.sigma(t)
        alpha_prime_t = diffusion_process.alpha_prime(t)
        sigma_prime_t = diffusion_process.sigma_prime(t)
        score_x_t = convert_vector_field_type(
            x_t,
            x0_x_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            VectorFieldType.X0,
            VectorFieldType.SCORE,
        )
        return score_x_t

    def x0(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Computes the denoiser ``E[x_0 | x_t]`` for an empirical distribution w.r.t. a given diffusion process.

        This method computes the denoiser by performing a weighted average of the
        dataset samples, where the weights are determined by the likelihood of ``x_t``
        given each sample.

        Arguments:
            x_t (``Array[*data_dims]``): The input tensor.
            t (``Array[]``): The time tensor.
            diffusion_process (``DiffusionProcess``): The diffusion process.

        Returns:
            ``Array[*data_dims]``: The prediction of ``x_0``.
        """
        data = self.dist_hparams["labeled_data"]

        alpha_t = diffusion_process.alpha(t)
        sigma_t = diffusion_process.sigma(t)

        softmax_denom = jnp.zeros_like(t)
        x0_hat = jnp.zeros_like(x_t)
        for X_batch, y_batch in data:
            squared_dists = jax.vmap(lambda x: jnp.sum((x_t - alpha_t * x) ** 2))(
                X_batch
            )
            exp_negative_dists = jnp.exp(-squared_dists / (2 * sigma_t**2))
            softmax_denom += jnp.sum(exp_negative_dists)
            x0_hat += jnp.sum(exp_negative_dists[:, None] * X_batch, axis=0)

        eps = cast(float, jnp.finfo(softmax_denom.dtype).eps)
        jax.debug.print("softmax_denom: {x}", x=softmax_denom)
        softmax_denom = jnp.maximum(softmax_denom, eps)
        x0_hat = x0_hat / softmax_denom
        return x0_hat

    def eps(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Computes the noise field ``eps(x_t, t)`` for an empirical distribution w.r.t. a given diffusion process.

        Args:
            x_t (``Array[*data_dims]``): The input tensor.
            t (``Array[]``): The time tensor.
            diffusion_process (``DiffusionProcess``): The diffusion process.

        Returns:
            ``Array[*data_dims]``: The noise field at ``(x_t, t)``.
        """
        x0_x_t = self.x0(x_t, t, diffusion_process)
        alpha_t = diffusion_process.alpha(t)
        sigma_t = diffusion_process.sigma(t)
        alpha_prime_t = diffusion_process.alpha_prime(t)
        sigma_prime_t = diffusion_process.sigma_prime(t)
        eps_x_t = convert_vector_field_type(
            x_t,
            x0_x_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            VectorFieldType.X0,
            VectorFieldType.EPS,
        )
        return eps_x_t

    def v(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Computes the velocity field ``v(x_t, t)`` for an empirical distribution w.r.t. a given diffusion process.

        Args:
            x_t (``Array[*data_dims]``): The input tensor.
            t (``Array[]``): The time tensor.
            diffusion_process (``DiffusionProcess``): The diffusion process.

        Returns:
            ``Array[*data_dims]``: The velocity field at ``(x_t, t)``.
        """
        x0_x_t = self.x0(x_t, t, diffusion_process)
        alpha_t = diffusion_process.alpha(t)
        sigma_t = diffusion_process.sigma(t)
        alpha_prime_t = diffusion_process.alpha_prime(t)
        sigma_prime_t = diffusion_process.sigma_prime(t)
        v_x_t = convert_vector_field_type(
            x_t,
            x0_x_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            VectorFieldType.X0,
            VectorFieldType.V,
        )
        return v_x_t
