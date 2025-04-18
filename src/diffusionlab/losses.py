from typing import Callable
from dataclasses import dataclass, field
import jax
from jax import numpy as jnp, Array

from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType


@dataclass
class DiffusionLoss:
    """
    Loss function for training diffusion models.

    This dataclass implements various loss functions for diffusion models based on the specified
    target type. The loss is computed as the mean squared error between the model's prediction
    and the target, which depends on the chosen vector field type.

    The loss supports different target types:

    - ``VectorFieldType.X0``: Learn to predict the original clean data x_0
    - ``VectorFieldType.EPS``: Learn to predict the noise component eps
    - ``VectorFieldType.V``: Learn to predict the velocity field v
    - ``VectorFieldType.SCORE``: Not directly supported (raises ValueError)

    Attributes:
        diffusion_process (``DiffusionProcess``): The diffusion process defining the forward dynamics
        vector_field_type (``VectorFieldType``): The type of target to learn to estimate via minimizing the loss function.
        num_noise_draws_per_sample (``int``): The number of noise draws per sample to use for the batchwise loss.
        target (``Callable[[Array, Array, Array, Array, Array], Array]``): Function that computes the target based on the specified target_type.

            Signature: ``(x_t: Array[*data_dims], f_x_t: Array[*data_dims], x_0: Array[*data_dims], eps: Array[*data_dims], t: Array[]) -> Array[*data_dims]``
    """

    diffusion_process: DiffusionProcess
    vector_field_type: VectorFieldType
    num_noise_draws_per_sample: int
    target: Callable[[Array, Array, Array, Array, Array], Array] = field(init=False)

    def __post_init__(self):
        match self.vector_field_type:
            case VectorFieldType.X0:

                def target(
                    x_t: Array, f_x_t: Array, x_0: Array, eps: Array, t: Array
                ) -> Array:
                    return x_0

            case VectorFieldType.EPS:

                def target(
                    x_t: Array, f_x_t: Array, x_0: Array, eps: Array, t: Array
                ) -> Array:
                    return eps

            case VectorFieldType.V:

                def target(
                    x_t: Array, f_x_t: Array, x_0: Array, eps: Array, t: Array
                ) -> Array:
                    return (
                        self.diffusion_process.alpha_prime(t) * x_0
                        + self.diffusion_process.sigma_prime(t) * eps
                    )

            case VectorFieldType.SCORE:
                raise ValueError(
                    "Direct score matching is not supported due to lack of a known target function, and other ways (like Hutchinson's trace estimator) are very high variance."
                )

            case _:
                raise ValueError(f"Invalid target type: {self.vector_field_type}")

        self.target = target

    def prediction_loss(
        self, x_t: Array, f_x_t: Array, x_0: Array, eps: Array, t: Array
    ) -> Array:
        """
        Compute the loss given a prediction and inputs/targets.

        This method calculates the mean squared error between the model's prediction (``f_x_t``)
        and the target value determined by the target_type (``self.target``).

        Args:
            x_t (``Array[*data_dims]``): The noised data at time ``t``.
            f_x_t (``Array[*data_dims]``): The model's prediction at time ``t``.
            x_0 (``Array[*data_dims]``): The original clean data.
            eps (``Array[*data_dims]``): The noise used to generate ``x_t``.
            t (``Array[]``): The scalar time parameter.

        Returns:
            ``Array[]``: The scalar loss value for the given sample.
        """
        squared_residuals = (f_x_t - self.target(x_t, f_x_t, x_0, eps, t)) ** 2
        samplewise_loss = jnp.sum(squared_residuals)
        return samplewise_loss

    def loss(
        self,
        key: Array,
        vector_field: Callable[[Array, Array], Array],
        x_0: Array,
        t: Array,
    ) -> Array:
        """
        Compute the average loss over multiple noise draws for a single data point and time.

        This method estimates the expected loss at a given time ``t`` for a clean data sample ``x_0``.
        It does this by drawing ``num_noise_draws_per_sample`` noise vectors (``eps``), generating
        the corresponding noisy samples ``x_t`` using the ``diffusion_process``, predicting the
        target quantity ``f_x_t`` using the provided ``vector_field`` (vmapped internally), and then calculating the
        ``prediction_loss`` for each noise sample. The final loss is the average over these samples.

        Args:
            key (``Array``): The PRNG key for noise generation.
            vector_field (``Callable[[Array, Array], Array]``): The vector field function that takes
                a single noisy data sample ``x_t`` and its corresponding time ``t``, and returns the model's prediction ``f_x_t``.
                This function will be vmapped internally over the batch dimension created by ``num_noise_draws_per_sample``.

                Signature: ``(x_t: Array[*data_dims], t: Array[]) -> Array[*data_dims]``.

            x_0 (``Array[*data_dims]``): The original clean data sample.
            t (``Array[]``): The scalar time parameter.

        Returns:
            ``Array[]``: The scalar loss value, averaged over ``num_noise_draws_per_sample`` noise instances.
        """
        x_0_batch = x_0[None, ...].repeat(self.num_noise_draws_per_sample, axis=0)
        t_batch = t[None].repeat(self.num_noise_draws_per_sample, axis=0)
        eps_batch = jax.random.normal(key, x_0_batch.shape)

        batch_diffusion_forward = jax.vmap(
            self.diffusion_process.forward, in_axes=(0, 0, 0)
        )
        x_t_batch = batch_diffusion_forward(x_0_batch, t_batch, eps_batch)

        batch_vector_field = jax.vmap(vector_field, in_axes=(0, 0))
        f_x_t_batch = batch_vector_field(x_t_batch, t_batch)

        batch_prediction_loss = jax.vmap(self.prediction_loss, in_axes=(0, 0, 0, 0, 0))
        losses = batch_prediction_loss(
            x_t_batch, f_x_t_batch, x_0_batch, eps_batch, t_batch
        )

        loss_value = jnp.mean(losses)
        return loss_value
