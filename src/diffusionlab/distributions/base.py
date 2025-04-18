from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from jax import Array

from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType


@dataclass(frozen=True)
class Distribution:
    """
    Base class for all distributions.

    This class should be subclassed by other distributions when you want to use ground truth
    scores, denoisers, noise predictors, or velocity estimators.

    Each distribution implementation provides functions to sample from it and compute various vector fields
    related to a diffusion process, such as denoising (``x0``), noise prediction (``eps``),
    velocity estimation (``v``), and score estimation (``score``).

    Attributes:
        dist_params (``Dict[str, Array]``): Dictionary containing distribution parameters as JAX arrays.
                                             Shapes depend on the specific distribution.
        dist_hparams (``Dict[str, Any]``): Dictionary containing distribution hyperparameters (non-array values).
    """

    dist_params: Dict[str, Array]
    dist_hparams: Dict[str, Any]

    def sample(
        self,
        key: Array,
        num_samples: int,
    ) -> Tuple[Array, Any]:
        """
        Sample from the distribution.

        Args:
            key (``Array``): The JAX PRNG key to use for sampling.
            num_samples (``int``): The number of samples to draw.

        Returns:
            ``Tuple[Array[num_samples, *data_dims], Any]``: A tuple containing the samples and any additional information.
        """
        raise NotImplementedError

    def get_vector_field(
        self, vector_field_type: VectorFieldType
    ) -> Callable[[Array, Array, DiffusionProcess], Array]:
        """
        Get the vector field function of a given type associated with this distribution.

        Args:
            vector_field_type (``VectorFieldType``): The type of vector field to retrieve (e.g., ``VectorFieldType.SCORE``, ``VectorFieldType.X0``, ``VectorFieldType.EPS``, ``VectorFieldType.V``).

        Returns:
            ``Callable[[Array[*data_dims], Array[], DiffusionProcess], Array[*data_dims]]``:
                The requested vector field function. It takes the current state ``x_t`` (``Array[*data_dims]``),
                time ``t`` (``Array[]``), and the ``diffusion_process`` as input and returns the
                corresponding vector field value (``Array[*data_dims]``).
        """
        match vector_field_type:
            case VectorFieldType.X0:
                vector_field = self.x0
            case VectorFieldType.EPS:
                vector_field = self.eps
            case VectorFieldType.V:
                vector_field = self.v
            case VectorFieldType.SCORE:
                vector_field = self.score
            case _:
                raise ValueError(
                    f"Vector field type {vector_field_type} is not supported."
                )
        return vector_field

    def score(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Compute the score function (``∇_x log p_t(x)``) of the distribution at time ``t``,
        given the noisy state ``x_t`` and the ``diffusion_process``.

        Args:
            x_t (``Array[*data_dims]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step.
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[*data_dims]``: The score of the distribution at ``(x_t, t)``.
        """
        raise NotImplementedError

    def x0(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Predict the initial state ``x0`` (denoised sample) from the noisy state ``x_t`` at time ``t``,
        given the ``diffusion_process``.

        Args:
            x_t (``Array[*data_dims]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step.
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[*data_dims]``: The predicted initial state ``x0``.
        """
        raise NotImplementedError

    def eps(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Predict the noise component ``ε`` corresponding to the noisy state ``x_t`` at time ``t``,
        given the ``diffusion_process``.

        Args:
            x_t (``Array[*data_dims]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step.
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[*data_dims]``: The predicted noise ``ε``.
        """
        raise NotImplementedError

    def v(
        self,
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
    ) -> Array:
        """
        Compute the velocity field ``v(x_t, t)`` corresponding to the noisy state ``x_t`` at time ``t``,
        given the ``diffusion_process``.

        Args:
            x_t (``Array[*data_dims]``): The noisy state tensor at time ``t``.
            t (``Array[]``): The time step.
            diffusion_process (``DiffusionProcess``): The diffusion process definition.

        Returns:
            ``Array[*data_dims]``: The computed velocity field ``v``.
        """
        raise NotImplementedError
