from dataclasses import dataclass
from typing import Any
from jax import Array, numpy as jnp


@dataclass(frozen=True)
class Scheduler:
    """
    Base class for time step schedulers used in diffusion, denoising, and sampling.

    Allows for extensible scheduler implementations where subclasses can define
    their own initialization and time step generation parameters via **kwargs.
    """

    def get_ts(self, **ts_hparams: Any) -> Array:
        """
        Generate the sequence of time steps.

        This is an abstract method that must be implemented by subclasses.
        Subclasses should define the specific keyword arguments they expect
        within ``**ts_hparams``.

        Args:
            **ts_hparams (``Dict[str, Any]``): Keyword arguments containing parameters for generating time steps.

        Returns:
            ``Array``: A tensor containing the sequence of time steps in descending order.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            KeyError: If a required parameter is missing in ``**ts_hparams`` (in subclass).
        """
        raise NotImplementedError


@dataclass(frozen=True)
class UniformScheduler(Scheduler):
    """
    A scheduler that generates uniformly spaced time steps.

    Requires ``t_min``, ``t_max``, and ``num_steps`` to be passed
    to the ``get_ts`` method via keyword arguments. The number of points generated
    will be ``num_steps + 1``.
    """

    def get_ts(self, **ts_hparams: Any) -> Array:
        """
        Generate uniformly spaced time steps.

        Args:
            **ts_hparams (``Dict[str, Any]``): Keyword arguments must contain

                - ``t_min`` (``float``): The minimum time value, typically close to 0.
                - ``t_max`` (``float``): The maximum time value, typically close to 1.
                - ``num_steps`` (``int``): The number of diffusion steps. The function will generate ``num_steps + 1`` time points.

        Returns:
            ``Array[num_steps+1]``: A JAX array containing uniformly spaced time steps
                                    in descending order (from ``t_max`` to ``t_min``).

        Raises:
            KeyError: If ``t_min``, ``t_max``, or ``num_steps`` is not found in ``ts_hparams``.
            AssertionError: If ``t_min``/``t_max`` constraints are violated or ``num_steps`` < 1.
        """
        try:
            t_min = ts_hparams["t_min"]
            t_max = ts_hparams["t_max"]
            num_steps = ts_hparams["num_steps"]
        except KeyError as e:
            raise KeyError(
                f"Missing required parameter for UniformScheduler.get_ts: {e}"
            ) from e

        assert 0 <= t_min <= t_max <= 1, "t_min and t_max must be in the range [0, 1]"
        assert num_steps >= 1, "num_steps must be at least 1"

        ts = jnp.linspace(t_min, t_max, num_steps + 1)[::-1]
        return ts
