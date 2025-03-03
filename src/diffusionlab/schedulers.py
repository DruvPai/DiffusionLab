from typing import Any

import torch


class Scheduler:
    """
    Base class for time step schedulers used in diffusion, denoising, and sampling.

    A scheduler determines the sequence of time steps used during the sampling process.
    Different scheduling strategies can affect the quality and efficiency of the
    generative process.

    The scheduler generates a sequence of time values, typically in the range [0, 1],
    which are used to control the noise level at each step of the sampling process.
    """

    def __init__(self, **schedule_hparams: Any) -> None:
        """
        Initialize the scheduler.

        This base implementation does not store any variables.
        Subclasses may override this method to initialize specific parameters.

        Args:
            **schedule_hparams: Keyword arguments containing scheduler parameters.
                                Not used in the base class but available for subclasses.
        """
        pass

    def get_ts(self, **ts_hparams: Any) -> torch.Tensor:
        """
        Generate the sequence of time steps.

        This is an abstract method that must be implemented by subclasses.

        Args:
            **ts_hparams: Keyword arguments containing parameters for generating time steps.
                          The specific parameters depend on the scheduler implementation.
                          Typically includes:
                          - t_min (float): The minimum time value
                          - t_max (float): The maximum time value
                          - L (int): The number of time steps to generate

        Returns:
            torch.Tensor: A tensor of shape (L,) containing the sequence of time steps
                         in descending order, where L is the number of time steps.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError


class UniformScheduler(Scheduler):
    """
    A scheduler that generates uniformly spaced time steps.

    This scheduler creates a sequence of time steps that are uniformly distributed
    between a minimum and maximum time value. The time steps are returned in
    descending order (from t_max to t_min).

    This is the simplest scheduling strategy and is often used as a baseline.
    """

    def __init__(self, **schedule_hparams: Any) -> None:
        """
        Initialize the uniform scheduler.

        This implementation does not store any variables, following the base class design.

        Args:
            **schedule_hparams: Keyword arguments containing scheduler parameters.
                                Not used but passed to the parent class.
        """
        super().__init__(**schedule_hparams)

    def get_ts(self, **ts_hparams: Any) -> torch.Tensor:
        """
        Generate uniformly spaced time steps.

        Args:
            **ts_hparams: Keyword arguments containing:
                - t_min (float): The minimum time value, typically close to 0.
                - t_max (float): The maximum time value, typically close to 1.
                - L (int): The number of time steps to generate.

        Returns:
            torch.Tensor: A tensor of shape (L,) containing uniformly spaced time steps
                         in descending order (from t_max to t_min), where L is the number
                         of time steps specified in ts_hparams.

        Raises:
            AssertionError: If t_min or t_max are outside the range [0, 1] or if t_min > t_max.
        """
        t_min = ts_hparams["t_min"]
        t_max = ts_hparams["t_max"]
        L = ts_hparams["L"]
        assert 0 <= t_min <= t_max <= 1, "t_min and t_max must be in the range [0, 1]"
        assert L >= 2, "L must be at least 2"

        ts = torch.linspace(t_min, t_max, L)
        ts = ts.flip(0)
        return ts
