from typing import Any, Callable
import torch

from diffusionlab.utils import pad_shape_back, scalar_derivative


class DiffusionProcess:
    """
    Base class for implementing various diffusion processes.

    A diffusion process defines how data evolves over time when noise is added according to
    specific dynamics. This class provides a framework for implementing different types of
    diffusion processes used in generative modeling.

    The diffusion is parameterized by two functions:
    - alpha(t): Controls how much of the original signal is preserved at time t
    - sigma(t): Controls how much noise is added at time t

    The forward process is defined as: x_t = alpha(t) * x_0 + sigma(t) * eps, where:
    - x_0 is the original data
    - x_t is the noised data at time t
    - eps is random noise sampled from a standard Gaussian distribution
    - t is the diffusion time parameter, typically in range [0, 1]

    Attributes:
        alpha (Callable): Function that determines signal preservation at time t, differentiable,
                         maps any tensor to tensor of same shape
        sigma (Callable): Function that determines noise level at time t, differentiable,
                         maps any tensor to tensor of same shape
        alpha_prime (Callable): Derivative of alpha, maps any tensor to tensor of same shape
        sigma_prime (Callable): Derivative of sigma, maps any tensor to tensor of same shape
    """

    def __init__(self, **dynamics_hparams: Any) -> None:
        """
        Initialize a diffusion process with specific dynamics parameters.

        Args:
            **dynamics_hparams: Keyword arguments containing the dynamics parameters.
                Must include:
                - alpha: Callable that maps time t to signal coefficient
                - sigma: Callable that maps time t to noise coefficient

        Raises:
            AssertionError: If alpha or sigma is not provided in dynamics_hparams
        """
        super().__init__()
        assert "alpha" in dynamics_hparams
        assert "sigma" in dynamics_hparams
        alpha: Callable[[torch.Tensor], torch.Tensor] = dynamics_hparams["alpha"]
        sigma: Callable[[torch.Tensor], torch.Tensor] = dynamics_hparams["sigma"]
        self.alpha: Callable[[torch.Tensor], torch.Tensor] = alpha
        self.sigma: Callable[[torch.Tensor], torch.Tensor] = sigma
        self.alpha_prime: Callable[[torch.Tensor], torch.Tensor] = scalar_derivative(
            alpha
        )
        self.sigma_prime: Callable[[torch.Tensor], torch.Tensor] = scalar_derivative(
            sigma
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the dynamics model.

        This method implements the forward diffusion process, which gradually adds noise to the input data
        according to the specified dynamics (alpha and sigma functions).

        Args:
            x (torch.Tensor): The input data tensor of shape (N, *D), where N is the batch size
                             and D represents the data dimensions.
            t (torch.Tensor): The time parameter tensor of shape (N,) or broadcastable to x's shape,
                             with values typically in the range [0, 1].
            eps (torch.Tensor): The Gaussian noise tensor of shape (N, *D), where N is the batch size
                               and D represents the data dimensions.
        Returns:
            torch.Tensor: The noised data at time t, computed as alpha(t) * x + sigma(t) * eps,
                         of shape (N, *D) matching the input shape.
        """
        alpha = pad_shape_back(self.alpha(t), x.shape)
        sigma = pad_shape_back(self.sigma(t), x.shape)
        return alpha * x + sigma * eps


class VarianceExplodingProcess(DiffusionProcess):
    """
    Implements a Variance Exploding (VE) diffusion process.

    In a VE process, the signal component remains constant (alpha(t) = 1) while the
    noise component increases according to the provided sigma function. This leads to
    the variance of the process "exploding" as t increases.

    The forward process is defined as: x_t = x_0 + sigma(t) * eps

    This is used in models like NCSN (Noise Conditional Score Network) and Score SDE.
    """

    def __init__(self, sigma: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """
        Initialize a Variance Exploding diffusion process.

        Args:
            sigma (Callable): Function that determines how noise scales with time t.
                             Should map a tensor of time values of shape (N,) to noise
                             coefficients of the same shape.
        """
        super().__init__(alpha=lambda t: torch.ones_like(t), sigma=sigma)


class OrnsteinUhlenbeckProcess(DiffusionProcess):
    """
    Implements an Ornstein-Uhlenbeck diffusion process.

    The Ornstein-Uhlenbeck process is a mean-reverting stochastic process that describes
    the velocity of a particle undergoing Brownian motion while being subject to friction.

    In this implementation:
    - alpha(t) = sqrt(1 - t²)
    - sigma(t) = t

    This process has properties that make it useful for certain generative modeling tasks,
    particularly when a smooth transition between clean and noisy states is desired.
    """

    def __init__(self) -> None:
        """
        Initialize an Ornstein-Uhlenbeck diffusion process with predefined dynamics.

        The process uses:
        - alpha(t) = sqrt(1 - t²)
        - sigma(t) = t

        Both functions map tensors of shape (N,) to tensors of the same shape.
        """
        super().__init__(alpha=lambda t: torch.sqrt(1 - t**2), sigma=lambda t: t)


class FlowMatchingProcess(DiffusionProcess):
    """
    Implements a Flow Matching diffusion process.

    Flow Matching is a technique used in generative modeling where the goal is to learn
    a continuous transformation (flow) between a simple distribution and a complex data
    distribution.

    In this implementation:
    - alpha(t) = 1 - t
    - sigma(t) = t

    This creates a linear interpolation between the original data (at t=0) and
    the noise (at t=1), which is useful for training flow-based generative models.
    """

    def __init__(self) -> None:
        """
        Initialize a Flow Matching diffusion process with predefined dynamics.

        The process uses:
        - alpha(t) = 1 - t
        - sigma(t) = t

        Both functions map tensors of shape (N,) to tensors of the same shape.
        This creates a linear interpolation between the original data and noise.
        """
        super().__init__(alpha=lambda t: 1 - t, sigma=lambda t: t)
