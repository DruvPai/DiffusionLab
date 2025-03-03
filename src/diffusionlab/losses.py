from typing import Callable

import torch
from torch import nn

from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.utils import pad_shape_back
from diffusionlab.vector_fields import VectorFieldType


class SamplewiseDiffusionLoss(nn.Module):
    """
    Sample-wise loss function for training diffusion models.

    This class implements various loss functions for diffusion models based on the specified
    target type. The loss is computed as the mean squared error between the model's prediction
    and the target, which depends on the chosen vector field type.

    The loss supports different target types:
    - X0: Learn to predict the original clean data x_0
    - EPS: Learn to predict the noise component eps
    - V: Learn to predict the velocity field v
    - SCORE: Not directly supported (raises ValueError)

    Attributes:
        diffusion (DiffusionProcess): The diffusion process defining the forward dynamics
        target_type (VectorFieldType): The type of target to learn via minimizing the loss function
        target (Callable): Function that computes the target based on the specified target_type.
                          Takes tensors of shapes (N, *D) for x_t, f_x_t, x_0, eps and (N,) for t,
                          and returns a tensor of shape (N, *D).
    """

    def __init__(
        self, diffusion_process: DiffusionProcess, target_type: VectorFieldType
    ) -> None:
        """
        Initialize the diffusion loss function.

        Args:
            diffusion_process: The diffusion process to use, containing data about the forward evolution.
            target_type: The type of target to learn via minimizing the loss function.
                         Must be one of VectorFieldType.X0, VectorFieldType.EPS, or VectorFieldType.V.

        Raises:
            ValueError: If target_type is VectorFieldType.SCORE, which is not directly supported.
        """
        super().__init__()
        self.diffusion_process: DiffusionProcess = diffusion_process
        self.target_type: VectorFieldType = target_type

        if target_type == VectorFieldType.X0:

            def target(
                x_t: torch.Tensor,
                f_x_t: torch.Tensor,
                x_0: torch.Tensor,
                eps: torch.Tensor,
                t: torch.Tensor,
            ) -> torch.Tensor:
                """
                Target function for predicting the original clean data x_0.

                Args:
                    x_t (torch.Tensor): The noised data at time t, of shape (N, *D).
                    f_x_t (torch.Tensor): The model's prediction at time t, of shape (N, *D).
                    x_0 (torch.Tensor): The original clean data, of shape (N, *D).
                    eps (torch.Tensor): The noise used to generate x_t, of shape (N, *D).
                    t (torch.Tensor): The time parameter, of shape (N,).

                Returns:
                    torch.Tensor: The target tensor x_0, of shape (N, *D).
                """
                return x_0

        elif target_type == VectorFieldType.EPS:

            def target(
                x_t: torch.Tensor,
                f_x_t: torch.Tensor,
                x_0: torch.Tensor,
                eps: torch.Tensor,
                t: torch.Tensor,
            ) -> torch.Tensor:
                """
                Target function for predicting the noise component eps.

                Args:
                    x_t (torch.Tensor): The noised data at time t, of shape (N, *D).
                    f_x_t (torch.Tensor): The model's prediction at time t, of shape (N, *D).
                    x_0 (torch.Tensor): The original clean data, of shape (N, *D).
                    eps (torch.Tensor): The noise used to generate x_t, of shape (N, *D).
                    t (torch.Tensor): The time parameter, of shape (N,).

                Returns:
                    torch.Tensor: The target tensor eps, of shape (N, *D).
                """
                return eps

        elif target_type == VectorFieldType.V:

            def target(
                x_t: torch.Tensor,
                f_x_t: torch.Tensor,
                x_0: torch.Tensor,
                eps: torch.Tensor,
                t: torch.Tensor,
            ) -> torch.Tensor:
                """
                Target function for predicting the velocity field v.

                Args:
                    x_t (torch.Tensor): The noised data at time t, of shape (N, *D).
                    f_x_t (torch.Tensor): The model's prediction at time t, of shape (N, *D).
                    x_0 (torch.Tensor): The original clean data, of shape (N, *D).
                    eps (torch.Tensor): The noise used to generate x_t, of shape (N, *D).
                    t (torch.Tensor): The time parameter, of shape (N,).

                Returns:
                    torch.Tensor: The velocity field target tensor, of shape (N, *D).
                """
                return (
                    pad_shape_back(self.diffusion_process.alpha_prime(t), x_0.shape)
                    * x_0
                    + pad_shape_back(self.diffusion_process.sigma_prime(t), x_0.shape)
                    * eps
                )

        elif target_type == VectorFieldType.SCORE:
            raise ValueError(
                "Direct score matching is not supported due to lack of a known target function, and other ways (like Hutchinson's trace estimator) are very high variance."
            )

        self.target: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ] = target

    def forward(
        self,
        x_t: torch.Tensor,
        f_x_t: torch.Tensor,
        x_0: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the loss for each sample in the batch.

        This method calculates the mean squared error between the model's prediction (f_x_t)
        and the target value determined by the target_type.

        Args:
            x_t (torch.Tensor): The noised data at time t, of shape (N, *D) where N is the batch size
                               and D represents the data dimensions.
            f_x_t (torch.Tensor): The model's prediction at time t, of shape (N, *D).
            x_0 (torch.Tensor): The original clean data, of shape (N, *D).
            eps (torch.Tensor): The noise used to generate x_t, of shape (N, *D).
            t (torch.Tensor): The time parameter, of shape (N,).

        Returns:
            torch.Tensor: The per-sample loss values, of shape (N,) where N is the batch size.
        """
        # Compute squared error between prediction and target
        squared_residuals = (f_x_t - self.target(x_t, f_x_t, x_0, eps, t)) ** 2

        # Sum over all dimensions except batch dimension
        samplewise_loss = torch.sum(
            torch.flatten(squared_residuals, start_dim=1, end_dim=-1), dim=1
        )

        return samplewise_loss
