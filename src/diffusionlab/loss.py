from typing import Callable

import torch
from torch import nn

from diffusionlab.samplers import Sampler
from diffusionlab.utils import pad_shape_back
from diffusionlab.vector_fields import VectorFieldType


class SamplewiseDiffusionLoss(nn.Module):
    """
    The diffusion loss function.

    Parameters:
        sampler: The sampler to use, containing data about the forward evolution of the process.
        target_type: The type of target to learn via minimizing the loss function.
    """

    def __init__(self, sampler: Sampler, target_type: VectorFieldType) -> None:
        super().__init__()
        self.sampler = sampler
        self.target_type = target_type
        if target_type == VectorFieldType.X0:

            def target(
                xt: torch.Tensor,
                fxt: torch.Tensor,
                x0: torch.Tensor,
                eps: torch.Tensor,
                t: torch.Tensor,
            ) -> torch.Tensor:
                return x0

        elif target_type == VectorFieldType.EPS:

            def target(
                xt: torch.Tensor,
                fxt: torch.Tensor,
                x0: torch.Tensor,
                eps: torch.Tensor,
                t: torch.Tensor,
            ) -> torch.Tensor:
                return eps

        elif target_type == VectorFieldType.V:

            def target(
                xt: torch.Tensor,
                fxt: torch.Tensor,
                x0: torch.Tensor,
                eps: torch.Tensor,
                t: torch.Tensor,
            ) -> torch.Tensor:
                return (
                    pad_shape_back(sampler.alpha_prime(t), x0.shape) * x0
                    + pad_shape_back(sampler.sigma_prime(t), x0.shape) * eps
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
        xt: torch.Tensor,
        fxt: torch.Tensor,
        x0: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        squared_residuals = (fxt - self.target(xt, fxt, x0, eps, t)) ** 2
        samplewise_loss = torch.sum(
            torch.flatten(squared_residuals, start_dim=1, end_dim=-1), dim=1
        )
        return samplewise_loss
