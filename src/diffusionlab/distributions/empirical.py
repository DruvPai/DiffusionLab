from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.distributions.base import Distribution
from diffusionlab.utils import pad_shape_back


class EmpiricalDistribution(Distribution):
    """
    An empirical distribution, i.e., the uniform distribution over a dataset.
    Formally, the distribution is defined as:

    mu(B) = (1/N) * sum_(i=1)^(N) delta(x_i in B)

    where x_i is the ith data point in the dataset, and N is the number of data points.

    Distribution Parameters:
        - None

    Distribution Hyperparameters:
        - labeled_data: A DataLoader of data which spawns the empirical distribution, where each data sample is a (data, label) tuple. Both data and label are PyTorch tensors.

    Note:
        - This class has no sample() method as it's difficult to sample randomly from a DataLoader. In practice, you can sample directly from the DataLoader and apply filtering there.
    """

    @classmethod
    def validate_hparams(cls, dist_hparams: Dict[str, Any]) -> None:
        """
        Validate the hyperparameters for the empirical distribution.

        Arguments:
            dist_hparams: A dictionary of hyperparameters for the distribution.
                Must contain 'labeled_data' which is a DataLoader.

        Returns:
            None

        Throws:
            AssertionError: If the parameters are invalid.
        """
        assert "labeled_data" in dist_hparams
        labeled_data = dist_hparams["labeled_data"]
        assert isinstance(labeled_data, DataLoader)
        assert len(labeled_data) > 0

    @classmethod
    def x0(
        cls,
        x_t: torch.Tensor,
        t: torch.Tensor,
        diffusion_process: DiffusionProcess,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the denoiser E[x_0 | x_t] for an empirical distribution.

        This method computes the denoiser by performing a weighted average of the
        dataset samples, where the weights are determined by the likelihood of x_t
        given each sample.

        Arguments:
            x_t: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            diffusion_process: The diffusion process.
            batched_dist_params: A dictionary of batched parameters for the distribution.
                Not used for empirical distribution.
            dist_hparams: A dictionary of hyperparameters for the distribution.
                Must contain 'labeled_data' which is a DataLoader.

        Returns:
            The prediction of x_0, of shape (N, *D).
        """
        data = dist_hparams["labeled_data"]

        x_flattened = torch.flatten(x_t, start_dim=1, end_dim=-1)  # (N, *D)

        alpha = diffusion_process.alpha(t)  # (N, )
        sigma = diffusion_process.sigma(t)  # (N, )

        softmax_denom = torch.zeros_like(t)  # (N, )
        x0_hat = torch.zeros_like(x_t)  # (N, *D)
        for X_batch, y_batch in data:
            X_batch = X_batch.to(x_t.device, non_blocking=True)  # (B, *D)
            X_batch_flattened = torch.flatten(X_batch, start_dim=1, end_dim=-1)[
                None, ...
            ]  # (1, B, D*)
            alpha_X_batch_flattened = (
                pad_shape_back(alpha, X_batch_flattened.shape) * X_batch_flattened
            )  # (N, B, D*)
            dists = (
                torch.cdist(x_flattened[:, None, ...], alpha_X_batch_flattened)[
                    :, 0, ...
                ]
                ** 2
            )  # (N, B)
            exp_dists = torch.exp(
                -dists / (2 * pad_shape_back(sigma, dists.shape) ** 2)
            )  # (N, B)
            softmax_denom += torch.sum(exp_dists, dim=-1)  # (N, )
            x0_hat += torch.sum(
                pad_shape_back(exp_dists, X_batch[None, ...].shape)
                * X_batch[None, ...],  # (N, B, *D)
                dim=1,
            )  # (N, *D)

        softmax_denom = torch.maximum(
            softmax_denom,
            torch.tensor(
                torch.finfo(softmax_denom.dtype).eps, device=softmax_denom.device
            ),
        )
        x0_hat = x0_hat / pad_shape_back(softmax_denom, x0_hat.shape)  # (N, *D)
        return x0_hat
