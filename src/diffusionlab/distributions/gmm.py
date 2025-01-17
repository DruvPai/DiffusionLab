from typing import Any, Dict, Literal, Tuple
import torch

from diffusionlab.distributions.base import Distribution
from diffusionlab.samplers import Sampler
from diffusionlab.utils import vector_lstsq, logdet_pd, sqrt_psd


class GMMDistribution(Distribution):
    def __init__(
        self,
        sampler: Sampler,
        means: torch.Tensor,
        covs: torch.Tensor,
        priors: torch.Tensor,
    ):
        super().__init__(sampler, {"means": means, "covs": covs, "priors": priors})
        self.means = means
        self.covs = covs
        self.priors = priors

    @classmethod
    def validate_params(cls, dist_params: Dict[str, Any]) -> None:
        assert (
            "means" in dist_params and "covs" in dist_params and "priors" in dist_params
        )
        means = dist_params["means"]
        covs = dist_params["covs"]
        priors = dist_params["priors"]
        assert len(means.shape) == 2
        K, D = means.shape
        assert (
            len(covs.shape) == 3
            and covs.shape[0] == K
            and covs.shape[1] == D
            and covs.shape[2] == D
        )
        assert len(priors.shape) == 1 and priors.shape[0] == K
        assert means.device == covs.device == priors.device

        assert torch.allclose(
            torch.sum(priors), torch.tensor(1.0, device=priors.device)
        )

        evals = torch.linalg.eigvalsh(covs)
        assert torch.all(evals >= 0)

    @classmethod
    def stateless_x0(
        cls,
        sampler: Sampler,
        dist_params: Dict[str, Any],
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        means = dist_params["means"]  # (K, D)
        covs = dist_params["covs"]  # (K, D, D)
        priors = dist_params["priors"]  # (K, )

        K, D = means.shape
        N = x.shape[0]

        alpha = sampler.alpha(t)  # (N, )
        sigma = sampler.sigma(t)  # (N, )

        covs_t = (alpha[:, None, None, None] ** 2) * covs[None, :, :, :] + (
            sigma[:, None, None, None] ** 2
        ) * torch.eye(D, device=x.device)[
            None, None, :, :
        ]  # (N, K, D, D)
        centered_x = (
            x[:, None, :] - alpha[:, None, None] * means[None, :, :]
        )  # (N, K, D)
        covs_t_inv_centered_x = vector_lstsq(covs_t, centered_x)  # (N, K, D)

        mahalanobis_dists = torch.sum(
            centered_x * covs_t_inv_centered_x, dim=-1
        )  # (N, K)
        logdets_covs_t = logdet_pd(covs_t)  # (N, K)
        w = (
            torch.log(priors)[None, :]
            - 1 / 2 * logdets_covs_t
            - 1 / 2 * mahalanobis_dists
        )  # (N, K)
        softmax_w = torch.softmax(w, dim=-1)  # (N, K)

        weighted_normalized_x = torch.sum(
            softmax_w[:, :, None] * covs_t_inv_centered_x, dim=-2
        )  # (N, D)
        x0_hat = (1 / alpha[:, None]) * (
            x - (sigma[:, None] ** 2) * weighted_normalized_x
        )  # (N, D)

        return x0_hat

    def sample(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        K, D = self.means.shape

        device = self.priors.device
        y = torch.multinomial(self.priors, N, replacement=True)  # (N, )
        X = torch.empty((N, D), device=device)
        for k in range(K):
            idx = y == k
            cov_k = self.covs[k]
            cov_sqrt_k = sqrt_psd(cov_k)
            means_k = self.means[k]
            X[idx] = (
                torch.randn((X[idx].shape[0], D), device=device) @ cov_sqrt_k + means_k
            )
        return X, y


class IsoHomoGMMDistribution(Distribution):
    def __init__(
        self,
        sampler: Sampler,
        means: torch.Tensor,
        var: torch.Tensor,
        priors: torch.Tensor,
    ):
        super().__init__(sampler, {"means": means, "var": var, "priors": priors})
        self.means = means
        self.var = var
        self.priors = priors

    @classmethod
    def validate_params(cls, dist_params: Dict[str, Any]) -> None:
        assert (
            "means" in dist_params and "var" in dist_params and "priors" in dist_params
        )
        means = dist_params["means"]
        var = dist_params["var"]
        priors = dist_params["priors"]
        assert len(means.shape) == 2
        K, D = means.shape
        assert len(var.shape) == 0
        assert len(priors.shape) == 1 and priors.shape[0] == K
        assert means.device == var.device == priors.device

        assert torch.allclose(
            torch.sum(priors), torch.tensor(1.0, device=priors.device)
        )

        assert var >= 0

    @classmethod
    def stateless_x0(
        cls,
        sampler: Sampler,
        dist_params: Dict[str, Any],
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        means = dist_params["means"]  # (K, D)
        var = dist_params["var"]  # ()
        priors = dist_params["priors"]  # (K, )

        K, D = means.shape
        N = x.shape[0]

        alpha = sampler.alpha(t)  # (N, )
        sigma = sampler.sigma(t)  # (N, )

        var_t = (alpha**2) * var[None] + (sigma**2)  # (N, )
        centered_x = (
            x[:, None, :] - alpha[:, None, None] * means[None, :, :]
        )  # (N, K, D)
        vars_t_inv_centered_x = centered_x / var_t[:, None, None]  # (N, K, D)

        mahalanobis_dists = torch.sum(
            centered_x * vars_t_inv_centered_x, dim=-1
        )  # (N, K)
        w = torch.log(priors)[None, :] - 1 / 2 * mahalanobis_dists  # (N, K)
        softmax_w = torch.softmax(w, dim=-1)  # (N, K)

        weighted_normalized_x = torch.sum(
            softmax_w[:, :, None] * vars_t_inv_centered_x, dim=-2
        )  # (N, D)
        x0_hat = (1 / alpha[:, None]) * (
            x - (sigma[:, None] ** 2) * weighted_normalized_x
        )  # (N, D)

        return x0_hat

    def sample(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        K, D = self.means.shape
        covs = (
            torch.eye(D, device=self.var.device)[None, :, :].expand(K, -1, -1)
            * self.var
        )
        return GMMDistribution(self.sampler, self.means, covs, self.priors).sample(N)
