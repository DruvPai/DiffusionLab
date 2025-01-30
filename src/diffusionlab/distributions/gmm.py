from typing import Any, Dict, Tuple

import torch

from diffusionlab.distributions.base import Distribution
from diffusionlab.samplers import Sampler
from diffusionlab.utils import logdet_pd, sqrt_psd, vector_lstsq


class GMMDistribution(Distribution):
    """
    A Gaussian Mixture Model (GMM) with K components.
    Formally, the distribution is defined as:

    mu(B) = sum_(i=1)^(K) pi_i * N(mu_i, Sigma_i)(B)

    where mu_i is the mean of the ith component, Sigma_i is the covariance matrix of the ith component,
    and pi_i is the prior probability of the ith component.

    Distribution Parameters:
        - means: A tensor of shape (K, D) containing the means of the components.
        - covs: A tensor of shape (K, D, D) containing the covariance matrices of the components.
        - priors: A tensor of shape (K, ) containing the prior probabilities of the components.

    Distribution Hyperparameters:
        - None
    """

    @classmethod
    def validate_params(
        cls, possibly_batched_dist_params: Dict[str, torch.Tensor]
    ) -> None:
        assert (
            "means" in possibly_batched_dist_params
            and "covs" in possibly_batched_dist_params
            and "priors" in possibly_batched_dist_params
        )
        means = possibly_batched_dist_params["means"]
        covs = possibly_batched_dist_params["covs"]
        priors = possibly_batched_dist_params["priors"]

        if len(means.shape) == 2:
            assert len(covs.shape) == 3
            assert len(priors.shape) == 1
            means = means[None, :, :]
            covs = covs[None, :, :, :]
            priors = priors[None, :]

        assert len(means.shape) == 3
        assert len(covs.shape) == 4
        assert len(priors.shape) == 2

        N, K, D = means.shape
        assert (
            len(covs.shape) == 4
            and covs.shape[0] == N
            and covs.shape[1] == K
            and covs.shape[2] == D
            and covs.shape[3] == D
        )
        assert len(priors.shape) == 2 and priors.shape[0] == N and priors.shape[1] == K
        assert means.device == covs.device == priors.device

        assert torch.all(priors >= 0)
        sum_priors = torch.sum(priors, dim=-1)
        assert torch.allclose(sum_priors, torch.ones_like(sum_priors))

        evals = torch.linalg.eigvalsh(covs)
        assert torch.all(
            evals >= -D * torch.finfo(evals.dtype).eps
        )  # Allow for numerical errors

    @classmethod
    def x0(
        cls,
        xt: torch.Tensor,
        t: torch.Tensor,
        sampler: Sampler,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        means = batched_dist_params["means"]  # (N, K, D)
        covs = batched_dist_params["covs"]  # (N, K, D, D)
        priors = batched_dist_params["priors"]  # (N, K)

        N, K, D = means.shape

        alpha = sampler.alpha(t)  # (N, )
        sigma = sampler.sigma(t)  # (N, )

        covs_t = (alpha[:, None, None, None] ** 2) * covs + (
            sigma[:, None, None, None] ** 2
        ) * torch.eye(D, device=xt.device)[None, None, :, :]  # (N, K, D, D)
        centered_x = xt[:, None, :] - alpha[:, None, None] * means  # (N, K, D)
        covs_t_inv_centered_x = vector_lstsq(covs_t, centered_x)  # (N, K, D)

        mahalanobis_dists = torch.sum(
            centered_x * covs_t_inv_centered_x, dim=-1
        )  # (N, K)
        logdets_covs_t = logdet_pd(covs_t)  # (N, K)
        w = (
            torch.log(priors) - 1 / 2 * logdets_covs_t - 1 / 2 * mahalanobis_dists
        )  # (N, K)
        softmax_w = torch.softmax(w, dim=-1)  # (N, K)

        weighted_normalized_x = torch.sum(
            softmax_w[:, :, None] * covs_t_inv_centered_x, dim=-2
        )  # (N, D)
        x0_hat = (1 / alpha[:, None]) * (
            xt - (sigma[:, None] ** 2) * weighted_normalized_x
        )  # (N, D)

        return x0_hat

    @classmethod
    def sample(
        cls,
        N: int,
        dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = dist_params["means"]  # (K, D)
        covs = dist_params["covs"]  # (K, D, D)
        priors = dist_params["priors"]  # (K, )

        K, D = means.shape

        device = priors.device
        y = torch.multinomial(priors, N, replacement=True)  # (N, )
        X = torch.empty((N, D), device=device)
        for k in range(K):
            idx = y == k
            X[idx] = (
                torch.randn((X[idx].shape[0], D), device=device) @ sqrt_psd(covs[k])
                + means[k][None, :]
            )
        return X, y


class IsoGMMDistribution(Distribution):
    """
    An isotropic (i.e., spherical variances) Gaussian Mixture Model (GMM) with K components.
    Formally, the distribution is defined as:

    mu(B) = sum_(i=1)^(K) pi_i * N(mu_i, tau_i^2 * I_D)(B)

    where mu_i is the mean of the ith component, tau is the standard deviation of the spherical variances,
    and pi_i is the prior probability of the ith component.

    Distribution Parameters:
        - means: A tensor of shape (K, D) containing the means of the components.
        - vars: A tensor of shape (K, ) containing the variances of the components.
        - priors: A tensor of shape (K, ) containing the prior probabilities of the components.

    Distribution Hyperparameters:
        - None
    """

    @classmethod
    def validate_params(
        cls, possibly_batched_dist_params: Dict[str, torch.Tensor]
    ) -> None:
        assert (
            "means" in possibly_batched_dist_params
            and "vars" in possibly_batched_dist_params
            and "priors" in possibly_batched_dist_params
        )
        means = possibly_batched_dist_params["means"]
        vars_ = possibly_batched_dist_params["vars"]
        priors = possibly_batched_dist_params["priors"]

        if len(means.shape) == 2:
            assert len(vars_.shape) == 1
            assert len(priors.shape) == 1
            means = means[None, :, :]
            vars_ = vars_[None, :]
            priors = priors[None, :]

        assert len(means.shape) == 3
        N, K, D = means.shape
        assert len(vars_.shape) == 2 and vars_.shape[0] == N and vars_.shape[1] == K
        assert len(priors.shape) == 2 and priors.shape[0] == N and priors.shape[1] == K
        assert means.device == vars_.device == priors.device

        priors_sum = torch.sum(priors, dim=-1)
        assert torch.all(priors_sum >= 0)
        assert torch.allclose(priors_sum, torch.ones_like(priors_sum))
        assert torch.all(
            vars_ >= -D * torch.finfo(vars_.dtype).eps
        )  # Allow for numerical errors

    @classmethod
    def x0(
        cls,
        xt: torch.Tensor,
        t: torch.Tensor,
        sampler: Sampler,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        means = batched_dist_params["means"]  # (N, K, D)
        vars_ = batched_dist_params["vars"]  # (N, K)
        priors = batched_dist_params["priors"]  # (N, K)

        N, K, D = means.shape

        alpha = sampler.alpha(t)  # (N, )
        sigma = sampler.sigma(t)  # (N, )

        vars_t = (alpha[:, None] ** 2) * vars_ + (sigma[:, None] ** 2)  # (N, K)
        centered_x = xt[:, None, :] - alpha[:, None, None] * means  # (N, K, D)
        vars_t_inv_centered_x = centered_x / vars_t[:, :, None]  # (N, K, D)

        mahalanobis_dists = torch.sum(
            centered_x * vars_t_inv_centered_x, dim=-1
        )  # (N, K)
        w = (
            torch.log(priors) - D / 2 * torch.log(vars_t) - 1 / 2 * mahalanobis_dists
        )  # (N, K)
        softmax_w = torch.softmax(w, dim=-1)  # (N, K)

        weighted_normalized_x = torch.sum(
            softmax_w[:, :, None] * vars_t_inv_centered_x, dim=-2
        )  # (N, D)
        x0_hat = (1 / alpha[:, None]) * (
            xt - (sigma[:, None] ** 2) * weighted_normalized_x
        )  # (N, D)

        return x0_hat

    @classmethod
    def sample(
        cls,
        N: int,
        dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = dist_params["means"]  # (K, D)
        vars_ = dist_params["vars"]  # (K, )
        priors = dist_params["priors"]  # (K, )

        K, D = means.shape
        covs = (
            torch.eye(D, device=vars_.device)[None, :, :].expand(K, -1, -1)
            * vars_[:, None, None]
        )
        return GMMDistribution.sample(
            N, {"means": means, "covs": covs, "priors": priors}, dict()
        )


class IsoHomoGMMDistribution(Distribution):
    """
    An isotropic homoscedastic (i.e., equal spherical variances) Gaussian Mixture Model (GMM) with K components.
    Formally, the distribution is defined as:

    mu(B) = sum_(i=1)^(K) pi_i * N(mu_i, tau^2 * I_D)(B)

    where mu_i is the mean of the ith component, tau is the standard deviation of the spherical variances,
    and pi_i is the prior probability of the ith component.

    Distribution Parameters:
        - means: A tensor of shape (K, D) containing the means of the components.
        - var: A tensor of shape () containing the variances of the components.
        - priors: A tensor of shape (K, ) containing the prior probabilities of the components.

    Distribution Hyperparameters:
        - None
    """

    @classmethod
    def validate_params(
        cls, possibly_batched_dist_params: Dict[str, torch.Tensor]
    ) -> None:
        assert (
            "means" in possibly_batched_dist_params
            and "var" in possibly_batched_dist_params
            and "priors" in possibly_batched_dist_params
        )
        means = possibly_batched_dist_params["means"]
        var = possibly_batched_dist_params["var"]
        priors = possibly_batched_dist_params["priors"]

        if len(means.shape) == 2:
            assert len(var.shape) == 0
            assert len(priors.shape) == 1
            means = means[None, :, :]
            var = var[None]
            priors = priors[None, :]

        assert len(means.shape) == 3
        N, K, D = means.shape
        assert len(var.shape) == 1 and var.shape[0] == N
        assert len(priors.shape) == 2 and priors.shape[0] == N and priors.shape[1] == K
        assert means.device == var.device == priors.device

        priors_sum = torch.sum(priors, dim=-1)
        assert torch.all(priors_sum >= 0)
        assert torch.allclose(priors_sum, torch.ones_like(priors_sum))
        assert torch.all(
            var >= -D * torch.finfo(var.dtype).eps
        )  # Allow for numerical errors

    @classmethod
    def x0(
        cls,
        xt: torch.Tensor,
        t: torch.Tensor,
        sampler: Sampler,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        means = batched_dist_params["means"]  # (N, K, D)
        var = batched_dist_params["var"]  # (N, )
        priors = batched_dist_params["priors"]  # (N, K)

        N, K, D = means.shape

        alpha = sampler.alpha(t)  # (N, )
        sigma = sampler.sigma(t)  # (N, )

        var_t = (alpha**2) * var + (sigma**2)  # (N, )
        centered_x = xt[:, None, :] - alpha[:, None, None] * means  # (N, K, D)
        vars_t_inv_centered_x = centered_x / var_t[:, None, None]  # (N, K, D)

        mahalanobis_dists = torch.sum(
            centered_x * vars_t_inv_centered_x, dim=-1
        )  # (N, K)
        w = torch.log(priors) - 1 / 2 * mahalanobis_dists  # (N, K)
        softmax_w = torch.softmax(w, dim=-1)  # (N, K)

        weighted_normalized_x = torch.sum(
            softmax_w[:, :, None] * vars_t_inv_centered_x, dim=-2
        )  # (N, D)
        x0_hat = (1 / alpha[:, None]) * (
            xt - (sigma[:, None] ** 2) * weighted_normalized_x
        )  # (N, D)

        return x0_hat

    @classmethod
    def sample(
        cls,
        N: int,
        dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = dist_params["means"]  # (K, D)
        var = dist_params["var"]  # ()
        priors = dist_params["priors"]  # (K, )

        K, D = means.shape
        covs = torch.eye(D, device=var.device)[None, :, :].expand(K, -1, -1) * var
        return GMMDistribution.sample(
            N, {"means": means, "covs": covs, "priors": priors}, dict()
        )


class LowRankGMMDistribution(Distribution):
    """
    A Gaussian Mixture Model (GMM) with K low-rank components.
    Formally, the distribution is defined as:

    mu(B) = sum_(i=1)^(K) pi_i * N(mu_i, Sigma_i)(B)

    where mu_i is the mean of the ith component, Sigma_i is the covariance matrix of the ith component,
    and pi_i is the prior probability of the ith component. Notably, Sigma_i is a low-rank matrix of the form

    Sigma_i =  A_i @ A_i^T

    Distribution Parameters:
        - means: A tensor of shape (K, D) containing the means of the components.
        - covs_factors: A tensor of shape (K, D, P) containing the tall factors of the covariance matrices of the components.
        - priors: A tensor of shape (K, ) containing the prior probabilities of the components.

    Distribution Hyperparameters:
        - None

    Note:
        - The covariance matrices are not explicitly stored, but rather computed as Sigma_i = A_i @ A_i^T.
        - The time and memory complexity is much lower in this class compared to the full GMM class, if and only if each covariance is low-rank (P << D).
    """

    @classmethod
    def validate_params(
        cls, possibly_batched_dist_params: Dict[str, torch.Tensor]
    ) -> None:
        assert (
            "means" in possibly_batched_dist_params
            and "covs_factors" in possibly_batched_dist_params
            and "priors" in possibly_batched_dist_params
        )
        means = possibly_batched_dist_params["means"]
        covs_factors = possibly_batched_dist_params["covs_factors"]
        priors = possibly_batched_dist_params["priors"]

        if len(means.shape) == 2:
            assert len(covs_factors.shape) == 3
            assert len(priors.shape) == 1
            means = means[None, :, :]
            covs_factors = covs_factors[None, :, :, :]
            priors = priors[None, :]

        assert len(means.shape) == 3
        assert len(covs_factors.shape) == 4
        assert len(priors.shape) == 2

        N, K, D, P = covs_factors.shape
        assert means.shape[0] == N and means.shape[1] == K and means.shape[2] == D
        assert len(priors.shape) == 2 and priors.shape[0] == N and priors.shape[1] == K
        assert means.device == covs_factors.device == priors.device

        assert torch.all(priors >= 0)
        sum_priors = torch.sum(priors, dim=-1)
        assert torch.allclose(sum_priors, torch.ones_like(sum_priors))

    @classmethod
    def x0(
        cls,
        xt: torch.Tensor,
        t: torch.Tensor,
        sampler: Sampler,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        means = batched_dist_params["means"]  # (N, K, D)
        covs_factors = batched_dist_params["covs_factors"]  # (N, K, D, R)
        priors = batched_dist_params["priors"]  # (N, K)

        N, K, D, P = covs_factors.shape
        covs_factors_T = covs_factors.transpose(-1, -2)  # (N, K, R, D)

        alpha = sampler.alpha(t)  # (N, )
        sigma = sampler.sigma(t)  # (N, )
        alpha_sigma_ratio_sq = (alpha / sigma) ** 2  # (N, )
        sigma_alpha_ratio_sq = 1 / alpha_sigma_ratio_sq  # (N, )

        internal_covs = covs_factors_T @ covs_factors  # (N, K, R, R)
        logdets_covs_t = 2 * D * torch.log(sigma[:, None]) + logdet_pd(
            torch.eye(P, device=covs_factors.device)[None, None, :, :]  # (1, 1, P, P)
            + alpha_sigma_ratio_sq[:, None, None, None] * internal_covs  # (N, K, P, P)
        )  # (N, K)

        centered_x = xt[:, None, :] - alpha[:, None, None] * means  # (N, K, D)
        covs_t_inv_centered_x = (1 / sigma[:, None, None] ** 2) * (
            centered_x  # (N, K, D)
            - (
                covs_factors  # (N, K, D, P)
                @ torch.linalg.lstsq(  # (N, K, P, 1)
                    internal_covs  # (N, K, P, P)
                    + sigma_alpha_ratio_sq[:, None, None, None]  # (N, K, 1, 1)
                    * torch.eye(P, device=internal_covs.device)[
                        None, None, :, :
                    ],  # (1, 1, P, P)
                    covs_factors_T @ centered_x[:, :, :, None],  # (N, K, P, 1)
                ).solution
            )[:, :, :, 0]  # (N, K, D)
        )  # (N, K, D)

        mahalanobis_dists = torch.sum(
            centered_x * covs_t_inv_centered_x, dim=-1
        )  # (N, K)
        w = (
            torch.log(priors) - 1 / 2 * logdets_covs_t - 1 / 2 * mahalanobis_dists
        )  # (N, K)
        softmax_w = torch.softmax(w, dim=-1)  # (N, K)

        weighted_normalized_x = torch.sum(
            softmax_w[:, :, None] * covs_t_inv_centered_x, dim=-2
        )  # (N, D)
        x0_hat = (1 / alpha[:, None]) * (
            xt - (sigma[:, None] ** 2) * weighted_normalized_x
        )  # (N, D)

        return x0_hat

    @classmethod
    def sample(
        cls,
        N: int,
        dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = dist_params["means"]  # (K, D)
        covs_factors = dist_params["covs_factors"]  # (K, D, P)
        priors = dist_params["priors"]  # (K, )

        K, D, P = covs_factors.shape

        device = priors.device
        y = torch.multinomial(priors, N, replacement=True)  # (N, )
        X = torch.empty((N, D), device=device)
        for k in range(K):
            idx = y == k
            X[idx] = (
                torch.randn((X[idx].shape[0], P), device=device) @ covs_factors[k].T
                + means[k][None, :]
            )
        return X, y
