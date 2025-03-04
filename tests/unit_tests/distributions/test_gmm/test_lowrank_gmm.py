import pytest
import torch
from diffusionlab.distributions.gmm import (
    GMMDistribution,
    LowRankGMMDistribution,
)
from diffusionlab.diffusions import DiffusionProcess


class TestLowRankGMM:
    """Tests for the Low-Rank GMM Distribution."""

    def test_low_rank_gmm_validation(self):
        """Test validation of parameters for low-rank GMM."""
        D = 2
        K = 3
        R = 1
        N = 3

        # Test valid sampling parameters (non-batched)
        sampling_params = {
            "means": torch.randn(K, D),
            "covs_factors": torch.randn(K, D, R),
            "priors": torch.ones(K) / K,
        }
        LowRankGMMDistribution.validate_params(sampling_params)

        # Test valid denoising parameters (batched)
        denoising_params = {
            "means": torch.randn(N, K, D),
            "covs_factors": torch.randn(N, K, D, R),
            "priors": torch.ones(N, K) / K,
        }
        LowRankGMMDistribution.validate_params(denoising_params)

        # Test error cases
        with pytest.raises(AssertionError):
            LowRankGMMDistribution.validate_params(
                {"means": torch.randn(K, D)}
            )  # Missing parameters

        invalid_params = sampling_params.copy()
        invalid_params["covs_factors"] = torch.randn(K, D + 1, R)  # Wrong dimension
        with pytest.raises(AssertionError):
            LowRankGMMDistribution.validate_params(invalid_params)

        invalid_params = denoising_params.copy()
        invalid_params["priors"] = torch.ones(N, K)  # Not normalized
        with pytest.raises(AssertionError):
            LowRankGMMDistribution.validate_params(invalid_params)

    def test_low_rank_gmm_sampling(self):
        """Test sampling from low-rank GMM."""
        # Create non-batched low-rank GMM parameters for sampling
        D = 2  # dimension
        device = torch.device("cpu")

        means = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=device)
        # Create low-rank factors that would result in diagonal covariances
        covs_factors = torch.stack(
            [
                torch.tensor([[0.3162]], device=device).expand(1, D).T,  # sqrt(0.1)
                torch.tensor([[0.4472]], device=device).expand(1, D).T,  # sqrt(0.2)
                torch.tensor([[0.5477]], device=device).expand(1, D).T,  # sqrt(0.3)
            ]
        )  # (K, D, R)
        priors = torch.tensor([0.3, 0.3, 0.4], device=device)

        sampling_low_rank_gmm_params = {
            "means": means,  # (K, D)
            "covs_factors": covs_factors,  # (K, D, P)
            "priors": priors,  # (K,)
        }

        N = 1000
        X, y = LowRankGMMDistribution.sample(N, sampling_low_rank_gmm_params, {})

        # Check shapes and ranges
        assert X.shape == (N, sampling_low_rank_gmm_params["means"].shape[1])
        assert y.shape == (N,)
        assert y.min() >= 0 and y.max() < sampling_low_rank_gmm_params["means"].shape[0]

        # Check component proportions match priors
        for k in range(sampling_low_rank_gmm_params["means"].shape[0]):
            count = (y == k).sum()
            ratio = count / N
            assert abs(ratio - sampling_low_rank_gmm_params["priors"][k]) < 0.1

        # Check component distributions
        for k in range(sampling_low_rank_gmm_params["means"].shape[0]):
            mask = y == k
            if mask.sum() > 0:
                component_samples = X[mask]
                mean = component_samples.mean(0)
                cov = torch.cov(component_samples.T)

                # Check statistics
                assert torch.allclose(
                    mean, sampling_low_rank_gmm_params["means"][k], atol=0.5
                )
                covs_factors = sampling_low_rank_gmm_params["covs_factors"][k]  # (D, P)
                expected_cov = covs_factors @ covs_factors.T
                assert torch.allclose(cov, expected_cov, atol=0.5)

    def test_low_rank_gmm_equals_full_gmm(self):
        """Test that low-rank GMM equals full GMM when rank equals dimension."""
        N = 10
        D = 3
        K = 2
        R = D  # Full rank

        # Create parameters
        means = torch.randn(N, K, D)
        priors = torch.softmax(torch.randn(N, K), dim=-1)

        # Create low-rank parameters
        covs_factors = torch.randn(N, K, D, R)
        low_rank_params = {
            "means": means,
            "covs_factors": covs_factors,
            "priors": priors,
        }

        # Create full-rank parameters (covs = covs_factors @ covs_factors.T)
        covs = torch.zeros(N, K, D, D)
        for i in range(N):
            for k in range(K):
                covs[i, k] = covs_factors[i, k] @ covs_factors[i, k].transpose(-1, -2)

        full_params = {
            "means": means,
            "covs": covs,
            "priors": priors,
        }

        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        x_t = torch.randn(N, D)
        t = torch.ones(N) * 0.5

        # Test x0
        x0_low_rank = LowRankGMMDistribution.x0(
            x_t, t, diffusion_process, low_rank_params, {}
        )
        x0_full = GMMDistribution.x0(x_t, t, diffusion_process, full_params, {})
        assert torch.allclose(x0_low_rank, x0_full, atol=1e-5)

        # Test eps
        eps_low_rank = LowRankGMMDistribution.eps(
            x_t, t, diffusion_process, low_rank_params, {}
        )
        eps_full = GMMDistribution.eps(x_t, t, diffusion_process, full_params, {})
        assert torch.allclose(eps_low_rank, eps_full, atol=1e-5)

        # Test v
        v_low_rank = LowRankGMMDistribution.v(
            x_t, t, diffusion_process, low_rank_params, {}
        )
        v_full = GMMDistribution.v(x_t, t, diffusion_process, full_params, {})
        assert torch.allclose(v_low_rank, v_full, atol=1e-5)

        # Test score
        score_low_rank = LowRankGMMDistribution.score(
            x_t, t, diffusion_process, low_rank_params, {}
        )
        score_full = GMMDistribution.score(x_t, t, diffusion_process, full_params, {})
        assert torch.allclose(score_low_rank, score_full, atol=1e-5)

    def test_low_rank_gmm_x0_shape(self):
        """Test x0 prediction shape for low-rank GMM."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        # Create batched low-rank GMM parameters for denoising
        N = 10  # batch size
        K = 3
        D = 2
        R = 1

        # Create batch of means
        means = torch.randn(N, K, D)

        # Create batch of factors
        covs_factors = torch.randn(N, K, D, R)

        # Create batch of priors
        priors = torch.softmax(torch.randn(N, K), dim=-1)

        denoising_low_rank_gmm_params = {
            "means": means,  # (N, K, D)
            "covs_factors": covs_factors,  # (N, K, D, P)
            "priors": priors,  # (N, K)
        }

        x_t = torch.randn(N, D)
        t = torch.ones(N) * 0.5

        x0_hat = LowRankGMMDistribution.x0(
            x_t, t, diffusion_process, denoising_low_rank_gmm_params, {}
        )
        assert x0_hat.shape == (N, D)

    def test_low_rank_gmm_vector_field_types(self):
        """Test all vector field types work correctly for low-rank GMM."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        # Create batched low-rank GMM parameters for denoising
        N = 10  # batch size
        K = 3
        D = 2
        R = 1

        # Create batch of means
        means = torch.randn(N, K, D)

        # Create batch of factors
        covs_factors = torch.randn(N, K, D, R)

        # Create batch of priors
        priors = torch.softmax(torch.randn(N, K), dim=-1)

        denoising_low_rank_gmm_params = {
            "means": means,  # (N, K, D)
            "covs_factors": covs_factors,  # (N, K, D, P)
            "priors": priors,  # (N, K)
        }

        x_t = torch.randn(N, D)
        t = torch.ones(N) * 0.5

        # Test each vector field type
        x0_hat = LowRankGMMDistribution.x0(
            x_t, t, diffusion_process, denoising_low_rank_gmm_params, {}
        )
        eps_hat = LowRankGMMDistribution.eps(
            x_t, t, diffusion_process, denoising_low_rank_gmm_params, {}
        )
        v_hat = LowRankGMMDistribution.v(
            x_t, t, diffusion_process, denoising_low_rank_gmm_params, {}
        )
        score_hat = LowRankGMMDistribution.score(
            x_t, t, diffusion_process, denoising_low_rank_gmm_params, {}
        )

        # Check shapes
        assert x0_hat.shape == (N, D)
        assert eps_hat.shape == (N, D)
        assert v_hat.shape == (N, D)
        assert score_hat.shape == (N, D)

        # Check consistency
        x_from_x0 = (
            diffusion_process.alpha(t)[:, None] * x0_hat
            + diffusion_process.sigma(t)[:, None] * eps_hat
        )
        assert torch.allclose(x_t, x_from_x0, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_low_rank_gmm_device_movement(self):
        """Test low-rank GMM distribution works with different devices."""
        device = torch.device("cuda:0")

        # Create non-batched low-rank GMM parameters for sampling
        D = 2  # dimension
        K = 3

        means = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        # Create low-rank factors
        covs_factors = torch.stack(
            [
                torch.tensor([[0.3162]]).expand(1, D).T,  # sqrt(0.1)
                torch.tensor([[0.4472]]).expand(1, D).T,  # sqrt(0.2)
                torch.tensor([[0.5477]]).expand(1, D).T,  # sqrt(0.3)
            ]
        )  # (K, D, R)
        priors = torch.tensor([0.3, 0.3, 0.4])

        sampling_low_rank_gmm_params = {
            "means": means,  # (K, D)
            "covs_factors": covs_factors,  # (K, D, P)
            "priors": priors,  # (K,)
        }

        # Create batched low-rank GMM parameters for denoising
        N = 10  # batch size

        # Create batch of means by adding random offsets
        means_offset = torch.randn(N, K, D) * 0.2
        means_batch = means[None, ...].expand(N, -1, -1) + means_offset

        # Create batch of factors by scaling the base factors
        covs_factors_scales = torch.exp(torch.randn(N, K) * 0.2)[
            :, :, None, None
        ]  # Random positive scales
        covs_factors_batch = (
            covs_factors[None, ...].expand(N, -1, -1, -1) * covs_factors_scales
        )

        # Create batch of priors by perturbing and renormalizing
        priors_logits = (
            torch.log(priors)[None, ...].expand(N, -1) + torch.randn(N, K) * 0.2
        )
        priors_batch = torch.softmax(priors_logits, dim=-1)

        denoising_low_rank_gmm_params = {
            "means": means_batch,  # (N, K, D)
            "covs_factors": covs_factors_batch,  # (N, K, D, P)
            "priors": priors_batch,  # (N, K)
        }

        # Test sampling
        cuda_sampling_params = {
            k: v.to(device) for k, v in sampling_low_rank_gmm_params.items()
        }
        N_samples = 10
        X, y = LowRankGMMDistribution.sample(N_samples, cuda_sampling_params, {})
        assert X.device == device
        assert y.device == device

        # Test denoising
        cuda_denoising_params = {
            k: v.to(device) for k, v in denoising_low_rank_gmm_params.items()
        }

        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        x = torch.randn(N, D, device=device)
        t = torch.ones(N, device=device) * 0.5

        x0_hat = LowRankGMMDistribution.x0(
            x, t, diffusion_process, cuda_denoising_params, {}
        )
        assert x0_hat.device == device

    def test_low_rank_gmm_numerical_stability(self):
        """Test numerical stability in edge cases for low-rank GMM."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        priors = torch.ones(2) / 2
        N = 10  # batch size for denoising

        # Test with very small factors
        covs_factors = torch.ones(2, 2, 1) * 1e-5
        denoising_params = {
            "means": means[None].expand(N, -1, -1),
            "covs_factors": covs_factors[None].expand(N, -1, -1, -1),
            "priors": priors[None].expand(N, -1),
        }

        N_test = 10
        x = torch.randn(N_test, 2)
        t = torch.ones(N_test) * 0.5

        x0_hat = LowRankGMMDistribution.x0(
            x, t, diffusion_process, denoising_params, {}
        )
        assert not torch.any(torch.isnan(x0_hat))
        assert not torch.any(torch.isinf(x0_hat))
        assert torch.all(torch.abs(x0_hat) < 100)

        # Test with very large factors
        denoising_params["covs_factors"] = denoising_params["covs_factors"] * 1e10
        x0_hat = LowRankGMMDistribution.x0(
            x, t, diffusion_process, denoising_params, {}
        )
        assert not torch.any(torch.isnan(x0_hat))
        assert not torch.any(torch.isinf(x0_hat))
        assert torch.all(torch.abs(x0_hat) < 100)
