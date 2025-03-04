import pytest
import torch
from diffusionlab.distributions.gmm import GMMDistribution
from diffusionlab.diffusions import DiffusionProcess


class TestGMM:
    """Tests for the full GMM Distribution."""

    def test_gmm_validation(self):
        """Test validation of parameters for GMM distribution."""
        D = 2
        K = 3
        N = 3

        # Test valid sampling parameters (non-batched)
        sampling_params = {
            "means": torch.randn(K, D),
            "covs": torch.stack([torch.eye(D)] * K),
            "priors": torch.ones(K) / K,
        }
        GMMDistribution.validate_params(sampling_params)

        # Test valid denoising parameters (batched)
        denoising_params = {
            "means": torch.randn(N, K, D),
            "covs": torch.stack([torch.eye(D)] * K)[None].expand(N, -1, -1, -1),
            "priors": torch.ones(N, K) / K,
        }
        GMMDistribution.validate_params(denoising_params)

        # Test error cases
        with pytest.raises(AssertionError):
            GMMDistribution.validate_params(
                {"means": torch.randn(K, D)}
            )  # Missing parameters

        invalid_params = sampling_params.copy()
        invalid_params["covs"] = torch.stack([torch.eye(3)] * K)  # Wrong dimension
        with pytest.raises(AssertionError):
            GMMDistribution.validate_params(invalid_params)

        invalid_params = denoising_params.copy()
        invalid_params["priors"] = torch.ones(N, K)  # Not normalized
        with pytest.raises(AssertionError):
            GMMDistribution.validate_params(invalid_params)

        invalid_params = denoising_params.copy()
        invalid_covs = torch.tensor([[1.0, 2.0], [2.0, 1.0]])[None, None].expand(
            N, K, -1, -1
        )
        invalid_params["covs"] = invalid_covs  # Non-positive definite
        with pytest.raises(AssertionError):
            GMMDistribution.validate_params(invalid_params)

    def test_gmm_sampling(self):
        """Test sampling from GMM distribution."""
        # Create non-batched GMM parameters for sampling
        D = 2  # dimension
        device = torch.device("cpu")

        means = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=device)
        covs = torch.stack(
            [
                torch.eye(D, device=device) * 0.1,
                torch.eye(D, device=device) * 0.2,
                torch.eye(D, device=device) * 0.3,
            ]
        )
        priors = torch.tensor([0.3, 0.3, 0.4], device=device)

        sampling_gmm_params = {
            "means": means,  # (K, D)
            "covs": covs,  # (K, D, D)
            "priors": priors,  # (K,)
        }

        N = 1000
        X, y = GMMDistribution.sample(N, sampling_gmm_params, {})

        # Check shapes and ranges
        assert X.shape == (N, sampling_gmm_params["means"].shape[1])
        assert y.shape == (N,)
        assert y.min() >= 0 and y.max() < sampling_gmm_params["means"].shape[0]

        # Check component proportions match priors
        for k in range(sampling_gmm_params["means"].shape[0]):
            count = (y == k).sum()
            ratio = count / N
            assert abs(ratio - sampling_gmm_params["priors"][k]) < 0.1

        # Check component distributions
        for k in range(sampling_gmm_params["means"].shape[0]):
            mask = y == k
            if mask.sum() > 0:
                component_samples = X[mask]
                mean = component_samples.mean(0)
                cov = torch.cov(component_samples.T)

                # Check statistics
                assert (
                    torch.norm(mean - sampling_gmm_params["means"][k])
                    / torch.norm(sampling_gmm_params["means"][k])
                    < 0.5
                )
                assert (
                    torch.norm(cov - sampling_gmm_params["covs"][k])
                    / torch.norm(sampling_gmm_params["covs"][k])
                    < 0.5
                )

    def test_gmm_x0_shape(self):
        """Test that the x0 method returns the correct shape."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        # Create batched GMM parameters for denoising
        N = 10  # batch size
        K = 3
        D = 2

        # Create batch of means
        means = torch.randn(N, K, D)

        # Create batch of covariances
        covs = torch.zeros(N, K, D, D)
        for i in range(N):
            for k in range(K):
                covs[i, k] = torch.eye(D) * (0.1 + 0.1 * k)

        # Create batch of priors
        priors = torch.softmax(torch.randn(N, K), dim=-1)

        denoising_gmm_params = {
            "means": means,  # (N, K, D)
            "covs": covs,  # (N, K, D, D)
            "priors": priors,  # (N, K)
        }

        x_t = torch.randn(N, D)
        t = torch.ones(N) * 0.5

        x0_hat = GMMDistribution.x0(x_t, t, diffusion_process, denoising_gmm_params, {})
        assert x0_hat.shape == (N, D)

    def test_gmm_vector_field_types(self):
        """Test vector field type conversions for GMM."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        # Create batched GMM parameters for denoising
        N = 10  # batch size
        K = 3
        D = 2

        # Create batch of means
        means = torch.randn(N, K, D)

        # Create batch of covariances
        covs = torch.zeros(N, K, D, D)
        for i in range(N):
            for k in range(K):
                covs[i, k] = torch.eye(D) * (0.1 + 0.1 * k)

        # Create batch of priors
        priors = torch.softmax(torch.randn(N, K), dim=-1)

        denoising_gmm_params = {
            "means": means,  # (N, K, D)
            "covs": covs,  # (N, K, D, D)
            "priors": priors,  # (N, K)
        }

        x_t = torch.randn(N, D)
        t = torch.ones(N) * 0.5

        # Test x0
        x0_hat = GMMDistribution.x0(x_t, t, diffusion_process, denoising_gmm_params, {})
        assert x0_hat.shape == (N, D)

        # Test eps
        eps_hat = GMMDistribution.eps(
            x_t, t, diffusion_process, denoising_gmm_params, {}
        )
        assert eps_hat.shape == (N, D)

        # Test v
        v_hat = GMMDistribution.v(x_t, t, diffusion_process, denoising_gmm_params, {})
        assert v_hat.shape == (N, D)

        # Test score
        score_hat = GMMDistribution.score(
            x_t, t, diffusion_process, denoising_gmm_params, {}
        )
        assert score_hat.shape == (N, D)

        # Check consistency
        x_from_x0 = (
            diffusion_process.alpha(t)[:, None] * x0_hat
            + diffusion_process.sigma(t)[:, None] * eps_hat
        )
        assert torch.allclose(x_t, x_from_x0, rtol=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gmm_device_movement(self):
        """Test GMM distribution works with different devices."""
        device = torch.device("cuda:0")

        # Create non-batched GMM parameters for sampling
        D = 2  # dimension
        K = 3

        means = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        covs = torch.stack(
            [
                torch.eye(D) * 0.1,
                torch.eye(D) * 0.2,
                torch.eye(D) * 0.3,
            ]
        )
        priors = torch.tensor([0.3, 0.3, 0.4])

        sampling_gmm_params = {
            "means": means,  # (K, D)
            "covs": covs,  # (K, D, D)
            "priors": priors,  # (K,)
        }

        # Create batched GMM parameters for denoising
        N = 10  # batch size

        # Create batch of means by adding random offsets
        means_offset = torch.randn(N, K, D) * 0.2
        means_batch = means[None, ...].expand(N, -1, -1) + means_offset

        # Create batch of covariances by scaling the base covariances
        cov_scales = torch.exp(torch.randn(N, K) * 0.2)  # Random positive scales
        covs_batch = covs[None, ...].expand(N, -1, -1, -1) * cov_scales[..., None, None]

        # Create batch of priors by perturbing and renormalizing
        priors_logits = (
            torch.log(priors)[None, ...].expand(N, -1) + torch.randn(N, K) * 0.2
        )
        priors_batch = torch.softmax(priors_logits, dim=-1)

        denoising_gmm_params = {
            "means": means_batch,  # (N, K, D)
            "covs": covs_batch,  # (N, K, D, D)
            "priors": priors_batch,  # (N, K)
        }

        # Test sampling
        cuda_sampling_params = {k: v.to(device) for k, v in sampling_gmm_params.items()}
        N_samples = 10
        X, y = GMMDistribution.sample(N_samples, cuda_sampling_params, {})
        assert X.device == device
        assert y.device == device

        # Test denoising
        cuda_denoising_params = {
            k: v.to(device) for k, v in denoising_gmm_params.items()
        }

        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        x = torch.randn(N, D, device=device)
        t = torch.ones(N, device=device) * 0.5

        x0_hat = GMMDistribution.x0(x, t, diffusion_process, cuda_denoising_params, {})
        assert x0_hat.device == device

    def test_gmm_numerical_stability(self):
        """Test numerical stability in edge cases for GMM."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        priors = torch.ones(2) / 2
        N = 10  # batch size for denoising

        # Test with very small covariances
        covs = torch.stack([torch.eye(2) * 1e-10] * 2)
        denoising_params = {
            "means": means[None].expand(N, -1, -1),
            "covs": covs[None].expand(N, -1, -1, -1),
            "priors": priors[None].expand(N, -1),
        }

        N_test = 10
        x = torch.randn(N_test, 2)
        t = torch.ones(N_test) * 0.5

        x0_hat = GMMDistribution.x0(x, t, diffusion_process, denoising_params, {})
        assert not torch.any(torch.isnan(x0_hat))
        assert not torch.any(torch.isinf(x0_hat))
        assert torch.all(torch.abs(x0_hat) < 100)

        # Test with very large covariances
        denoising_params["covs"] = denoising_params["covs"] * 1e20
        x0_hat = GMMDistribution.x0(x, t, diffusion_process, denoising_params, {})
        assert not torch.any(torch.isnan(x0_hat))
        assert not torch.any(torch.isinf(x0_hat))
        assert torch.all(torch.abs(x0_hat) < 100)
