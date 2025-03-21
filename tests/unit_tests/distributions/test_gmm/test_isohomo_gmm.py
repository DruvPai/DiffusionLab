import pytest
import torch
from diffusionlab.distributions.gmm import IsoHomoGMMDistribution, GMMDistribution
from diffusionlab.diffusions import DiffusionProcess


class TestIsoHomoGMM:
    """Tests for the Isotropic Homogeneous GMM Distribution."""

    def test_iso_homo_gmm_validation(self):
        """Test validation of parameters for isotropic homogeneous GMM."""
        D = 2
        K = 3
        N = 3

        # Test valid sampling parameters (non-batched)
        sampling_params = {
            "means": torch.randn(K, D),
            "var": torch.tensor(0.1),
            "priors": torch.ones(K) / K,
        }
        IsoHomoGMMDistribution.validate_params(sampling_params)

        # Test valid denoising parameters (batched)
        denoising_params = {
            "means": torch.randn(N, K, D),
            "var": torch.full((N,), 0.1),
            "priors": torch.ones(N, K) / K,
        }
        IsoHomoGMMDistribution.validate_params(denoising_params)

        # Test error cases
        with pytest.raises(AssertionError):
            IsoHomoGMMDistribution.validate_params(
                {"means": torch.randn(K, D)}
            )  # Missing parameters

        invalid_params = sampling_params.copy()
        invalid_params["var"] = torch.tensor(-1.0)  # Negative variance
        with pytest.raises(AssertionError):
            IsoHomoGMMDistribution.validate_params(invalid_params)

        invalid_params = denoising_params.copy()
        invalid_params["priors"] = torch.ones(N, K)  # Not normalized
        with pytest.raises(AssertionError):
            IsoHomoGMMDistribution.validate_params(invalid_params)

    def test_iso_homo_gmm_equals_full_gmm(self):
        """Test that isotropic homogeneous GMM equals full GMM when covariance matrices are all diagonal with the same value on each diagonal element and across all components."""
        N = 10
        D = 3
        K = 2

        # Create parameters
        means = torch.randn(N, K, D)
        priors = torch.softmax(torch.randn(N, K), dim=-1)

        # Create isotropic homogeneous GMM parameters
        var = torch.exp(torch.randn(N))  # Random positive values, one per batch
        iso_homo_params = {
            "means": means,
            "var": var,
            "priors": priors,
        }

        # Create full-rank parameters with diagonal covariances
        covs = torch.zeros(N, K, D, D)
        for i in range(N):
            for k in range(K):
                # Create diagonal covariance matrix with the same value on each diagonal element
                # and the same across all components in the batch
                covs[i, k] = torch.eye(D) * var[i]

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
        x0_iso_homo = IsoHomoGMMDistribution.x0(
            x_t, t, diffusion_process, iso_homo_params, {}
        )
        x0_full = GMMDistribution.x0(x_t, t, diffusion_process, full_params, {})
        assert torch.allclose(x0_iso_homo, x0_full, atol=1e-5)

        # Test eps
        eps_iso_homo = IsoHomoGMMDistribution.eps(
            x_t, t, diffusion_process, iso_homo_params, {}
        )
        eps_full = GMMDistribution.eps(x_t, t, diffusion_process, full_params, {})
        assert torch.allclose(eps_iso_homo, eps_full, atol=1e-5)

        # Test v
        v_iso_homo = IsoHomoGMMDistribution.v(
            x_t, t, diffusion_process, iso_homo_params, {}
        )
        v_full = GMMDistribution.v(x_t, t, diffusion_process, full_params, {})
        assert torch.allclose(v_iso_homo, v_full, atol=1e-5)

        # Test score
        score_iso_homo = IsoHomoGMMDistribution.score(
            x_t, t, diffusion_process, iso_homo_params, {}
        )
        score_full = GMMDistribution.score(x_t, t, diffusion_process, full_params, {})
        assert torch.allclose(score_iso_homo, score_full, atol=1e-5)

    def test_iso_homo_gmm_sampling(self):
        """Test sampling from isotropic homogeneous GMM."""
        # Create non-batched isotropic homogeneous GMM parameters for sampling
        device = torch.device("cpu")

        means = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=device)
        priors = torch.tensor([0.3, 0.3, 0.4], device=device)
        var = torch.tensor(0.2, device=device)

        sampling_iso_homo_gmm_params = {
            "means": means,  # (K, D)
            "var": var,  # ()
            "priors": priors,  # (K,)
        }

        N = 1000
        X, y = IsoHomoGMMDistribution.sample(N, sampling_iso_homo_gmm_params, {})

        # Check shapes and ranges
        assert X.shape == (N, sampling_iso_homo_gmm_params["means"].shape[1])
        assert y.shape == (N,)
        assert y.min() >= 0 and y.max() < sampling_iso_homo_gmm_params["means"].shape[0]

        # Check component proportions match priors
        for k in range(sampling_iso_homo_gmm_params["means"].shape[0]):
            count = (y == k).sum()
            ratio = count / N
            assert abs(ratio - sampling_iso_homo_gmm_params["priors"][k]) < 0.1

        # Check component distributions
        for k in range(sampling_iso_homo_gmm_params["means"].shape[0]):
            mask = y == k
            if mask.sum() > 0:
                component_samples = X[mask]
                mean = component_samples.mean(0)
                cov = torch.cov(component_samples.T)

                # Check statistics
                assert torch.allclose(
                    mean, sampling_iso_homo_gmm_params["means"][k], atol=0.5
                )
                expected_cov = torch.eye(2) * sampling_iso_homo_gmm_params["var"]
                assert torch.allclose(cov, expected_cov, atol=0.5)

    def test_iso_homo_gmm_x0_shape(self):
        """Test x0 prediction shape for isotropic homogeneous GMM."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        # Create batched isotropic homogeneous GMM parameters for denoising
        N = 10  # batch size
        K = 3
        D = 2

        # Create batch of means
        means = torch.randn(N, K, D)

        # Create batch of variances
        var = torch.exp(torch.randn(N) * 0.2)  # Random positive values

        # Create batch of priors
        priors = torch.softmax(torch.randn(N, K), dim=-1)

        denoising_iso_homo_gmm_params = {
            "means": means,  # (N, K, D)
            "var": var,  # (N,)
            "priors": priors,  # (N, K)
        }

        x_t = torch.randn(N, D)
        t = torch.ones(N) * 0.5

        x0_hat = IsoHomoGMMDistribution.x0(
            x_t, t, diffusion_process, denoising_iso_homo_gmm_params, {}
        )
        assert x0_hat.shape == (N, D)

    def test_iso_homo_gmm_vector_field_types(self):
        """Test all vector field types work correctly for isotropic homogeneous GMM."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        # Create batched isotropic homogeneous GMM parameters for denoising
        N = 10  # batch size
        K = 3
        D = 2

        # Create batch of means
        means = torch.randn(N, K, D)

        # Create batch of variances
        var = torch.exp(torch.randn(N) * 0.2)  # Random positive values

        # Create batch of priors
        priors = torch.softmax(torch.randn(N, K), dim=-1)

        denoising_iso_homo_gmm_params = {
            "means": means,  # (N, K, D)
            "var": var,  # (N,)
            "priors": priors,  # (N, K)
        }

        x_t = torch.randn(N, D)
        t = torch.ones(N) * 0.5

        # Test each vector field type
        x0_hat = IsoHomoGMMDistribution.x0(
            x_t, t, diffusion_process, denoising_iso_homo_gmm_params, {}
        )
        eps_hat = IsoHomoGMMDistribution.eps(
            x_t, t, diffusion_process, denoising_iso_homo_gmm_params, {}
        )
        v_hat = IsoHomoGMMDistribution.v(
            x_t, t, diffusion_process, denoising_iso_homo_gmm_params, {}
        )
        score_hat = IsoHomoGMMDistribution.score(
            x_t, t, diffusion_process, denoising_iso_homo_gmm_params, {}
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
    def test_iso_homo_gmm_device_movement(self):
        """Test isotropic homogeneous GMM distribution works with different devices."""
        device = torch.device("cuda:0")

        # Create non-batched isotropic homogeneous GMM parameters for sampling
        D = 2  # dimension
        K = 3

        means = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]])
        priors = torch.tensor([0.3, 0.3, 0.4])
        var = torch.tensor(0.2)

        sampling_iso_homo_gmm_params = {
            "means": means,  # (K, D)
            "var": var,  # ()
            "priors": priors,  # (K,)
        }

        # Create batched isotropic homogeneous GMM parameters for denoising
        N = 10  # batch size

        # Create batch of means by adding random offsets
        means_offset = torch.randn(N, K, D) * 0.2
        means_batch = means[None, ...].expand(N, -1, -1) + means_offset

        # Create batch of variances by scaling the base variance
        var_scales = torch.exp(torch.randn(N) * 0.2)  # Random positive scales
        var_batch = var * var_scales

        # Create batch of priors by perturbing and renormalizing
        priors_logits = (
            torch.log(priors)[None, ...].expand(N, -1) + torch.randn(N, K) * 0.2
        )
        priors_batch = torch.softmax(priors_logits, dim=-1)

        denoising_iso_homo_gmm_params = {
            "means": means_batch,  # (N, K, D)
            "var": var_batch,  # (N,)
            "priors": priors_batch,  # (N, K)
        }

        # Test sampling
        cuda_sampling_params = {
            k: v.to(device) for k, v in sampling_iso_homo_gmm_params.items()
        }
        N_samples = 10
        X, y = IsoHomoGMMDistribution.sample(N_samples, cuda_sampling_params, {})
        assert X.device == device
        assert y.device == device

        # Test denoising
        cuda_denoising_params = {
            k: v.to(device) for k, v in denoising_iso_homo_gmm_params.items()
        }

        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        x = torch.randn(N, D, device=device)
        t = torch.ones(N, device=device) * 0.5

        x0_hat = IsoHomoGMMDistribution.x0(
            x, t, diffusion_process, cuda_denoising_params, {}
        )
        assert x0_hat.device == device

    def test_iso_homo_gmm_numerical_stability(self):
        """Test numerical stability in edge cases for isotropic homogeneous GMM."""
        # Create diffusion process
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.cos((t * torch.pi) / 2),
            sigma=lambda t: torch.sin((t * torch.pi) / 2),
        )

        means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        priors = torch.ones(2) / 2
        N = 10  # batch size for denoising

        # Test with very small variance
        denoising_params = {
            "means": means[None].expand(N, -1, -1),
            "var": torch.tensor(1e-10).expand(N),
            "priors": priors[None].expand(N, -1),
        }

        N_test = 10
        x = torch.randn(N_test, 2)
        t = torch.ones(N_test) * 0.5

        x0_hat = IsoHomoGMMDistribution.x0(
            x, t, diffusion_process, denoising_params, {}
        )
        assert not torch.any(torch.isnan(x0_hat))
        assert not torch.any(torch.isinf(x0_hat))
        assert torch.all(torch.abs(x0_hat) < 100)

        # Test with very large variance
        denoising_params["var"] = denoising_params["var"] * 1e20
        x0_hat = IsoHomoGMMDistribution.x0(
            x, t, diffusion_process, denoising_params, {}
        )
        assert not torch.any(torch.isnan(x0_hat))
        assert not torch.any(torch.isinf(x0_hat))
        assert torch.all(torch.abs(x0_hat) < 100)
