import torch
import pytest
from diffusionlab.diffusions import (
    DiffusionProcess,
    VarianceExplodingProcess,
    OrnsteinUhlenbeckProcess,
    FlowMatchingProcess,
)


class TestDiffusionProcess:
    def test_initialization(self):
        """Test basic initialization of DiffusionProcess."""
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t

        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        assert diffusion_process.alpha is alpha
        assert diffusion_process.sigma is sigma
        # Check that alpha_prime and sigma_prime are callable
        assert callable(diffusion_process.alpha_prime)
        assert callable(diffusion_process.sigma_prime)

        # Test the derivatives with a sample input
        t = torch.tensor([0.5])
        # For constant alpha, derivative should be zero
        assert torch.allclose(diffusion_process.alpha_prime(t), torch.zeros_like(t))
        # For linear sigma, derivative should be one
        assert torch.allclose(diffusion_process.sigma_prime(t), torch.ones_like(t))

    def test_initialization_missing_params(self):
        """Test that initialization fails when required parameters are missing."""
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t

        with pytest.raises(AssertionError):
            DiffusionProcess(alpha=alpha)

        with pytest.raises(AssertionError):
            DiffusionProcess(sigma=sigma)

    def test_forward(self):
        """Test the forward diffusion process."""
        alpha = lambda t: 1 - t
        sigma = lambda t: t

        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Test with scalar t
        batch_size = 10
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        t = torch.ones(batch_size) * 0.5
        eps = torch.randn(batch_size, data_dim)

        noisy_x = diffusion_process.forward(x, t, eps)

        # Check shape
        assert noisy_x.shape == x.shape

        # Check values
        expected = (1 - t).unsqueeze(-1) * x + t.unsqueeze(-1) * eps
        assert torch.allclose(noisy_x, expected)

        # Test with batched t of different shape
        t = torch.linspace(0, 1, batch_size)
        noisy_x = diffusion_process.forward(x, t, eps)

        expected = (1 - t).unsqueeze(-1) * x + t.unsqueeze(-1) * eps
        assert torch.allclose(noisy_x, expected)


class TestVarianceExplodingProcess:
    def test_initialization(self):
        """Test initialization of VarianceExplodingProcess."""
        sigma = lambda t: t

        diffusion_process = VarianceExplodingProcess(sigma)

        # Check that alpha is always 1
        t = torch.linspace(0, 1, 10)
        assert torch.allclose(diffusion_process.alpha(t), torch.ones_like(t))
        assert torch.allclose(diffusion_process.sigma(t), t)

    def test_forward(self):
        """Test the forward process of VarianceExplodingProcess."""
        sigma = lambda t: t

        diffusion_process = VarianceExplodingProcess(sigma)

        batch_size = 10
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        t = torch.ones(batch_size) * 0.5
        eps = torch.randn(batch_size, data_dim)

        noisy_x = diffusion_process.forward(x, t, eps)

        # Check shape
        assert noisy_x.shape == x.shape

        # Check values - for VE, x_t = x_0 + sigma(t) * eps
        expected = x + t.unsqueeze(-1) * eps
        assert torch.allclose(noisy_x, expected)


class TestOrnsteinUhlenbeckProcess:
    def test_initialization(self):
        """Test initialization of OrnsteinUhlenbeckProcess."""
        diffusion_process = OrnsteinUhlenbeckProcess()

        # Check alpha and sigma functions
        t = torch.linspace(0, 1, 10)
        assert torch.allclose(diffusion_process.alpha(t), torch.sqrt(1 - t**2))
        assert torch.allclose(diffusion_process.sigma(t), t)

    def test_forward(self):
        """Test the forward process of OrnsteinUhlenbeckProcess."""
        diffusion_process = OrnsteinUhlenbeckProcess()

        batch_size = 10
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        t = torch.ones(batch_size) * 0.5
        eps = torch.randn(batch_size, data_dim)

        noisy_x = diffusion_process.forward(x, t, eps)

        # Check shape
        assert noisy_x.shape == x.shape

        # Check values - for OU, x_t = sqrt(1 - t^2) * x_0 + t * eps
        expected = torch.sqrt(1 - t**2).unsqueeze(-1) * x + t.unsqueeze(-1) * eps
        assert torch.allclose(noisy_x, expected)


class TestFlowMatchingProcess:
    def test_initialization(self):
        """Test initialization of FlowMatchingProcess."""
        diffusion_process = FlowMatchingProcess()

        # Check alpha and sigma functions
        t = torch.linspace(0, 1, 10)
        assert torch.allclose(diffusion_process.alpha(t), 1 - t)
        assert torch.allclose(diffusion_process.sigma(t), t)

    def test_forward(self):
        """Test the forward process of FlowMatchingProcess."""
        diffusion_process = FlowMatchingProcess()

        batch_size = 10
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        t = torch.ones(batch_size) * 0.5
        eps = torch.randn(batch_size, data_dim)

        noisy_x = diffusion_process.forward(x, t, eps)

        # Check shape
        assert noisy_x.shape == x.shape

        # Check values - for FM, x_t = (1 - t) * x_0 + t * eps
        expected = (1 - t).unsqueeze(-1) * x + t.unsqueeze(-1) * eps
        assert torch.allclose(noisy_x, expected)
