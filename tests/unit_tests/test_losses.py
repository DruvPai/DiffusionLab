import torch
import pytest
from diffusionlab.losses import SamplewiseDiffusionLoss
from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType
from diffusionlab.utils import pad_shape_back


class TestSamplewiseDiffusionLoss:
    """Tests for the SamplewiseDiffusionLoss class."""

    def test_initialization_with_x0_target(self):
        """Test initialization with X0 target type."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with X0 target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

        # Check attributes
        assert loss_fn.diffusion_process is diffusion_process
        assert loss_fn.target_type == VectorFieldType.X0
        assert callable(loss_fn.target)

        # Test target function
        batch_size = 4
        channels = 3
        height = width = 8

        x_t = torch.randn(batch_size, channels, height, width)
        f_x_t = torch.randn(batch_size, channels, height, width)
        x_0 = torch.randn(batch_size, channels, height, width)
        eps = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)

        # For X0 target type, target should be x_0
        target = loss_fn.target(x_t, f_x_t, x_0, eps, t)
        assert torch.allclose(target, x_0)

    def test_initialization_with_eps_target(self):
        """Test initialization with EPS target type."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with EPS target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.EPS)

        # Check attributes
        assert loss_fn.diffusion_process is diffusion_process
        assert loss_fn.target_type == VectorFieldType.EPS
        assert callable(loss_fn.target)

        # Test target function
        batch_size = 4
        channels = 3
        height = width = 8

        x_t = torch.randn(batch_size, channels, height, width)
        f_x_t = torch.randn(batch_size, channels, height, width)
        x_0 = torch.randn(batch_size, channels, height, width)
        eps = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)

        # For EPS target type, target should be eps
        target = loss_fn.target(x_t, f_x_t, x_0, eps, t)
        assert torch.allclose(target, eps)

    def test_initialization_with_v_target(self):
        """Test initialization with V target type."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with V target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.V)

        # Check attributes
        assert loss_fn.diffusion_process is diffusion_process
        assert loss_fn.target_type == VectorFieldType.V
        assert callable(loss_fn.target)

        # Test target function
        batch_size = 4
        channels = 3
        height = width = 8

        x_t = torch.randn(batch_size, channels, height, width)
        f_x_t = torch.randn(batch_size, channels, height, width)
        x_0 = torch.randn(batch_size, channels, height, width)
        eps = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)

        # For V target type, target should be calculated using alpha_prime and sigma_prime
        expected_target = (
            pad_shape_back(diffusion_process.alpha_prime(t), x_0.shape) * x_0
            + pad_shape_back(diffusion_process.sigma_prime(t), x_0.shape) * eps
        )

        target = loss_fn.target(x_t, f_x_t, x_0, eps, t)
        assert torch.allclose(target, expected_target)

    def test_initialization_with_score_target(self):
        """Test that initialization with SCORE target type raises ValueError."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with SCORE target type should raise ValueError
        with pytest.raises(ValueError):
            SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.SCORE)

    def test_forward_with_x0_target(self):
        """Test forward method with X0 target type."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with X0 target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

        # Create test data
        batch_size = 4
        channels = 3
        height = width = 8

        x_t = torch.randn(batch_size, channels, height, width)
        f_x_t = torch.randn(batch_size, channels, height, width)
        x_0 = torch.randn(batch_size, channels, height, width)
        eps = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)

        # Compute loss
        loss = loss_fn(x_t, f_x_t, x_0, eps, t)

        # Check shape
        assert loss.shape == (batch_size,), (
            "Loss should return batch-wise scalar values"
        )

        # Check values - for X0 target, loss should be MSE between f_x_t and x_0
        expected_loss = torch.sum((f_x_t - x_0) ** 2, dim=(1, 2, 3))
        assert torch.allclose(loss, expected_loss)

    def test_forward_with_eps_target(self):
        """Test forward method with EPS target type."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with EPS target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.EPS)

        # Create test data
        batch_size = 4
        channels = 3
        height = width = 8

        x_t = torch.randn(batch_size, channels, height, width)
        f_x_t = torch.randn(batch_size, channels, height, width)
        x_0 = torch.randn(batch_size, channels, height, width)
        eps = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)

        # Compute loss
        loss = loss_fn(x_t, f_x_t, x_0, eps, t)

        # Check shape
        assert loss.shape == (batch_size,), (
            "Loss should return batch-wise scalar values"
        )

        # Check values - for EPS target, loss should be MSE between f_x_t and eps
        expected_loss = torch.sum((f_x_t - eps) ** 2, dim=(1, 2, 3))
        assert torch.allclose(loss, expected_loss)

    def test_forward_with_v_target(self):
        """Test forward method with V target type."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with V target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.V)

        # Create test data
        batch_size = 4
        channels = 3
        height = width = 8

        x_t = torch.randn(batch_size, channels, height, width)
        f_x_t = torch.randn(batch_size, channels, height, width)
        x_0 = torch.randn(batch_size, channels, height, width)
        eps = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)

        # Compute expected target
        expected_target = (
            pad_shape_back(diffusion_process.alpha_prime(t), x_0.shape) * x_0
            + pad_shape_back(diffusion_process.sigma_prime(t), x_0.shape) * eps
        )

        # Compute loss
        loss = loss_fn(x_t, f_x_t, x_0, eps, t)

        # Check shape
        assert loss.shape == (batch_size,), (
            "Loss should return batch-wise scalar values"
        )

        # Check values - for V target, loss should be MSE between f_x_t and expected_target
        expected_loss = torch.sum((f_x_t - expected_target) ** 2, dim=(1, 2, 3))
        assert torch.allclose(loss, expected_loss)

    def test_with_1d_data(self):
        """Test loss computation with 1D data."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with X0 target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

        # Create 1D test data
        batch_size = 4
        data_dim = 10

        x_t = torch.randn(batch_size, data_dim)
        f_x_t = torch.randn(batch_size, data_dim)
        x_0 = torch.randn(batch_size, data_dim)
        eps = torch.randn(batch_size, data_dim)
        t = torch.rand(batch_size)

        # Compute loss
        loss = loss_fn(x_t, f_x_t, x_0, eps, t)

        # Check shape
        assert loss.shape == (batch_size,), (
            "Loss should return batch-wise scalar values"
        )

        # Check values
        expected_loss = torch.sum((f_x_t - x_0) ** 2, dim=1)
        assert torch.allclose(loss, expected_loss)

    def test_with_3d_data(self):
        """Test loss computation with 3D data."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with X0 target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

        # Create 3D test data
        batch_size = 4
        depth = 5
        height = 6
        width = 7

        x_t = torch.randn(batch_size, depth, height, width)
        f_x_t = torch.randn(batch_size, depth, height, width)
        x_0 = torch.randn(batch_size, depth, height, width)
        eps = torch.randn(batch_size, depth, height, width)
        t = torch.rand(batch_size)

        # Compute loss
        loss = loss_fn(x_t, f_x_t, x_0, eps, t)

        # Check shape
        assert loss.shape == (batch_size,), (
            "Loss should return batch-wise scalar values"
        )

        # Check values
        expected_loss = torch.sum((f_x_t - x_0) ** 2, dim=(1, 2, 3))
        assert torch.allclose(loss, expected_loss)

    def test_with_custom_diffusion(self):
        """Test loss computation with a custom diffusion process."""
        # Create a custom diffusion process with non-trivial alpha_prime and sigma_prime
        alpha = lambda t: torch.cos(t * torch.pi / 2)
        sigma = lambda t: torch.sin(t * torch.pi / 2)

        # For this alpha and sigma, the derivatives are:
        alpha_prime = lambda t: -torch.pi / 2 * torch.sin(t * torch.pi / 2)
        sigma_prime = lambda t: torch.pi / 2 * torch.cos(t * torch.pi / 2)

        class CustomDiffusionProcess(DiffusionProcess):
            def __init__(self):
                super().__init__(alpha=alpha, sigma=sigma)
                # Override the automatically computed derivatives with our analytical ones
                self.alpha_prime = alpha_prime
                self.sigma_prime = sigma_prime

        diffusion_process = CustomDiffusionProcess()

        # Initialize loss with V target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.V)

        # Create test data
        batch_size = 4
        channels = 3
        height = width = 8

        x_t = torch.randn(batch_size, channels, height, width)
        f_x_t = torch.randn(batch_size, channels, height, width)
        x_0 = torch.randn(batch_size, channels, height, width)
        eps = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)

        # Compute expected target
        expected_target = (
            pad_shape_back(diffusion_process.alpha_prime(t), x_0.shape) * x_0
            + pad_shape_back(diffusion_process.sigma_prime(t), x_0.shape) * eps
        )

        # Compute loss
        loss = loss_fn(x_t, f_x_t, x_0, eps, t)

        # Check shape
        assert loss.shape == (batch_size,), (
            "Loss should return batch-wise scalar values"
        )

        # Check values
        expected_loss = torch.sum((f_x_t - expected_target) ** 2, dim=(1, 2, 3))
        assert torch.allclose(loss, expected_loss)
