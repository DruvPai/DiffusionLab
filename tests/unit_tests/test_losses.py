import torch
import pytest
from diffusionlab.losses import SamplewiseDiffusionLoss
from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType, VectorField
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

    # New tests for batchwise_loss_factory
    def test_batchwise_loss_factory_creation(self):
        """Test that batchwise_loss_factory returns a callable function."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with X0 target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

        # Create batchwise loss function with different noise draws
        batchwise_loss_1 = loss_fn.batchwise_loss_factory(N_noise_draws_per_sample=1)
        batchwise_loss_3 = loss_fn.batchwise_loss_factory(N_noise_draws_per_sample=3)

        # Verify that returned objects are callable
        assert callable(batchwise_loss_1)
        assert callable(batchwise_loss_3)

    def test_batchwise_loss_with_simple_vector_field(self):
        """Test batchwise_loss_factory with a simple vector field."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with X0 target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

        # Create a simple vector field that returns the input unchanged
        class IdentityVectorField(VectorField):
            def __init__(self):
                # Define the function to pass to the parent class
                def identity_fn(x, t):
                    return x

                super().__init__(identity_fn, VectorFieldType.X0)

        # Create test data
        batch_size = 4
        channels = 3
        height = width = 8

        # Create clean input data, timesteps, and sample weights
        x_0 = torch.randn(batch_size, channels, height, width)
        t = torch.rand(batch_size)
        sample_weights = torch.ones(batch_size)

        # Create vector field
        vector_field = IdentityVectorField()

        # Create batchwise loss function with 1 noise draw per sample
        batchwise_loss = loss_fn.batchwise_loss_factory(N_noise_draws_per_sample=1)

        # Compute loss
        loss = batchwise_loss(vector_field, x_0, t, sample_weights)

        # Verify that the result is a scalar tensor
        assert loss.dim() == 0, "Loss should be a scalar tensor"

    def test_batchwise_loss_multiple_noise_draws(self):
        """Test batchwise_loss_factory with multiple noise draws per sample."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with EPS target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.EPS)

        # Create a vector field that always returns zeros
        class ZeroVectorField(VectorField):
            def __init__(self):
                # Define the function to pass to the parent class
                def zero_fn(x, t):
                    return torch.zeros_like(x)

                super().__init__(zero_fn, VectorFieldType.EPS)

        # Create test data
        batch_size = 4
        data_dim = 10

        # Create clean input data, timesteps, and sample weights
        x_0 = torch.randn(batch_size, data_dim)
        t = torch.rand(batch_size)
        sample_weights = torch.ones(batch_size)

        # Create vector field
        vector_field = ZeroVectorField()

        # Verify that increasing noise draws doesn't change the result format
        for noise_draws in [1, 3, 5]:
            batchwise_loss = loss_fn.batchwise_loss_factory(
                N_noise_draws_per_sample=noise_draws
            )
            loss = batchwise_loss(vector_field, x_0, t, sample_weights)

            # Verify that the result is a scalar tensor
            assert loss.dim() == 0, (
                f"Loss should be a scalar tensor with {noise_draws} noise draws"
            )

    def test_batchwise_loss_with_sample_weights(self):
        """Test batchwise_loss_factory with varying sample weights."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with EPS target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.EPS)

        # Create a test vector field that returns known values
        class ConstantVectorField(VectorField):
            def __init__(self):
                # Define the function to pass to the parent class
                def const_fn(x, t):
                    return torch.ones_like(x)

                super().__init__(const_fn, VectorFieldType.EPS)

        # Create test data
        batch_size = 4
        data_dim = 10

        # Create clean input data and timesteps
        x_0 = torch.randn(batch_size, data_dim)
        t = torch.rand(batch_size)

        # Create vector field
        vector_field = ConstantVectorField()

        # Create batchwise loss function with 1 noise draw per sample
        batchwise_loss = loss_fn.batchwise_loss_factory(N_noise_draws_per_sample=1)

        # Test with different sample weights
        uniform_weights = torch.ones(batch_size)
        loss_uniform = batchwise_loss(vector_field, x_0, t, uniform_weights)

        # Use weights that emphasize the first sample
        biased_weights = torch.tensor([3.0, 1.0, 1.0, 1.0])
        loss_biased = batchwise_loss(vector_field, x_0, t, biased_weights)

        # The losses should be different when using different weights
        assert loss_uniform.item() != loss_biased.item(), (
            "Different sample weights should produce different losses"
        )

    def test_batchwise_loss_vector_field_type_mismatch(self):
        """Test that batchwise_loss raises an assertion error when vector field type doesn't match target type."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with X0 target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

        # Create a vector field with a different type
        class MismatchedVectorField(VectorField):
            def __init__(self):
                # Define the function to pass to the parent class
                def identity_fn(x, t):
                    return x

                super().__init__(identity_fn, VectorFieldType.EPS)

        # Create test data
        batch_size = 4
        data_dim = 10

        # Create clean input data, timesteps, and sample weights
        x_0 = torch.randn(batch_size, data_dim)
        t = torch.rand(batch_size)
        sample_weights = torch.ones(batch_size)

        # Create vector field
        vector_field = MismatchedVectorField()

        # Create batchwise loss function
        batchwise_loss = loss_fn.batchwise_loss_factory(N_noise_draws_per_sample=1)

        # The call should raise an assertion error due to type mismatch
        with pytest.raises(AssertionError):
            batchwise_loss(vector_field, x_0, t, sample_weights)

    def test_batchwise_loss_reference_implementation(self):
        """Test batchwise_loss against a reference implementation to validate correctness."""
        # Create a simple diffusion process
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize loss with X0 target type
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

        # Create a simple vector field that returns the input unchanged
        class IdentityVectorField(VectorField):
            def __init__(self):
                # Define the function to pass to the parent class
                def identity_fn(x, t):
                    return x

                super().__init__(identity_fn, VectorFieldType.X0)

        # Create test data
        batch_size = 4
        data_dim = 10

        # Create clean input data, timesteps, and sample weights
        x_0 = torch.randn(batch_size, data_dim)
        t = torch.rand(batch_size)
        sample_weights = torch.ones(batch_size)

        # Create vector field
        vector_field = IdentityVectorField()

        # Use a single noise draw for deterministic testing
        N_noise_draws = 1
        batchwise_loss = loss_fn.batchwise_loss_factory(
            N_noise_draws_per_sample=N_noise_draws
        )

        # Fix the random seed for reproducibility
        torch.manual_seed(42)
        loss = batchwise_loss(vector_field, x_0, t, sample_weights)

        # Reference implementation manually reproducing the steps
        torch.manual_seed(42)  # Reset seed to get same noise
        x_expanded = torch.repeat_interleave(x_0, N_noise_draws, dim=0)
        t_expanded = torch.repeat_interleave(t, N_noise_draws, dim=0)
        weights_expanded = torch.repeat_interleave(sample_weights, N_noise_draws, dim=0)

        eps = torch.randn_like(x_expanded)
        x_t = diffusion_process.forward(x_expanded, t_expanded, eps)
        f_x_t = vector_field(x_t, t_expanded)

        samplewise_loss = loss_fn(x_t, f_x_t, x_expanded, eps, t_expanded)
        expected_loss = torch.mean(samplewise_loss * weights_expanded)

        # The loss should match the reference implementation
        assert torch.allclose(loss, expected_loss), (
            "Loss should match the reference implementation"
        )
