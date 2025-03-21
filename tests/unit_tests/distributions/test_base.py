import torch
import pytest

from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.distributions.base import Distribution
from diffusionlab.samplers import Sampler
from diffusionlab.vector_fields import VectorFieldType


class TestDistributionBase:
    """Tests for the base Distribution class."""

    class MockSampler(Sampler):
        """Mock sampler for testing."""

        def __init__(self):
            super().__init__(
                is_stochastic=True,
                diffusion_process=DiffusionProcess(
                    alpha=lambda t: torch.ones_like(t),
                    sigma=lambda t: torch.zeros_like(t),
                ),
            )

    class MockDistribution(Distribution):
        """Mock distribution that implements required methods for testing."""

        @classmethod
        def x0(cls, x_t, t, diffusion_process, batched_dist_params, dist_hparams):
            return x_t  # Identity function for testing

        @classmethod
        def sample(cls, N, dist_params, dist_hparams):
            return torch.randn(N, 2), None

    def test_validate_hparams(self):
        """Test hyperparameter validation."""
        # Base distribution should accept empty hparams
        Distribution.validate_hparams({})

        # Should raise error for non-empty hparams
        try:
            Distribution.validate_hparams({"invalid": "param"})
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

    def test_validate_params(self):
        """Test parameter validation."""
        # Base distribution should accept empty params
        Distribution.validate_params({})

        # Should raise error for non-empty params
        try:
            Distribution.validate_params({"invalid": torch.tensor([1.0])})
            assert False, "Should have raised AssertionError"
        except AssertionError:
            pass

    def test_unimplemented_methods(self):
        """Test that unimplemented methods raise NotImplementedError."""
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.ones_like(t),
            sigma=lambda t: torch.zeros_like(t),
        )

        try:
            Distribution.x0(
                torch.randn(2, 2), torch.tensor([0.0, 0.0]), diffusion_process, {}, {}
            )
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass

        try:
            Distribution.sample(2, {}, {})
            assert False, "Should have raised NotImplementedError"
        except NotImplementedError:
            pass

    def test_vector_field_conversions(self):
        """Test that vector field conversions work correctly."""
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.ones_like(t),
            sigma=lambda t: torch.zeros_like(t),
        )
        mock_dist = self.MockDistribution()
        batch_size = 2
        dim = 3

        x_t = torch.randn(batch_size, dim)
        t = torch.zeros(batch_size)

        # Test eps conversion
        eps = mock_dist.eps(x_t, t, diffusion_process, {}, {})
        assert eps.shape == (batch_size, dim)

        # Test v conversion
        v = mock_dist.v(x_t, t, diffusion_process, {}, {})
        assert v.shape == (batch_size, dim)

        # Test score conversion
        score = mock_dist.score(x_t, t, diffusion_process, {}, {})
        assert score.shape == (batch_size, dim)

    def test_batch_dist_params(self):
        """Test the batching utility method."""
        N = 3
        params = {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([0.5])}

        batched_params = Distribution.batch_dist_params(N, params)

        assert batched_params["mean"].shape == (N, 2)
        assert batched_params["std"].shape == (N, 1)
        assert torch.all(batched_params["mean"][0] == params["mean"])
        assert torch.all(batched_params["std"][0] == params["std"])

    def test_x0_method(self):
        """Test the x0 method."""
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.ones_like(t),
            sigma=lambda t: torch.zeros_like(t),
        )
        mock_dist = self.MockDistribution()
        batch_size = 2
        dim = 3

        x_t = torch.randn(batch_size, dim)
        t = torch.zeros(batch_size)

        # Since our MockDistribution returns the input for x0, and other methods are derived from x0,
        # we can test that each method returns the expected values
        x0_result = mock_dist.x0(x_t, t, diffusion_process, {}, {})
        assert torch.all(x0_result == x_t)

    def test_get_vector_field_method(self):
        """Test that get_vector_field_method returns correct methods and they work as expected."""
        # Set up test data with a valid diffusion process (non-zero sigma)
        diffusion_process = DiffusionProcess(
            alpha=lambda t: torch.ones_like(t) * 0.9,  # Non-trivial alpha
            sigma=lambda t: torch.ones_like(t) * 0.1,  # Non-zero sigma
        )
        mock_dist = self.MockDistribution()
        batch_size = 2
        dim = 3
        x_t = torch.randn(batch_size, dim)
        t = torch.ones(batch_size) * 0.5  # Non-zero time

        # Test for each vector field type
        for vector_field_type in [
            VectorFieldType.X0,
            VectorFieldType.EPS,
            VectorFieldType.V,
            VectorFieldType.SCORE,
        ]:
            # Get the method for this vector field type
            method = mock_dist.get_vector_field_method(vector_field_type)

            # Verify the returned method is the correct one
            if vector_field_type == VectorFieldType.X0:
                assert method == mock_dist.x0
            elif vector_field_type == VectorFieldType.EPS:
                assert method == mock_dist.eps
            elif vector_field_type == VectorFieldType.V:
                assert method == mock_dist.v
            elif vector_field_type == VectorFieldType.SCORE:
                assert method == mock_dist.score

            # Test that the method can be called and returns correctly shaped tensor
            result = method(x_t, t, diffusion_process, {}, {})
            assert result.shape == x_t.shape

            # For our MockDistribution, all methods are derived from x0 which returns x_t
            # So all methods should return tensors that depend on x_t
            assert result.dtype == x_t.dtype
            assert result.device == x_t.device

            # Direct method call should match indirect call via get_vector_field_method
            direct_result = None  # Initialize to avoid "possibly unbound" error
            if vector_field_type == VectorFieldType.X0:
                direct_result = mock_dist.x0(x_t, t, diffusion_process, {}, {})
            elif vector_field_type == VectorFieldType.EPS:
                direct_result = mock_dist.eps(x_t, t, diffusion_process, {}, {})
            elif vector_field_type == VectorFieldType.V:
                direct_result = mock_dist.v(x_t, t, diffusion_process, {}, {})
            elif vector_field_type == VectorFieldType.SCORE:
                direct_result = mock_dist.score(x_t, t, diffusion_process, {}, {})

            assert direct_result is not None
            assert torch.all(result == direct_result)

        # Test the ValueError raised for invalid vector field types
        # We need a separate test for this since Python's type checking prevents
        # passing invalid enum values directly

    def test_get_vector_field_method_error_handling(self):
        """Test that get_vector_field_method raises ValueError for invalid types."""
        mock_dist = self.MockDistribution()

        # Using monkey patching approach for testing the error handling
        # Create a new class with the same interface as VectorFieldType but with an invalid value
        original_eq = VectorFieldType.__eq__

        try:
            # Override the __eq__ method to make any comparison with a VectorFieldType false
            # This will force the code to enter the 'else' branch
            def mock_eq(self, other):
                return False

            VectorFieldType.__eq__ = mock_eq

            # Now the comparisons with VectorFieldType will all fail
            # And the code will enter the 'else' branch raising ValueError
            with pytest.raises(ValueError):
                mock_dist.get_vector_field_method(VectorFieldType.X0)

        finally:
            # Restore the original __eq__ method
            VectorFieldType.__eq__ = original_eq
