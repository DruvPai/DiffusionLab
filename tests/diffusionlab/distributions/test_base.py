import torch

from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.distributions.base import Distribution
from diffusionlab.samplers import Sampler


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
