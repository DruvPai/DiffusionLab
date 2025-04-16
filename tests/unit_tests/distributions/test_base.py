import pytest
import jax.numpy as jnp
from jax import random

from diffusionlab.distributions.base import Distribution
from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.vector_fields import VectorFieldType


class TestDistributionBase:
    """Tests for the Distribution base class."""

    def test_distribution_abstract_methods(self):
        """Test that the abstract methods raise NotImplementedError."""
        # Inlined setup
        base_distribution = Distribution(dist_params={}, dist_hparams={})
        key = random.PRNGKey(0)
        dummy_x = jnp.zeros((1,))
        dummy_t = jnp.array(0.5)
        dummy_process = VariancePreservingProcess()

        with pytest.raises(NotImplementedError):
            base_distribution.sample(key, 1)

        with pytest.raises(NotImplementedError):
            base_distribution.score(dummy_x, dummy_t, dummy_process)

        with pytest.raises(NotImplementedError):
            base_distribution.x0(dummy_x, dummy_t, dummy_process)

        with pytest.raises(NotImplementedError):
            base_distribution.eps(dummy_x, dummy_t, dummy_process)

        with pytest.raises(NotImplementedError):
            base_distribution.v(dummy_x, dummy_t, dummy_process)

    def test_get_vector_field_dispatch(self):
        """Test that get_vector_field returns the correct methods."""
        # Inlined setup
        base_distribution = Distribution(dist_params={}, dist_hparams={})
        # key, dummy_x, dummy_t, dummy_process not needed for this test

        assert (
            base_distribution.get_vector_field(VectorFieldType.X0)
            == base_distribution.x0
        )
        assert (
            base_distribution.get_vector_field(VectorFieldType.EPS)
            == base_distribution.eps
        )
        assert (
            base_distribution.get_vector_field(VectorFieldType.V) == base_distribution.v
        )
        assert (
            base_distribution.get_vector_field(VectorFieldType.SCORE)
            == base_distribution.score
        )

    def test_get_vector_field_invalid_type(self):
        """Test that get_vector_field raises ValueError for invalid types."""
        # Inlined setup
        base_distribution = Distribution(dist_params={}, dist_hparams={})
        # key, dummy_x, dummy_t, dummy_process not needed for this test

        with pytest.raises(ValueError):
            base_distribution.get_vector_field("INVALID_TYPE")  # type: ignore

        # Test with a potential non-enum value
        class MockEnum:
            pass

        mock_invalid_type = MockEnum()

        with pytest.raises(ValueError):
            base_distribution.get_vector_field(mock_invalid_type)  # type: ignore
