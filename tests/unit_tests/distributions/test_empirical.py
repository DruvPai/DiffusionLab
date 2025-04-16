import pytest
import jax
import jax.numpy as jnp
from jax import Array
from typing import Iterable, Tuple

from diffusionlab.distributions.empirical import EmpiricalDistribution
from diffusionlab.dynamics import FlowMatchingProcess


class TestEmpiricalDistribution:
    """Tests for the EmpiricalDistribution class."""

    def test_init_unlabeled(self):
        """Test initialization with unlabeled data."""
        # Define data inline
        unlabeled_data: Iterable[Tuple[Array, None]] = [
            (jnp.array([[1.0], [2.0], [3.0]]), None),
            (jnp.array([[4.0], [5.0]]), None),
        ]
        dist = EmpiricalDistribution(unlabeled_data)
        # Check if the data is stored correctly (checking object identity)
        assert dist.dist_hparams["labeled_data"] is unlabeled_data

    def test_init_labeled(self):
        """Test initialization with labeled data."""
        # Define data inline
        labeled_data: Iterable[Tuple[Array, Array]] = [
            (jnp.array([[1.0], [2.0]]), jnp.array([0, 1])),
            (jnp.array([[3.0], [4.0], [5.0]]), jnp.array([0, 1, 0])),
        ]
        dist = EmpiricalDistribution(labeled_data)
        assert dist.dist_hparams["labeled_data"] is labeled_data

    # --- Sampling Tests ---
    def test_sample_unlabeled_correct_number(self):
        """Test sampling the correct number of unlabeled samples."""
        # Define data inline
        unlabeled_data: Iterable[Tuple[Array, None]] = [
            (jnp.array([[1.0], [2.0], [3.0]]), None),  # Data dim (1,)
            (jnp.array([[4.0], [5.0]]), None),
        ]
        data_dim = (1,)  # Define locally
        dist = EmpiricalDistribution(unlabeled_data)
        key = jax.random.key(0)
        num_samples = 3
        samples, labels = dist.sample(key, num_samples)
        assert samples.shape == (num_samples,) + data_dim
        assert labels is None

    def test_sample_labeled_correct_number_and_type(self):
        """Test sampling the correct number and type of labeled samples."""
        # Define data inline
        labeled_data: Iterable[Tuple[Array, Array]] = [
            (
                jnp.array([[1.0], [2.0]]),
                jnp.array([0, 1]),
            ),  # Data dim (1,), Label dim ()
            (jnp.array([[3.0], [4.0], [5.0]]), jnp.array([0, 1, 0])),
        ]
        data_dim = (1,)  # Define locally
        label_dim = ()  # Define locally
        dist = EmpiricalDistribution(labeled_data)
        key = jax.random.key(1)
        num_samples = 4
        samples, labels = dist.sample(key, num_samples)
        assert samples.shape == (num_samples,) + data_dim
        assert labels is not None
        assert labels.shape == (num_samples,) + label_dim
        assert jnp.issubdtype(labels.dtype, jnp.integer)  # Check labels are integers

    def test_sample_request_more_than_available_raises_error(self):
        """Test that requesting more samples than available raises a ValueError."""
        # Define data inline
        unlabeled_data: Iterable[Tuple[Array, None]] = [
            (jnp.array([[1.0], [2.0], [3.0]]), None),
            (jnp.array([[4.0], [5.0]]), None),
        ]
        total_samples_available = 5  # Define locally
        dist = EmpiricalDistribution(unlabeled_data)
        key = jax.random.key(2)
        with pytest.raises(
            ValueError, match=f"only {total_samples_available} items are available"
        ):  # Use local variable
            dist.sample(key, total_samples_available + 1)

    def test_sample_malformatted_data_raises_error(self):
        """Test that malformatted data raises a ValueError."""
        # Define data inline
        malformatted_data: Iterable[Tuple[Array, Array]] = [
            (jnp.array([[1.0], [2.0], [3.0]]), jnp.array([0, 1])),
            (jnp.array([[4.0], [5.0]]), jnp.array([0, 1])),
        ]
        dist = EmpiricalDistribution(malformatted_data)
        key = jax.random.key(2)
        with pytest.raises(ValueError, match="has inconsistent shape"):
            dist.sample(key, 1)

        malformatted_data: Iterable[Tuple[Array, Array]] = [
            (jnp.array([[1.0], [2.0], [3.0]]), [0, 1, 0]),
            (jnp.array([[4.0], [5.0]]), [0, 1]),
        ]
        dist = EmpiricalDistribution(malformatted_data)
        key = jax.random.key(2)
        with pytest.raises(ValueError, match="has inconsistent shape"):
            dist.sample(key, 1)

    def test_sample_reproducibility(self):
        """Test that sampling is reproducible with the same PRNG key."""
        # Define data inline (must be identical for both instances)
        labeled_data_source: Iterable[Tuple[Array, Array]] = [
            (jnp.array([[1.0], [2.0]]), jnp.array([0, 1])),
            (jnp.array([[3.0], [4.0], [5.0]]), jnp.array([0, 1, 0])),
        ]
        key = jax.random.key(3)
        num_samples = 3
        # Need separate instances because the iterator is consumed
        # Use copies of the data source if it's mutable or an iterator
        dist1 = EmpiricalDistribution(list(labeled_data_source))  # Ensure list copy
        samples1, labels1 = dist1.sample(key, num_samples)
        dist2 = EmpiricalDistribution(list(labeled_data_source))  # Ensure list copy
        samples2, labels2 = dist2.sample(key, num_samples)

        assert jnp.array_equal(samples1, samples2)
        assert labels1 is not None and labels2 is not None
        assert jnp.array_equal(labels1, labels2)

    def test_sample_randomness(self):
        """Test that sampling produces different results with different PRNG keys."""
        # Define data inline
        unlabeled_data_source: Iterable[Tuple[Array, None]] = [
            (jnp.array([[1.0], [2.0], [3.0]]), None),
            (jnp.array([[4.0], [5.0]]), None),
        ]
        key1 = jax.random.key(4)
        key2 = jax.random.key(5)
        num_samples = 4
        dist1 = EmpiricalDistribution(list(unlabeled_data_source))  # Ensure list copy
        samples1, _ = dist1.sample(key1, num_samples)
        dist2 = EmpiricalDistribution(list(unlabeled_data_source))  # Ensure list copy
        samples2, _ = dist2.sample(key2, num_samples)
        # It's statistically unlikely but possible for them to be equal for small N/num_samples
        assert not jnp.array_equal(samples1, samples2)

    def test_sample_2d(self):
        """Test sampling with 2D data."""
        # Define data inline
        data_2d: Iterable[Tuple[Array, None]] = [
            (jnp.array([[1.0, 1.0], [2.0, 2.0]]), None),  # Data dim (2,)
            (jnp.array([[3.0, 3.0]]), None),
        ]
        data_dim = (2,)  # Define locally
        dist = EmpiricalDistribution(data_2d)
        key = jax.random.key(6)
        num_samples = 2
        samples, labels = dist.sample(key, num_samples)
        assert samples.shape == (num_samples,) + data_dim
        assert labels is None

    # --- Vector Field Tests ---

    @pytest.mark.parametrize("t_val", [0.1, 0.5, 0.9])
    def test_x0_shape(self, t_val):
        """Test the shape of the x0 prediction."""
        # Define data and process inline
        simple_data = [(jnp.array([[0.0], [10.0]]), None)]  # Two points: 0 and 10
        dist = EmpiricalDistribution(simple_data)
        process = FlowMatchingProcess()  # Use FlowMatchingProcess
        x_t = jnp.array([1.0])  # Example noisy point, shape (1,)
        t = jnp.array(t_val)
        x0_hat = dist.x0(x_t, t, process)
        assert x0_hat.shape == x_t.shape

    def test_x0_simple_case_t_near_0(self):
        """Test x0 prediction when t is close to 0."""
        # Define data and process inline
        simple_data = [(jnp.array([[0.0], [10.0]]), None)]  # Two points: 0 and 10
        dist = EmpiricalDistribution(simple_data)
        process = FlowMatchingProcess()  # Use FlowMatchingProcess

        # At small t, sigma is small, alpha is near 1.
        # If x_t = alpha_t * x_i + noise, denoiser should strongly favor x_i.
        # Note: FlowMatchingProcess T_min/T_max are not defined, use small value.
        t = jnp.array(1e-5)
        alpha_t = process.alpha(t)

        # Test case 1: x_t is near alpha_t * 0.0
        x_t_near_0 = alpha_t * jnp.array([0.0]) + jnp.array([1e-7])  # Small noise
        x0_hat_0 = dist.x0(x_t_near_0, t, process)
        assert jnp.allclose(x0_hat_0, jnp.array([0.0]), atol=1e-4)

        # Test case 2: x_t is near alpha_t * 10.0
        x_t_near_10 = alpha_t * jnp.array([10.0]) + jnp.array([1e-7])  # Small noise
        x0_hat_10 = dist.x0(x_t_near_10, t, process)
        assert jnp.allclose(x0_hat_10, jnp.array([10.0]), atol=1e-4)

    @pytest.mark.parametrize("t_val", [0.1, 0.5, 0.9])
    def test_score_shape(self, t_val):
        """Test the shape of the score function output."""
        # Define data and process inline
        simple_data = [(jnp.array([[0.0], [10.0]]), None)]
        dist = EmpiricalDistribution(simple_data)
        process = FlowMatchingProcess()  # Use FlowMatchingProcess
        x_t = jnp.array([1.0])
        t = jnp.array(t_val)
        score = dist.score(x_t, t, process)
        assert score.shape == x_t.shape

    @pytest.mark.parametrize("t_val", [0.1, 0.5, 0.9])
    def test_eps_shape(self, t_val):
        """Test the shape of the epsilon field output."""
        # Define data and process inline
        simple_data = [(jnp.array([[0.0], [10.0]]), None)]
        dist = EmpiricalDistribution(simple_data)
        process = FlowMatchingProcess()  # Use FlowMatchingProcess
        x_t = jnp.array([1.0])
        t = jnp.array(t_val)
        eps = dist.eps(x_t, t, process)
        assert eps.shape == x_t.shape

    @pytest.mark.parametrize("t_val", [0.1, 0.5, 0.9])
    def test_v_shape(self, t_val):
        """Test the shape of the velocity field output."""
        # Define data and process inline
        simple_data = [(jnp.array([[0.0], [10.0]]), None)]
        dist = EmpiricalDistribution(simple_data)
        process = FlowMatchingProcess()  # Use FlowMatchingProcess
        x_t = jnp.array([1.0])
        t = jnp.array(t_val)
        v = dist.v(x_t, t, process)
        assert v.shape == x_t.shape

    def test_x0_2d(self):
        """Test x0 prediction with 2D data."""
        # Define data and process inline
        data_2d: Iterable[Tuple[Array, None]] = [
            (jnp.array([[1.0, 1.0], [2.0, 2.0]]), None),
            (jnp.array([[3.0, 3.0]]), None),
        ]
        data_dim = (2,)  # Define locally
        dist = EmpiricalDistribution(data_2d)
        process = FlowMatchingProcess()  # Use FlowMatchingProcess
        x_t = jnp.array([1.5, 1.5])  # Shape (2,)
        t = jnp.array(0.5)
        x0_hat = dist.x0(x_t, t, process)
        assert x0_hat.shape == x_t.shape
        assert x0_hat.shape == data_dim  # Explicit check

    def test_vector_fields_2d_shape(self):
        """Test shapes of all vector fields with 2D data."""
        # Define data and process inline
        data_2d: Iterable[Tuple[Array, None]] = [
            (jnp.array([[1.0, 1.0], [2.0, 2.0]]), None),
            (jnp.array([[3.0, 3.0]]), None),
        ]
        dist = EmpiricalDistribution(data_2d)
        process = FlowMatchingProcess()  # Use FlowMatchingProcess
        x_t = jnp.array([1.5, 1.5])  # Shape (2,)
        t = jnp.array(0.5)

        x0_hat = dist.x0(x_t, t, process)
        score = dist.score(x_t, t, process)
        eps = dist.eps(x_t, t, process)
        v = dist.v(x_t, t, process)

        assert x0_hat.shape == x_t.shape
        assert score.shape == x_t.shape
        assert eps.shape == x_t.shape
        assert v.shape == x_t.shape
