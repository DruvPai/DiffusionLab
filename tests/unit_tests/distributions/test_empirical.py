import pytest
import jax
import jax.numpy as jnp
from jax import Array
from typing import Iterable, Tuple

from diffusionlab.distributions.empirical import EmpiricalDistribution
from diffusionlab.distributions.gmm.iso_hom_gmm import IsoHomGMM
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
    @pytest.mark.parametrize(
        "test_id, data, num_samples_to_request, expected_data_shape, expected_label_shape, expect_labels",
        [
            (
                "unlabeled_1d",
                [
                    (jnp.array([[1.0], [2.0], [3.0]]), None),
                    (jnp.array([[4.0], [5.0]]), None),
                ],
                3,
                (1,),
                None,
                False,
            ),
            (
                "labeled_1d",
                [
                    (jnp.array([[1.0], [2.0]]), jnp.array([0, 1])),
                    (jnp.array([[3.0], [4.0], [5.0]]), jnp.array([0, 1, 0])),
                ],
                4,
                (1,),
                (),
                True,
            ),
            (
                "unlabeled_2d",
                [
                    (jnp.array([[1.0, 1.0], [2.0, 2.0]]), None),
                    (jnp.array([[3.0, 3.0]]), None),
                ],
                2,
                (2,),
                None,
                False,
            ),
            (
                "unlabeled_high_dim",
                [
                    (jnp.ones((2, 2, 3, 1)), None),
                    (jnp.zeros((1, 2, 3, 1)), None),
                ],
                2,
                (2, 3, 1),
                None,
                False,
            ),
            (
                "labeled_high_dim",
                [
                    (jnp.ones((2, 2, 3, 1)), jnp.arange(2)),
                    (jnp.zeros((1, 2, 3, 1)), jnp.array([0])),
                ],
                2,
                (2, 3, 1),
                (),
                True,
            ),
        ],
    )
    def test_sample_shapes(
        self,
        test_id,
        data,
        num_samples_to_request,
        expected_data_shape,
        expected_label_shape,
        expect_labels,
    ):
        """Test sampling output shapes for various data dimensions and labels."""
        dist = EmpiricalDistribution(data)
        key = jax.random.key(hash(test_id))  # Use test_id for key uniqueness
        samples, labels = dist.sample(key, num_samples_to_request)

        assert samples.shape == (num_samples_to_request,) + expected_data_shape
        if expect_labels:
            assert labels is not None
            assert labels.shape == (num_samples_to_request,) + expected_label_shape
            assert jnp.issubdtype(labels.dtype, jnp.integer)
        else:
            assert labels is None

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

        # Test case 2: Inconsistent data dimensions
        malformatted_data_inconsistent_dims: Iterable[Tuple[Array, Array]] = [
            (jnp.array([[1.0], [2.0], [3.0]]), jnp.array([0, 1, 0])),  # Shape (3, 1)
            (
                jnp.array([[4.0, 4.0], [5.0, 5.0]]),
                jnp.array([0, 1]),
            ),  # Shape (2, 2) - Inconsistent dim
        ]
        total_malformatted_samples = 3 + 2  # Total number of samples
        dist = EmpiricalDistribution(malformatted_data_inconsistent_dims)
        key = jax.random.key(2)
        # This should fail when trying to concatenate arrays with different shapes[1:]
        # Sample *all* elements to force concatenation attempt
        with pytest.raises(ValueError):
            # Match message removed as the exact JAX error might vary
            dist.sample(key, total_malformatted_samples)

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

    def test_sample_high_dim(self):
        """Test sampling with high-dimensional data (e.g., images)."""
        # Define data inline - Ensure consistent shapes
        data_shape = (2, 3, 1)  # Define locally
        high_dim_data: Iterable[Tuple[Array, None]] = [
            (jnp.ones((2,) + data_shape), None),  # Shape (2, 2, 3, 1)
            (jnp.zeros((1,) + data_shape), None),  # Shape (1, 2, 3, 1)
        ]
        dist = EmpiricalDistribution(high_dim_data)
        key = jax.random.key(7)
        num_samples = 2  # Sample more than one to test batching
        samples, labels = dist.sample(key, num_samples)
        assert samples.shape == (num_samples,) + data_shape
        assert labels is None

    # --- Vector Field Tests ---
    @pytest.mark.parametrize(
        "test_id, data, x_t_shape, t_val",
        [
            (
                "1d",
                [(jnp.array([[0.0], [10.0]]), None)],
                (1,),
                0.5,  # Test one t value, shape is independent
            ),
            (
                "2d",
                [
                    (jnp.array([[1.0, 1.0], [2.0, 2.0]]), None),
                    (jnp.array([[3.0, 3.0]]), None),
                ],
                (2,),
                0.5,
            ),
            (
                "high_dim",
                [
                    (jnp.ones((2, 2, 3, 1)), None),
                    (jnp.zeros((1, 2, 3, 1)), None),
                ],
                (2, 3, 1),
                0.5,
            ),
        ],
    )
    def test_vector_field_shapes(self, test_id, data, x_t_shape, t_val):
        """Test the shapes of x0, score, eps, and v predictions."""
        dist = EmpiricalDistribution(data)
        process = FlowMatchingProcess()
        # Create a dummy x_t with the correct shape
        x_t = jnp.ones(x_t_shape)
        t = jnp.array(t_val)

        # Test shapes of all relevant fields
        x0_hat = dist.x0(x_t, t, process)
        score = dist.score(x_t, t, process)
        eps = dist.eps(x_t, t, process)
        v = dist.v(x_t, t, process)

        assert x0_hat.shape == x_t_shape, f"x0 shape mismatch in {test_id}"
        assert score.shape == x_t_shape, f"score shape mismatch in {test_id}"
        assert eps.shape == x_t_shape, f"eps shape mismatch in {test_id}"
        assert v.shape == x_t_shape, f"v shape mismatch in {test_id}"

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

    def test_vector_fields_high_dim_shape(self):
        """Test shapes of all vector fields with high-dimensional data."""
        # Define data and process inline - Ensure consistent shapes
        data_shape = (2, 3, 1)  # Define locally
        high_dim_data: Iterable[Tuple[Array, None]] = [
            (jnp.ones((2,) + data_shape), None),  # Shape (2, 2, 3, 1)
            (jnp.zeros((1,) + data_shape), None),  # Shape (1, 2, 3, 1)
        ]
        dist = EmpiricalDistribution(high_dim_data)
        process = FlowMatchingProcess()  # Use FlowMatchingProcess
        x_t = jnp.ones(data_shape)  # Shape (2, 3, 1)
        t = jnp.array(0.5)

        x0_hat = dist.x0(x_t, t, process)
        score = dist.score(x_t, t, process)
        eps = dist.eps(x_t, t, process)
        v = dist.v(x_t, t, process)

        assert x0_hat.shape == x_t.shape
        assert score.shape == x_t.shape
        assert eps.shape == x_t.shape
        assert v.shape == x_t.shape
        assert x0_hat.shape == data_shape  # Explicit check
        assert score.shape == data_shape  # Explicit check
        assert eps.shape == data_shape  # Explicit check
        assert v.shape == data_shape  # Explicit check

    # --- Comparison Tests with Zero-Variance GMM ---
    @pytest.mark.parametrize(
        "test_id, data, x_t, t_val",
        [
            (
                "1d_simple",
                [(jnp.array([[0.0], [10.0]]), None)],  # Data points 0 and 10
                jnp.array([1.0]),  # Test point
                0.5,
            ),
            (
                "1d_closer_t0",
                [(jnp.array([[0.0], [10.0]]), None)],
                jnp.array([0.1]),  # Closer to one of the means
                0.1,  # Smaller t
            ),
            (
                "2d_simple",
                [
                    (jnp.array([[1.0, 1.0], [2.0, 2.0]]), None),
                    (jnp.array([[3.0, 3.0]]), None),
                ],
                jnp.array([1.5, 1.5]),  # Test point
                0.8,  # Larger t
            ),
            (
                "2d_further",
                [
                    (jnp.array([[1.0, 1.0], [-1.0, -1.0]]), None),
                ],
                jnp.array([5.0, -5.0]),  # Further test point
                0.5,
            ),
        ],
    )
    def test_vector_field_comparison_with_gmm(self, test_id, data, x_t, t_val):
        """Compare vector fields with an equivalent zero-variance GMM."""
        # Consolidate data for GMM
        all_data_points = jnp.concatenate([d[0] for d in data], axis=0)
        priors = (
            jnp.ones(all_data_points.shape[0]) / all_data_points.shape[0]
        )  # Assume uniform priors for GMM

        # Initialize distributions
        emp_dist = EmpiricalDistribution(data)
        # Use a very small variance for numerical stability
        gmm_dist = IsoHomGMM(
            means=all_data_points, variance=jnp.array(1e-9), priors=priors
        )

        # Initialize process
        process = FlowMatchingProcess()
        t = jnp.array(t_val)

        # Calculate vector fields for EmpiricalDistribution
        emp_x0 = emp_dist.x0(x_t, t, process)
        emp_score = emp_dist.score(x_t, t, process)
        emp_eps = emp_dist.eps(x_t, t, process)
        emp_v = emp_dist.v(x_t, t, process)

        # Calculate vector fields for IsotropicHomoscedasticGMM
        gmm_x0 = gmm_dist.x0(x_t, t, process)
        gmm_score = gmm_dist.score(x_t, t, process)
        gmm_eps = gmm_dist.eps(x_t, t, process)
        gmm_v = gmm_dist.v(x_t, t, process)

        # Compare results (using a reasonable tolerance)
        atol = 1e-5
        assert jnp.allclose(emp_x0, gmm_x0, atol=atol), f"x0 mismatch in {test_id}"
        assert jnp.allclose(emp_score, gmm_score, atol=atol), (
            f"score mismatch in {test_id}"
        )
        assert jnp.allclose(emp_eps, gmm_eps, atol=atol), f"eps mismatch in {test_id}"
        assert jnp.allclose(emp_v, gmm_v, atol=atol), f"v mismatch in {test_id}"
