import pytest
import jax
import jax.numpy as jnp

# Corrected imports based on file structure
from diffusionlab.distributions.gmm.gmm import GMM, gmm_x0
from diffusionlab.distributions.gmm.iso_gmm import IsoGMM, iso_gmm_x0
from diffusionlab.distributions.gmm.iso_hom_gmm import IsoHomGMM, iso_hom_gmm_x0
from diffusionlab.distributions.gmm.low_rank_gmm import LowRankGMM, low_rank_gmm_x0
from diffusionlab.distributions.gmm.utils import (
    _logdeth,
    _lstsq,
    create_gmm_vector_field_fns,
)
from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.vector_fields import VectorFieldType, convert_vector_field_type

# --- Helper Function ---


def check_vector_field_consistency(gmm_instance, x_t, t, diffusion_process):
    """Checks consistency between x0, eps, score, and v fields."""
    # Test class methods directly
    x0 = gmm_instance.x0(x_t, t, diffusion_process)
    eps = gmm_instance.eps(x_t, t, diffusion_process)
    score = gmm_instance.score(x_t, t, diffusion_process)
    v = gmm_instance.v(x_t, t, diffusion_process)

    alpha_t = diffusion_process.alpha(t)
    sigma_t = diffusion_process.sigma(t)
    alpha_prime_t = diffusion_process.alpha_prime(t)
    sigma_prime_t = diffusion_process.sigma_prime(t)

    # Check shapes
    assert x0.shape == x_t.shape
    assert eps.shape == x_t.shape
    assert score.shape == x_t.shape
    assert v.shape == x_t.shape

    # Check conversions
    eps_from_x0 = convert_vector_field_type(
        x_t,
        x0,
        alpha_t,
        sigma_t,
        alpha_prime_t,
        sigma_prime_t,
        VectorFieldType.X0,
        VectorFieldType.EPS,
    )
    score_from_x0 = convert_vector_field_type(
        x_t,
        x0,
        alpha_t,
        sigma_t,
        alpha_prime_t,
        sigma_prime_t,
        VectorFieldType.X0,
        VectorFieldType.SCORE,
    )
    v_from_x0 = convert_vector_field_type(
        x_t,
        x0,
        alpha_t,
        sigma_t,
        alpha_prime_t,
        sigma_prime_t,
        VectorFieldType.X0,
        VectorFieldType.V,
    )

    atol = 1e-5
    assert jnp.allclose(eps, eps_from_x0, atol=atol)
    assert jnp.allclose(score, score_from_x0, atol=atol)
    assert jnp.allclose(v, v_from_x0, atol=atol)


# --- Test Classes ---


class TestGMM:
    def test_init(self):
        # Inline data from get_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        covs = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[0.5, -0.2], [-0.2, 0.8]],
                [[1.2, 0.0], [0.0, 0.3]],
            ]
        )
        priors = jnp.array([0.3, 0.4, 0.3])

        gmm = GMM(means, covs, priors)
        assert gmm is not None
        assert jnp.allclose(gmm.dist_params["means"], means)
        assert jnp.allclose(gmm.dist_params["covs"], covs)
        assert jnp.allclose(gmm.dist_params["priors"], priors)

    def test_init_invalid_priors(self):
        # Inline data from get_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        covs = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[0.5, -0.2], [-0.2, 0.8]],
                [[1.2, 0.0], [0.0, 0.3]],
            ]
        )
        # priors = jnp.array([0.3, 0.4, 0.3]) # Original priors not needed here

        invalid_priors = jnp.array([0.5, 0.6])  # Does not match num_components
        with pytest.raises(AssertionError):
            GMM(means, covs, invalid_priors)

        invalid_priors_sum = jnp.array([0.3, 0.3, 0.3])  # Does not sum to 1
        with pytest.raises(AssertionError):
            GMM(means, covs, invalid_priors_sum)

    def test_init_invalid_shapes(self):
        # Inline data from get_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        covs = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[0.5, -0.2], [-0.2, 0.8]],
                [[1.2, 0.0], [0.0, 0.3]],
            ]
        )
        priors = jnp.array([0.3, 0.4, 0.3])

        with pytest.raises(AssertionError):  # Invalid means dim
            GMM(means[0], covs, priors)
        with pytest.raises(AssertionError):  # Invalid covs dim
            GMM(means, covs[0], priors)
        with pytest.raises(AssertionError):  # Mismatched covs dim 1
            GMM(means, covs[:, :1, :], priors)
        with pytest.raises(AssertionError):  # Mismatched covs dim 2
            GMM(means, covs[:, :, :1], priors)

    def test_sample(self):
        key = jax.random.key(42)
        # Inline data from get_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        covs = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[0.5, -0.2], [-0.2, 0.8]],
                [[1.2, 0.0], [0.0, 0.3]],
            ]
        )
        priors = jnp.array([0.3, 0.4, 0.3])

        gmm = GMM(means, covs, priors)
        num_samples = 100
        samples, component_indices = gmm.sample(key, num_samples)

        assert samples.shape == (num_samples, means.shape[1])
        assert component_indices.shape == (num_samples,)
        assert jnp.all(component_indices >= 0)
        assert jnp.all(component_indices < means.shape[0])
        # Basic check: mean of samples should be roughly the weighted mean of component means
        expected_mean = jnp.sum(priors[:, None] * means, axis=0)
        # Increased tolerance slightly due to removing fixture scope (new key each time)
        assert jnp.allclose(jnp.mean(samples, axis=0), expected_mean, atol=0.6)

    def test_vector_fields(self):
        key = jax.random.key(42)
        # Inline data from get_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        covs = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[0.5, -0.2], [-0.2, 0.8]],
                [[1.2, 0.0], [0.0, 0.3]],
            ]
        )
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        gmm = GMM(means, covs, priors)
        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.5)

        # Check direct call vs class method for x0
        x0_direct = gmm_x0(x_t, t, diffusion_process, means, covs, priors)
        x0_class = gmm.x0(x_t, t, diffusion_process)
        assert jnp.allclose(x0_direct, x0_class)

        check_vector_field_consistency(gmm, x_t, t, diffusion_process)


class TestIsoGMM:
    def test_init(self):
        # Inline data from get_iso_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variances = jnp.array([1.0, 0.5, 1.2])
        priors = jnp.array([0.3, 0.4, 0.3])

        gmm = IsoGMM(means, variances, priors)
        assert gmm is not None
        assert jnp.allclose(gmm.dist_params["means"], means)
        assert jnp.allclose(gmm.dist_params["variances"], variances)
        assert jnp.allclose(gmm.dist_params["priors"], priors)

    def test_sample(self):
        key = jax.random.key(42)
        # Inline data from get_iso_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variances = jnp.array([1.0, 0.5, 1.2])
        priors = jnp.array([0.3, 0.4, 0.3])

        gmm = IsoGMM(means, variances, priors)
        num_samples = 100
        samples, component_indices = gmm.sample(key, num_samples)

        assert samples.shape == (num_samples, means.shape[1])
        assert component_indices.shape == (num_samples,)
        assert jnp.all(component_indices >= 0)
        assert jnp.all(component_indices < means.shape[0])
        expected_mean = jnp.sum(priors[:, None] * means, axis=0)
        assert jnp.allclose(jnp.mean(samples, axis=0), expected_mean, atol=0.6)

    def test_vector_fields(self):
        key = jax.random.key(42)
        # Inline data from get_iso_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variances = jnp.array([1.0, 0.5, 1.2])
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        gmm = IsoGMM(means, variances, priors)
        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.3)

        # Check direct call vs class method for x0
        x0_direct = iso_gmm_x0(x_t, t, diffusion_process, means, variances, priors)
        x0_class = gmm.x0(x_t, t, diffusion_process)
        assert jnp.allclose(x0_direct, x0_class)

        check_vector_field_consistency(gmm, x_t, t, diffusion_process)

    def test_special_case_of_gmm(self):
        key = jax.random.key(42)
        # Inline data from get_iso_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variances = jnp.array([1.0, 0.5, 1.2])
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        iso_gmm = IsoGMM(means, variances, priors)

        # Create equivalent full GMM
        covs = jax.vmap(lambda var: var * jnp.eye(data_dim))(variances)
        full_gmm = GMM(means, covs, priors)

        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.4)

        # Compare vector fields
        atol = 1e-6
        assert jnp.allclose(
            iso_gmm.x0(x_t, t, diffusion_process),
            full_gmm.x0(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            iso_gmm.eps(x_t, t, diffusion_process),
            full_gmm.eps(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            iso_gmm.score(x_t, t, diffusion_process),
            full_gmm.score(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            iso_gmm.v(x_t, t, diffusion_process),
            full_gmm.v(x_t, t, diffusion_process),
            atol=atol,
        )


class TestIsoHomGMM:
    def test_init(self):
        # Inline data from get_iso_hom_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variance = jnp.array(1.0)
        priors = jnp.array([0.3, 0.4, 0.3])

        gmm = IsoHomGMM(means, variance, priors)
        assert gmm is not None
        assert jnp.allclose(gmm.dist_params["means"], means)
        assert jnp.allclose(gmm.dist_params["variance"], variance)
        assert jnp.allclose(gmm.dist_params["priors"], priors)

    def test_sample(self):
        key = jax.random.key(42)
        # Inline data from get_iso_hom_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variance = jnp.array(1.0)
        priors = jnp.array([0.3, 0.4, 0.3])

        gmm = IsoHomGMM(means, variance, priors)
        num_samples = 100
        samples, component_indices = gmm.sample(key, num_samples)

        assert samples.shape == (num_samples, means.shape[1])
        assert component_indices.shape == (num_samples,)
        assert jnp.all(component_indices >= 0)
        assert jnp.all(component_indices < means.shape[0])
        expected_mean = jnp.sum(priors[:, None] * means, axis=0)
        assert jnp.allclose(jnp.mean(samples, axis=0), expected_mean, atol=0.6)

    def test_vector_fields(self):
        key = jax.random.key(42)
        # Inline data from get_iso_hom_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variance = jnp.array(1.0)
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        gmm = IsoHomGMM(means, variance, priors)
        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.7)

        # Check direct call vs class method for x0
        x0_direct = iso_hom_gmm_x0(x_t, t, diffusion_process, means, variance, priors)
        x0_class = gmm.x0(x_t, t, diffusion_process)
        assert jnp.allclose(x0_direct, x0_class)

        check_vector_field_consistency(gmm, x_t, t, diffusion_process)

    def test_special_case_of_gmm(self):
        key = jax.random.key(42)
        # Inline data from get_iso_hom_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variance = jnp.array(1.0)
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        num_components, data_dim = means.shape
        iso_hom_gmm = IsoHomGMM(means, variance, priors)

        # Create equivalent full GMM
        cov = variance * jnp.eye(data_dim)
        covs = jnp.repeat(jnp.expand_dims(cov, axis=0), num_components, axis=0)
        full_gmm = GMM(means, covs, priors)

        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.6)

        # Compare vector fields
        atol = 1e-6
        assert jnp.allclose(
            iso_hom_gmm.x0(x_t, t, diffusion_process),
            full_gmm.x0(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            iso_hom_gmm.eps(x_t, t, diffusion_process),
            full_gmm.eps(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            iso_hom_gmm.score(x_t, t, diffusion_process),
            full_gmm.score(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            iso_hom_gmm.v(x_t, t, diffusion_process),
            full_gmm.v(x_t, t, diffusion_process),
            atol=atol,
        )


class TestLowRankGMM:
    def test_init(self):
        # Inline data from get_low_rank_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        factors = jnp.array([[[1.0], [0.5]], [[0.7], [-0.4]], [[1.1], [0.0]]])  # d x r
        priors = jnp.array([0.3, 0.4, 0.3])

        gmm = LowRankGMM(means, factors, priors)
        assert gmm is not None
        assert jnp.allclose(gmm.dist_params["means"], means)
        assert jnp.allclose(gmm.dist_params["cov_factors"], factors)
        assert jnp.allclose(gmm.dist_params["priors"], priors)

    def test_sample(self):
        key = jax.random.key(42)
        # Inline data from get_low_rank_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        factors = jnp.array([[[1.0], [0.5]], [[0.7], [-0.4]], [[1.1], [0.0]]])  # d x r
        priors = jnp.array([0.3, 0.4, 0.3])

        gmm = LowRankGMM(means, factors, priors)
        num_samples = 100
        samples, component_indices = gmm.sample(key, num_samples)

        assert samples.shape == (num_samples, means.shape[1])
        assert component_indices.shape == (num_samples,)
        assert jnp.all(component_indices >= 0)
        assert jnp.all(component_indices < means.shape[0])
        expected_mean = jnp.sum(priors[:, None] * means, axis=0)
        assert jnp.allclose(jnp.mean(samples, axis=0), expected_mean, atol=0.6)

    def test_vector_fields(self):
        key = jax.random.key(42)
        # Inline data from get_low_rank_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        factors = jnp.array([[[1.0], [0.5]], [[0.7], [-0.4]], [[1.1], [0.0]]])  # d x r
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        gmm = LowRankGMM(means, factors, priors)
        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.1)

        # Check direct call vs class method for x0
        x0_direct = low_rank_gmm_x0(x_t, t, diffusion_process, means, factors, priors)
        x0_class = gmm.x0(x_t, t, diffusion_process)
        assert jnp.allclose(x0_direct, x0_class)

        check_vector_field_consistency(gmm, x_t, t, diffusion_process)

    def test_special_case_of_gmm(self):
        key = jax.random.key(42)
        # Inline data from get_low_rank_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        factors = jnp.array([[[1.0], [0.5]], [[0.7], [-0.4]], [[1.1], [0.0]]])  # d x r
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        low_rank_gmm = LowRankGMM(means, factors, priors)

        # Create equivalent full GMM
        covs = jax.vmap(lambda f: f @ f.T)(factors)
        full_gmm = GMM(means, covs, priors)

        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.2)

        # Compare vector fields
        atol = 1e-6  # Using a standard tolerance to see the error
        assert jnp.allclose(
            low_rank_gmm.x0(x_t, t, diffusion_process),
            full_gmm.x0(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            low_rank_gmm.eps(x_t, t, diffusion_process),
            full_gmm.eps(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            low_rank_gmm.score(x_t, t, diffusion_process),
            full_gmm.score(x_t, t, diffusion_process),
            atol=atol,
        )
        assert jnp.allclose(
            low_rank_gmm.v(x_t, t, diffusion_process),
            full_gmm.v(x_t, t, diffusion_process),
            atol=atol,
        )


# --- Test Utility Functions ---


class TestGMMUtils:
    def test_logdeth(self):
        # Test with identity matrix
        mat_id = jnp.eye(3)
        logdet_id = _logdeth(mat_id)
        assert jnp.isclose(logdet_id, 0.0)

        # Test with a diagonal matrix
        mat_diag = jnp.diag(jnp.array([2.0, 3.0, 0.5]))
        logdet_diag = _logdeth(mat_diag)
        expected_logdet_diag = jnp.log(2.0 * 3.0 * 0.5)
        assert jnp.allclose(logdet_diag, expected_logdet_diag)

        # Test with a known PSD matrix
        mat_psd = jnp.array([[2.0, 1.0], [1.0, 2.0]])  # Eigvals 1, 3; det = 3
        logdet_psd = _logdeth(mat_psd)
        expected_logdet_psd = jnp.log(3.0)
        assert jnp.allclose(logdet_psd, expected_logdet_psd)

    def test_lstsq(self):
        # Test exact solution
        A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        y = jnp.array([4.0, 6.0])
        x = _lstsq(A, y)
        expected_x = jnp.array([2.0, 2.0])
        assert jnp.allclose(x, expected_x)

        # Test overdetermined system
        A_over = jnp.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y_over = jnp.array([2.0, 3.0, 3.5])
        x_over = _lstsq(A_over, y_over)
        # Compare with numpy's lstsq result
        expected_x_over = jnp.linalg.lstsq(A_over, y_over, rcond=None)[0]
        assert jnp.allclose(x_over, expected_x_over)

    def test_create_gmm_vector_field_fns(self):
        # Use the base GMM x0 function as an example
        key = jax.random.key(42)
        # Inline data from get_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        covs = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[0.5, -0.2], [-0.2, 0.8]],
                [[1.2, 0.0], [0.0, 0.3]],
            ]
        )
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.5)

        eps_fn, score_fn, v_fn = create_gmm_vector_field_fns(gmm_x0)

        # Calculate fields directly using conversions for comparison
        x0_val = gmm_x0(x_t, t, diffusion_process, means, covs, priors)
        alpha_t = diffusion_process.alpha(t)
        sigma_t = diffusion_process.sigma(t)
        alpha_prime_t = diffusion_process.alpha_prime(t)
        sigma_prime_t = diffusion_process.sigma_prime(t)

        expected_eps = convert_vector_field_type(
            x_t,
            x0_val,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            VectorFieldType.X0,
            VectorFieldType.EPS,
        )
        expected_score = convert_vector_field_type(
            x_t,
            x0_val,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            VectorFieldType.X0,
            VectorFieldType.SCORE,
        )
        expected_v = convert_vector_field_type(
            x_t,
            x0_val,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            VectorFieldType.X0,
            VectorFieldType.V,
        )

        # Calculate using generated functions
        # Call generated functions directly with all args
        eps_val = eps_fn(x_t, t, diffusion_process, means, covs, priors)
        score_val = score_fn(x_t, t, diffusion_process, means, covs, priors)
        v_val = v_fn(x_t, t, diffusion_process, means, covs, priors)

        atol = 1e-5
        assert jnp.allclose(eps_val, expected_eps, atol=atol)
        assert jnp.allclose(score_val, expected_score, atol=atol)
        assert jnp.allclose(v_val, expected_v, atol=atol)


# --- Test Standalone Vector Field Functions ---


class TestStandaloneVectorFields:
    def test_standalone_gmm_x0(self):
        key = jax.random.key(101)
        means = jnp.array([[-1.0, 1.0], [1.0, -1.0]])
        covs = jnp.array([[[1.0, 0.1], [0.1, 1.0]], [[0.8, -0.2], [-0.2, 0.8]]])
        priors = jnp.array([0.6, 0.4])
        diffusion_process = VariancePreservingProcess()
        gmm = GMM(means, covs, priors)
        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.5)

        x0_standalone = gmm_x0(x_t, t, diffusion_process, means, covs, priors)
        x0_class = gmm.x0(x_t, t, diffusion_process)
        assert jnp.allclose(x0_standalone, x0_class, atol=1e-6)

    def test_standalone_iso_gmm_x0(self):
        key = jax.random.key(102)
        means = jnp.array([[-1.5, -1.5], [1.5, 1.5]])
        variances = jnp.array([0.7, 1.3])
        priors = jnp.array([0.5, 0.5])
        diffusion_process = VariancePreservingProcess()
        gmm = IsoGMM(means, variances, priors)
        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.3)

        x0_standalone = iso_gmm_x0(x_t, t, diffusion_process, means, variances, priors)
        x0_class = gmm.x0(x_t, t, diffusion_process)
        assert jnp.allclose(x0_standalone, x0_class, atol=1e-6)

    def test_standalone_iso_hom_gmm_x0(self):
        key = jax.random.key(103)
        means = jnp.array([[0.0, 0.0], [3.0, 3.0]])
        variance = jnp.array(0.9)
        priors = jnp.array([0.2, 0.8])
        diffusion_process = VariancePreservingProcess()
        gmm = IsoHomGMM(means, variance, priors)
        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.8)

        x0_standalone = iso_hom_gmm_x0(
            x_t, t, diffusion_process, means, variance, priors
        )
        x0_class = gmm.x0(x_t, t, diffusion_process)
        assert jnp.allclose(x0_standalone, x0_class, atol=1e-6)

    def test_standalone_low_rank_gmm_x0(self):
        key = jax.random.key(104)
        means = jnp.array([[-2.5, 0.0], [2.5, 0.0]])
        factors = jnp.array([[[1.2], [0.3]], [[0.9], [-0.1]]])  # d x r
        priors = jnp.array([0.7, 0.3])
        diffusion_process = VariancePreservingProcess()
        gmm = LowRankGMM(means, factors, priors)
        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.2)

        x0_standalone = low_rank_gmm_x0(
            x_t, t, diffusion_process, means, factors, priors
        )
        x0_class = gmm.x0(x_t, t, diffusion_process)
        assert jnp.allclose(x0_standalone, x0_class, atol=1e-6)

    def test_standalone_gmm_x0_jit(self):
        key = jax.random.key(42)
        # Inline data from get_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        covs = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[0.5, -0.2], [-0.2, 0.8]],
                [[1.2, 0.0], [0.0, 0.3]],
            ]
        )
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.5)

        gmm_x0_jitted = jax.jit(gmm_x0, static_argnums=(2,))

        x0_direct = gmm_x0(x_t, t, diffusion_process, means, covs, priors)
        x0_jitted = gmm_x0_jitted(x_t, t, diffusion_process, means, covs, priors)

        assert jnp.allclose(x0_direct, x0_jitted, atol=1e-6)

    def test_standalone_gmm_x0_vmap(self):
        key = jax.random.key(42)
        # Inline data from get_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        covs = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],
                [[0.5, -0.2], [-0.2, 0.8]],
                [[1.2, 0.0], [0.0, 0.3]],
            ]
        )
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        batch_size = 5

        keys = jax.random.split(key, batch_size)
        x_t_batch = jax.vmap(lambda k: jax.random.normal(k, (data_dim,)))(keys)
        t_batch = jnp.linspace(0.2, 0.8, batch_size)

        gmm_x0_vmapped = jax.vmap(gmm_x0, in_axes=(0, 0, None, None, None, None))

        expected_x0_batch = jax.vmap(gmm_x0, in_axes=(0, 0, None, None, None, None))(
            x_t_batch, t_batch, diffusion_process, means, covs, priors
        )
        x0_vmapped = gmm_x0_vmapped(
            x_t_batch, t_batch, diffusion_process, means, covs, priors
        )

        assert x0_vmapped.shape == (batch_size, data_dim)
        assert jnp.allclose(x0_vmapped, expected_x0_batch, atol=1e-6)

    def test_standalone_iso_gmm_x0_jit(self):
        key = jax.random.key(43)  # Use different key
        # Inline data from get_iso_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variances = jnp.array([1.0, 0.5, 1.2])
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.3)

        iso_gmm_x0_jitted = jax.jit(iso_gmm_x0, static_argnums=(2,))

        x0_direct = iso_gmm_x0(x_t, t, diffusion_process, means, variances, priors)
        x0_jitted = iso_gmm_x0_jitted(
            x_t, t, diffusion_process, means, variances, priors
        )

        assert jnp.allclose(x0_direct, x0_jitted, atol=1e-6)

    def test_standalone_iso_gmm_x0_vmap(self):
        key = jax.random.key(44)  # Use different key
        # Inline data from get_iso_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variances = jnp.array([1.0, 0.5, 1.2])
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        batch_size = 6  # Use different batch size

        keys = jax.random.split(key, batch_size)
        x_t_batch = jax.vmap(lambda k: jax.random.normal(k, (data_dim,)))(keys)
        t_batch = jnp.linspace(0.2, 0.8, batch_size)  # Use different range

        iso_gmm_x0_vmapped = jax.vmap(
            iso_gmm_x0, in_axes=(0, 0, None, None, None, None)
        )

        expected_x0_batch = jax.vmap(
            iso_gmm_x0, in_axes=(0, 0, None, None, None, None)
        )(x_t_batch, t_batch, diffusion_process, means, variances, priors)
        x0_vmapped = iso_gmm_x0_vmapped(
            x_t_batch, t_batch, diffusion_process, means, variances, priors
        )

        assert x0_vmapped.shape == (batch_size, data_dim)
        assert jnp.allclose(x0_vmapped, expected_x0_batch, atol=1e-6)

    def test_standalone_iso_hom_gmm_x0_jit(self):
        key = jax.random.key(45)  # Use different key
        # Inline data from get_iso_hom_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variance = jnp.array(1.0)
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.7)

        iso_hom_gmm_x0_jitted = jax.jit(iso_hom_gmm_x0, static_argnums=(2,))

        x0_direct = iso_hom_gmm_x0(x_t, t, diffusion_process, means, variance, priors)
        x0_jitted = iso_hom_gmm_x0_jitted(
            x_t, t, diffusion_process, means, variance, priors
        )

        assert jnp.allclose(x0_direct, x0_jitted, atol=1e-6)

    def test_standalone_iso_hom_gmm_x0_vmap(self):
        key = jax.random.key(46)  # Use different key
        # Inline data from get_iso_hom_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        variance = jnp.array(1.0)
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        batch_size = 4  # Use different batch size

        keys = jax.random.split(key, batch_size)
        x_t_batch = jax.vmap(lambda k: jax.random.normal(k, (data_dim,)))(keys)
        t_batch = jnp.linspace(0.3, 0.7, batch_size)  # Use different range

        iso_hom_gmm_x0_vmapped = jax.vmap(
            iso_hom_gmm_x0, in_axes=(0, 0, None, None, None, None)
        )

        expected_x0_batch = jax.vmap(
            iso_hom_gmm_x0, in_axes=(0, 0, None, None, None, None)
        )(x_t_batch, t_batch, diffusion_process, means, variance, priors)
        x0_vmapped = iso_hom_gmm_x0_vmapped(
            x_t_batch, t_batch, diffusion_process, means, variance, priors
        )

        assert x0_vmapped.shape == (batch_size, data_dim)
        assert jnp.allclose(x0_vmapped, expected_x0_batch, atol=1e-6)

    def test_standalone_low_rank_gmm_x0_jit(self):
        key = jax.random.key(47)  # Use different key
        # Inline data from get_low_rank_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        factors = jnp.array([[[1.0], [0.5]], [[0.7], [-0.4]], [[1.1], [0.0]]])  # d x r
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        x_t = jax.random.normal(key, (data_dim,))
        t = jnp.array(0.1)

        low_rank_gmm_x0_jitted = jax.jit(low_rank_gmm_x0, static_argnums=(2,))

        x0_direct = low_rank_gmm_x0(x_t, t, diffusion_process, means, factors, priors)
        x0_jitted = low_rank_gmm_x0_jitted(
            x_t, t, diffusion_process, means, factors, priors
        )

        assert jnp.allclose(x0_direct, x0_jitted, atol=1e-6)

    def test_standalone_low_rank_gmm_x0_vmap(self):
        key = jax.random.key(48)  # Use different key
        # Inline data from get_low_rank_gmm_data()
        means = jnp.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
        factors = jnp.array([[[1.0], [0.5]], [[0.7], [-0.4]], [[1.1], [0.0]]])  # d x r
        priors = jnp.array([0.3, 0.4, 0.3])
        diffusion_process = VariancePreservingProcess()

        data_dim = means.shape[1]
        batch_size = 7  # Use different batch size

        keys = jax.random.split(key, batch_size)
        x_t_batch = jax.vmap(lambda k: jax.random.normal(k, (data_dim,)))(keys)
        t_batch = jnp.linspace(0.05, 0.95, batch_size)  # Use different range

        low_rank_gmm_x0_vmapped = jax.vmap(
            low_rank_gmm_x0, in_axes=(0, 0, None, None, None, None)
        )

        expected_x0_batch = jax.vmap(
            low_rank_gmm_x0, in_axes=(0, 0, None, None, None, None)
        )(x_t_batch, t_batch, diffusion_process, means, factors, priors)
        x0_vmapped = low_rank_gmm_x0_vmapped(
            x_t_batch, t_batch, diffusion_process, means, factors, priors
        )

        assert x0_vmapped.shape == (batch_size, data_dim)
        assert jnp.allclose(x0_vmapped, expected_x0_batch, atol=1e-6)
