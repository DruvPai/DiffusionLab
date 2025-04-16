import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
import pytest

from diffusionlab.vector_fields import VectorFieldType, convert_vector_field_type


class TestVectorFieldConversion:
    """Tests for the vector field type conversion logic."""

    def _setup_data(self, key):
        """Helper to generate consistent test data."""
        x_key, score_key, eps_key = jax.random.split(key, 3)
        x = jax.random.normal(x_key, (3, 4))  # Shape: (3, 4)
        t = jnp.array(0.6)  # Example time step
        alpha = jnp.sqrt(1.0 - t**2)  # Example VP alpha
        sigma = t  # Example VP sigma
        alpha_prime = -t / jnp.sqrt(1.0 - t**2)  # Example VP alpha derivative
        sigma_prime = jnp.array(1.0)  # Example VP sigma derivative

        # Generate reference fields based on arbitrary "true" score and eps
        # These don't need to be exact, just consistent for testing conversions
        true_score = jax.random.normal(score_key, x.shape)  # Shape: (3, 4)
        jax.random.normal(eps_key, x.shape)  # Shape: (3, 4)

        # Calculate other reference fields based on equations and true_score/true_eps
        # Using equations from convert_vector_field_type docstring
        ref_score = true_score
        ref_x0 = (x + sigma**2 * ref_score) / alpha  # Eq (12)
        ref_eps = -sigma * ref_score  # Eq (14)

        alpha_ratio = alpha_prime / alpha
        sigma_ratio = sigma_prime / sigma
        ratio_diff = sigma_ratio - alpha_ratio
        ref_v = alpha_ratio * x - sigma**2 * ratio_diff * ref_score  # Eq (16)

        params = {
            "x": x,
            "alpha": alpha,
            "sigma": sigma,
            "alpha_prime": alpha_prime,
            "sigma_prime": sigma_prime,
        }
        references = {
            VectorFieldType.SCORE: ref_score,
            VectorFieldType.X0: ref_x0,
            VectorFieldType.EPS: ref_eps,
            VectorFieldType.V: ref_v,
        }
        return params, references

    def test_score_to_others(self):
        """Test conversions starting from SCORE."""
        key = jax.random.key(0)
        params, references = self._setup_data(key)
        f_x = references[VectorFieldType.SCORE]

        # SCORE -> X0
        converted_x0 = convert_vector_field_type(
            **params,
            f_x=f_x,
            in_type=VectorFieldType.SCORE,
            out_type=VectorFieldType.X0,
        )
        assert_allclose(converted_x0, references[VectorFieldType.X0], atol=1e-5)

        # SCORE -> EPS
        converted_eps = convert_vector_field_type(
            **params,
            f_x=f_x,
            in_type=VectorFieldType.SCORE,
            out_type=VectorFieldType.EPS,
        )
        assert_allclose(converted_eps, references[VectorFieldType.EPS], atol=1e-5)

        # SCORE -> V
        converted_v = convert_vector_field_type(
            **params, f_x=f_x, in_type=VectorFieldType.SCORE, out_type=VectorFieldType.V
        )
        assert_allclose(converted_v, references[VectorFieldType.V], atol=1e-5)

    def test_x0_to_others(self):
        """Test conversions starting from X0."""
        key = jax.random.key(1)
        params, references = self._setup_data(key)
        f_x = references[VectorFieldType.X0]

        # X0 -> SCORE
        converted_score = convert_vector_field_type(
            **params,
            f_x=f_x,
            in_type=VectorFieldType.X0,
            out_type=VectorFieldType.SCORE,
        )
        assert_allclose(converted_score, references[VectorFieldType.SCORE], atol=1e-5)

        # X0 -> EPS
        converted_eps = convert_vector_field_type(
            **params, f_x=f_x, in_type=VectorFieldType.X0, out_type=VectorFieldType.EPS
        )
        assert_allclose(converted_eps, references[VectorFieldType.EPS], atol=1e-5)

        # X0 -> V
        converted_v = convert_vector_field_type(
            **params, f_x=f_x, in_type=VectorFieldType.X0, out_type=VectorFieldType.V
        )
        assert_allclose(converted_v, references[VectorFieldType.V], atol=1e-5)

    def test_eps_to_others(self):
        """Test conversions starting from EPS."""
        key = jax.random.key(2)
        params, references = self._setup_data(key)
        f_x = references[VectorFieldType.EPS]

        # EPS -> SCORE
        converted_score = convert_vector_field_type(
            **params,
            f_x=f_x,
            in_type=VectorFieldType.EPS,
            out_type=VectorFieldType.SCORE,
        )
        assert_allclose(converted_score, references[VectorFieldType.SCORE], atol=1e-5)

        # EPS -> X0
        converted_x0 = convert_vector_field_type(
            **params, f_x=f_x, in_type=VectorFieldType.EPS, out_type=VectorFieldType.X0
        )
        assert_allclose(converted_x0, references[VectorFieldType.X0], atol=1e-5)

        # EPS -> V
        converted_v = convert_vector_field_type(
            **params, f_x=f_x, in_type=VectorFieldType.EPS, out_type=VectorFieldType.V
        )
        assert_allclose(converted_v, references[VectorFieldType.V], atol=1e-5)

    def test_v_to_others(self):
        """Test conversions starting from V."""
        key = jax.random.key(3)
        params, references = self._setup_data(key)
        f_x = references[VectorFieldType.V]

        # V -> SCORE
        converted_score = convert_vector_field_type(
            **params, f_x=f_x, in_type=VectorFieldType.V, out_type=VectorFieldType.SCORE
        )
        assert_allclose(converted_score, references[VectorFieldType.SCORE], atol=1e-5)

        # V -> X0
        converted_x0 = convert_vector_field_type(
            **params, f_x=f_x, in_type=VectorFieldType.V, out_type=VectorFieldType.X0
        )
        assert_allclose(converted_x0, references[VectorFieldType.X0], atol=1e-5)

        # V -> EPS
        converted_eps = convert_vector_field_type(
            **params, f_x=f_x, in_type=VectorFieldType.V, out_type=VectorFieldType.EPS
        )
        assert_allclose(converted_eps, references[VectorFieldType.EPS], atol=1e-5)

    def test_identity_conversions(self):
        """Test converting a type to itself."""
        key = jax.random.key(4)
        params, references = self._setup_data(key)

        for field_type in VectorFieldType:
            f_x = references[field_type]
            converted_fx = convert_vector_field_type(
                **params, f_x=f_x, in_type=field_type, out_type=field_type
            )
            assert_allclose(
                converted_fx,
                f_x,
                atol=1e-6,
                rtol=1e-6,
                err_msg=f"Identity conversion failed for {field_type}",
            )

    def test_numerical_stability(self):
        """Test edge cases for numerical stability."""
        key = jax.random.key(5)
        x_key, score_key = jax.random.split(key)
        # Use default float32
        x = jax.random.normal(x_key, (3, 4))
        ref_score = jax.random.normal(score_key, x.shape)

        # Case 1: sigma is very small (close to t=0 for VP)
        # Use 1e-3 to avoid cancellation issues
        t_near_zero = jnp.array(1e-3)
        alpha_nz = jnp.sqrt(1.0 - t_near_zero**2)
        sigma_nz = t_near_zero
        alpha_prime_nz = -t_near_zero / jnp.sqrt(1.0 - t_near_zero**2)
        sigma_prime_nz = jnp.array(1.0)
        params_nz = {
            "x": x,
            "alpha": alpha_nz,
            "sigma": sigma_nz,
            "alpha_prime": alpha_prime_nz,
            "sigma_prime": sigma_prime_nz,
        }
        # Calculate reference with float32
        ref_x0_nz = (x + sigma_nz**2 * ref_score) / alpha_nz

        # Check X0 -> SCORE (involves division by sigma^2)
        converted_score_nz = convert_vector_field_type(
            **params_nz,
            f_x=ref_x0_nz,
            in_type=VectorFieldType.X0,
            out_type=VectorFieldType.SCORE,
        )
        # Further loosen tolerance for float32 precision limit -- very unstable conversion
        assert_allclose(converted_score_nz, ref_score, atol=1e-1, rtol=1e-1)

        # Case 2: ratio_diff is very small (e.g., alpha'/alpha approx sigma'/sigma)
        alpha_rd = jnp.array(0.5)
        sigma_rd = jnp.array(0.8)
        # Make primes such that ratio_diff is small
        alpha_prime_rd = jnp.array(-0.1)
        sigma_prime_rd = jnp.array(0.1600000001)
        params_rd = {
            "x": x,
            "alpha": alpha_rd,
            "sigma": sigma_rd,
            "alpha_prime": alpha_prime_rd,
            "sigma_prime": sigma_prime_rd,
        }
        # Calculate reference V with float32
        alpha_ratio_rd = alpha_prime_rd / alpha_rd
        sigma_ratio_rd = sigma_prime_rd / sigma_rd
        ratio_diff_rd = sigma_ratio_rd - alpha_ratio_rd
        ref_v_rd = alpha_ratio_rd * x - sigma_rd**2 * ratio_diff_rd * ref_score

        # Check V -> SCORE (involves division by ratio_diff)
        converted_score_rd = convert_vector_field_type(
            **params_rd,
            f_x=ref_v_rd,
            in_type=VectorFieldType.V,
            out_type=VectorFieldType.SCORE,
        )
        # Adjust tolerance for float32 in V->SCORE with small ratio_diff
        assert_allclose(converted_score_rd, ref_score, atol=1e-6, rtol=1e-6)

        # Check V -> X0 (involves division by ratio_diff)
        ref_x0_rd = (x + sigma_rd**2 * ref_score) / alpha_rd  # Calculate reference X0
        converted_x0_rd = convert_vector_field_type(
            **params_rd,
            f_x=ref_v_rd,
            in_type=VectorFieldType.V,
            out_type=VectorFieldType.X0,
        )
        # Adjust tolerance for float32 in V->X0 with small ratio_diff
        assert_allclose(converted_x0_rd, ref_x0_rd, atol=1e-6, rtol=1e-6)

        # Check V -> EPS (involves division by ratio_diff)
        ref_eps_rd = -sigma_rd * ref_score  # Calculate reference EPS
        converted_eps_rd = convert_vector_field_type(
            **params_rd,
            f_x=ref_v_rd,
            in_type=VectorFieldType.V,
            out_type=VectorFieldType.EPS,
        )
        # Adjust tolerance for float32 in V->EPS with small ratio_diff
        assert_allclose(converted_eps_rd, ref_eps_rd, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("in_type", VectorFieldType)
    @pytest.mark.parametrize("out_type", VectorFieldType)
    def test_conversion_jit(self, in_type: VectorFieldType, out_type: VectorFieldType):
        """Test that convert_vector_field_type works under JIT."""
        key = jax.random.key(10)
        params, references = self._setup_data(key)
        f_x = references[in_type]

        # Calculate expected result without JIT
        expected_result = convert_vector_field_type(
            **params, f_x=f_x, in_type=in_type, out_type=out_type
        )

        # JIT the conversion function, marking types as static
        jitted_convert = jax.jit(convert_vector_field_type, static_argnums=(6, 7))
        actual_result = jitted_convert(
            **params, f_x=f_x, in_type=in_type, out_type=out_type
        )

        assert_allclose(actual_result, expected_result, atol=1e-6, rtol=1e-6)

    @pytest.mark.parametrize("in_type", VectorFieldType)
    @pytest.mark.parametrize("out_type", VectorFieldType)
    def test_conversion_vmap(self, in_type: VectorFieldType, out_type: VectorFieldType):
        """Test convert_vector_field_type with VMAP and JIT combinations."""
        batch_size = 5
        key = jax.random.key(11)
        data_keys = jax.random.split(key, batch_size)

        # Generate batched data and references
        batched_params = []
        batched_references = {vf_type: [] for vf_type in VectorFieldType}
        scalar_params = {}  # Initialize here
        for k in data_keys:
            p, r = self._setup_data(k)
            # Keep only non-array params for simplicity in batching
            # Note: We use the *same* alpha, sigma etc. for the whole batch
            if not scalar_params:  # Populate only once
                # Store scalar params from the first sample
                scalar_params = {k_p: v for k_p, v in p.items() if jnp.ndim(v) == 0}
            batched_params.append(p["x"])  # Only batch x
            for vf_type in VectorFieldType:
                batched_references[vf_type].append(r[vf_type])

        # Stack batched arrays
        batched_x = jnp.stack(batched_params, axis=0)
        batched_f_x = jnp.stack(batched_references[in_type], axis=0)

        # Define the function signature for vmap/jit
        convert_fn = convert_vector_field_type
        vmap_axes = (0, 0, None, None, None, None, None, None)
        static_argnums = (6, 7)

        # 1. Calculate expected output using vmap on non-jitted function
        vmap_convert_expected = jax.vmap(convert_fn, in_axes=vmap_axes)
        expected_batch_result = vmap_convert_expected(
            batched_x,
            batched_f_x,
            scalar_params["alpha"],
            scalar_params["sigma"],
            scalar_params["alpha_prime"],
            scalar_params["sigma_prime"],
            in_type,
            out_type,
        )
        assert expected_batch_result.shape[0] == batch_size

        # 2. Vmap only
        vmap_convert = jax.vmap(convert_fn, in_axes=vmap_axes)
        actual_batch_result_vmap = vmap_convert(
            batched_x,
            batched_f_x,
            scalar_params["alpha"],
            scalar_params["sigma"],
            scalar_params["alpha_prime"],
            scalar_params["sigma_prime"],
            in_type,
            out_type,
        )
        assert_allclose(
            actual_batch_result_vmap, expected_batch_result, atol=1e-6, rtol=1e-6
        )

        # 3. Jit(Vmap)
        # The static arguments are the types passed to the vmapped function, at indices 6 and 7
        jit_vmap_convert = jax.jit(vmap_convert, static_argnums=(6, 7))
        actual_batch_result_jit_vmap = jit_vmap_convert(
            batched_x,
            batched_f_x,
            scalar_params["alpha"],
            scalar_params["sigma"],
            scalar_params["alpha_prime"],
            scalar_params["sigma_prime"],
            in_type,
            out_type,
        )
        assert_allclose(
            actual_batch_result_jit_vmap, expected_batch_result, atol=1e-5, rtol=1e-5
        )

        # 4. Vmap(Jit)
        jit_convert_scalar = jax.jit(convert_fn, static_argnums=static_argnums)
        vmap_jit_convert = jax.vmap(jit_convert_scalar, in_axes=vmap_axes)
        actual_batch_result_vmap_jit = vmap_jit_convert(
            batched_x,
            batched_f_x,
            scalar_params["alpha"],
            scalar_params["sigma"],
            scalar_params["alpha_prime"],
            scalar_params["sigma_prime"],
            in_type,
            out_type,
        )
        assert_allclose(
            actual_batch_result_vmap_jit, expected_batch_result, atol=1e-5, rtol=1e-5
        )
