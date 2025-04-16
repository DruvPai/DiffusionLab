import pytest
import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
from typing import Callable

from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.vector_fields import VectorFieldType, convert_vector_field_type
from diffusionlab.samplers import Sampler, EulerMaruyamaSampler, DDMSampler


# --- Test Classes (Self-contained) ---


class TestSampler:
    """Tests for the base Sampler class. Tests define dependencies internally."""

    def test_init(self):
        """Test sampler initialization and sample_step assignment."""
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x

        # Minimal mock sampler defined locally for this test scope
        class MockSampler(Sampler):
            def get_sample_step_function(
                self,
            ) -> Callable[[int, jax.Array, jax.Array, jax.Array], jax.Array]:
                return lambda idx, x, zs, ts: x + 1.0  # Simple step

        sampler = MockSampler(
            diffusion_process=simple_vp_process,
            vector_field=identity_vector_field,
            vector_field_type=VectorFieldType.V,  # Arbitrary for this test
            use_stochastic_sampler=False,
        )
        assert sampler.sample_step is not None
        # Indirectly test assignment via sample/sample_trajectory

    def test_sample(self):
        """Test the sample method executes the loop correctly."""
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x

        class MockSampler(Sampler):
            def get_sample_step_function(
                self,
            ) -> Callable[[int, jax.Array, jax.Array, jax.Array], jax.Array]:
                return lambda idx, x, zs, ts: x + 1.0

        num_steps = 5
        data_shape = (2, 3)
        key = jax.random.key(0)
        x_init = jax.random.normal(key, data_shape)  # Shape: (2, 3)
        zs = jax.random.normal(key, (num_steps,) + data_shape)  # Shape: (5, 2, 3)
        ts = jnp.linspace(1.0, 0.0, num_steps + 1)  # Shape: (6,)

        sampler = MockSampler(
            diffusion_process=simple_vp_process,
            vector_field=identity_vector_field,
            vector_field_type=VectorFieldType.V,
            use_stochastic_sampler=False,
        )

        final_x = sampler.sample(x_init, zs, ts)  # Shape: (2, 3)

        # Expect final_x = x_init + num_steps * 1.0
        expected_final_x = x_init + float(num_steps)
        assert final_x.shape == data_shape
        assert_allclose(final_x, expected_final_x, atol=1e-6)

    def test_sample_trajectory(self):
        """Test the sample_trajectory method executes the loop correctly."""
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x

        class MockSampler(Sampler):
            def get_sample_step_function(
                self,
            ) -> Callable[[int, jax.Array, jax.Array, jax.Array], jax.Array]:
                return lambda idx, x, zs, ts: x + 1.0

        num_steps = 3
        data_shape = (4,)
        key = jax.random.key(1)
        x_init = jax.random.normal(key, data_shape)  # Shape: (4,)
        zs = jax.random.normal(key, (num_steps,) + data_shape)  # Shape: (3, 4,)
        ts = jnp.linspace(1.0, 0.1, num_steps + 1)  # Shape: (4,)

        sampler = MockSampler(
            diffusion_process=simple_vp_process,
            vector_field=identity_vector_field,
            vector_field_type=VectorFieldType.V,
            use_stochastic_sampler=False,
        )

        trajectory = sampler.sample_trajectory(x_init, zs, ts)  # Shape: (4, 4,)

        expected_trajectory_shape = (num_steps + 1,) + data_shape
        assert trajectory.shape == expected_trajectory_shape

        # Check values: trajectory[i] = x_init + i * 1.0
        expected_trajectory = jnp.array(
            [x_init + float(i) for i in range(num_steps + 1)]
        )
        assert_allclose(trajectory, expected_trajectory, atol=1e-6)

    def test_get_sample_step_function_abstract(self):
        """Test that the base class raises NotImplementedError."""
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x

        # Cannot instantiate Sampler directly, so create a minimal subclass
        class IncompleteSampler(Sampler):
            pass  # Missing get_sample_step_function

        with pytest.raises(NotImplementedError):
            # Error should happen during __post_init__ call to get_sample_step_function
            IncompleteSampler(
                diffusion_process=simple_vp_process,
                vector_field=identity_vector_field,
                vector_field_type=VectorFieldType.V,
                use_stochastic_sampler=False,
            )


class TestEulerMaruyamaSampler:
    """Tests for the EulerMaruyamaSampler class. Dependencies defined internally."""

    # Define common test data as class attributes for convenience
    data_shape = (2, 2)
    num_steps = 4
    key = jax.random.key(42)
    x_init = jax.random.normal(key, data_shape)  # Shape: (2, 2)
    zs = jax.random.normal(key, (num_steps,) + data_shape)  # Shape: (4, 2, 2)
    # Use non-uniform timesteps to catch potential bugs
    ts = jnp.array([1.0, 0.7, 0.5, 0.2, 0.0])  # Shape: (5,)
    idx = 1  # Corresponds to t=0.7 -> t1=0.5

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("vf_type", VectorFieldType)
    def test_get_sample_step_function_selection(self, vf_type, use_stochastic):
        """Test that the correct internal step function is selected."""
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)

        sampler = EulerMaruyamaSampler(
            diffusion_process=simple_vp_process,
            vector_field=zero_vector_field,  # Field content doesn't matter here
            vector_field_type=vf_type,
            use_stochastic_sampler=use_stochastic,
        )
        expected_func_name = f"_sample_step_{vf_type.name.lower()}_{'stochastic' if use_stochastic else 'deterministic'}"
        expected_func = getattr(sampler, expected_func_name)
        assert sampler.sample_step == expected_func

    @pytest.mark.parametrize("use_stochastic", [False, True])
    def test_get_sample_step_function_selection_with_undefined_vf_type(
        self, use_stochastic
    ):
        """Test that the correct internal step function is selected."""
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)

        with pytest.raises(ValueError, match="Unsupported vector field type"):
            EulerMaruyamaSampler(
                diffusion_process=simple_vp_process,
                vector_field=zero_vector_field,  # Field content doesn't matter here
                vector_field_type=None,
                use_stochastic_sampler=use_stochastic,
            )

    def test_get_step_quantities(self):
        """Test the calculation of intermediate quantities."""
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, VectorFieldType.SCORE, False
        )

        idx = self.idx  # 1
        t = self.ts[idx]  # 0.7
        t1 = self.ts[idx + 1]  # 0.5
        dt = t1 - t  # -0.2
        dw_t = self.zs[idx] * jnp.sqrt(-dt)

        alpha_t = simple_vp_process.alpha(t)
        sigma_t = simple_vp_process.sigma(t)
        alpha_prime_t = simple_vp_process.alpha_prime(t)
        sigma_prime_t = simple_vp_process.sigma_prime(t)
        alpha_ratio_t = alpha_prime_t / alpha_t
        sigma_ratio_t = sigma_prime_t / sigma_t
        diff_ratio_t = sigma_ratio_t - alpha_ratio_t

        actual = sampler._get_step_quantities(self.idx, self.zs, self.ts)

        expected = (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        )

        for act, exp in zip(actual, expected):
            assert_allclose(act, exp, atol=1e-6, rtol=1e-6)

    # --- Individual Step Function Tests ---
    # We use simple vector fields (zero or identity) and a known process (VP)
    # defined locally within each test.

    def test_sample_step_score_deterministic(self):
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, VectorFieldType.SCORE, False
        )
        x = self.x_init
        (
            t,
            t1,
            dt,
            _,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = sampler._get_step_quantities(self.idx, self.zs, self.ts)
        score_x_t = zero_vector_field(x, t)  # 0
        drift_t = alpha_ratio_t * x - (sigma_t**2) * diff_ratio_t * score_x_t
        expected_x_next = x + drift_t * dt

        actual_x_next = sampler._sample_step_score_deterministic(
            self.idx, x, self.zs, self.ts
        )
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    def test_sample_step_score_stochastic(self):
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, VectorFieldType.SCORE, True
        )
        x = self.x_init
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = sampler._get_step_quantities(self.idx, self.zs, self.ts)
        score_x_t = zero_vector_field(x, t)  # 0
        drift_t = alpha_ratio_t * x - 2 * (sigma_t**2) * diff_ratio_t * score_x_t
        diffusion_t = jnp.sqrt(2 * diff_ratio_t) * sigma_t
        expected_x_next = x + drift_t * dt + diffusion_t * dw_t

        actual_x_next = sampler._sample_step_score_stochastic(
            self.idx, x, self.zs, self.ts
        )
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    def test_sample_step_x0_deterministic(self):
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x
        sampler = EulerMaruyamaSampler(
            simple_vp_process, identity_vector_field, VectorFieldType.X0, False
        )
        x = self.x_init
        (
            t,
            t1,
            dt,
            _,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = sampler._get_step_quantities(self.idx, self.zs, self.ts)
        x0_x_t = identity_vector_field(x, t)  # x
        drift_t = sigma_ratio_t * x - alpha_t * diff_ratio_t * x0_x_t
        expected_x_next = x + drift_t * dt

        actual_x_next = sampler._sample_step_x0_deterministic(
            self.idx, x, self.zs, self.ts
        )
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    def test_sample_step_x0_stochastic(self):
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x
        sampler = EulerMaruyamaSampler(
            simple_vp_process, identity_vector_field, VectorFieldType.X0, True
        )
        x = self.x_init
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = sampler._get_step_quantities(self.idx, self.zs, self.ts)
        x0_x_t = identity_vector_field(x, t)  # x
        drift_t = (
            alpha_ratio_t + 2 * diff_ratio_t
        ) * x - 2 * alpha_t * diff_ratio_t * x0_x_t
        diffusion_t = jnp.sqrt(2 * diff_ratio_t) * sigma_t
        expected_x_next = x + drift_t * dt + diffusion_t * dw_t

        actual_x_next = sampler._sample_step_x0_stochastic(
            self.idx, x, self.zs, self.ts
        )
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    def test_sample_step_eps_deterministic(self):
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, VectorFieldType.EPS, False
        )
        x = self.x_init
        (
            t,
            t1,
            dt,
            _,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = sampler._get_step_quantities(self.idx, self.zs, self.ts)
        eps_x_t = zero_vector_field(x, t)  # 0
        drift_t = alpha_ratio_t * x + sigma_t * diff_ratio_t * eps_x_t
        expected_x_next = x + drift_t * dt

        actual_x_next = sampler._sample_step_eps_deterministic(
            self.idx, x, self.zs, self.ts
        )
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    def test_sample_step_eps_stochastic(self):
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, VectorFieldType.EPS, True
        )
        x = self.x_init
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = sampler._get_step_quantities(self.idx, self.zs, self.ts)
        eps_x_t = zero_vector_field(x, t)  # 0
        drift_t = alpha_ratio_t * x + 2 * sigma_t * diff_ratio_t * eps_x_t
        diffusion_t = jnp.sqrt(2 * diff_ratio_t) * sigma_t
        expected_x_next = x + drift_t * dt + diffusion_t * dw_t

        actual_x_next = sampler._sample_step_eps_stochastic(
            self.idx, x, self.zs, self.ts
        )
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    def test_sample_step_v_deterministic(self):
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x
        sampler = EulerMaruyamaSampler(
            simple_vp_process, identity_vector_field, VectorFieldType.V, False
        )
        x = self.x_init
        (
            t,
            t1,
            dt,
            _,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = sampler._get_step_quantities(self.idx, self.zs, self.ts)
        v_x_t = identity_vector_field(x, t)  # x
        drift_t = v_x_t
        expected_x_next = x + drift_t * dt

        actual_x_next = sampler._sample_step_v_deterministic(
            self.idx, x, self.zs, self.ts
        )
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    def test_sample_step_v_stochastic(self):
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x
        sampler = EulerMaruyamaSampler(
            simple_vp_process, identity_vector_field, VectorFieldType.V, True
        )
        x = self.x_init
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = sampler._get_step_quantities(self.idx, self.zs, self.ts)
        v_x_t = identity_vector_field(x, t)  # x
        drift_t = -alpha_ratio_t * x + 2 * v_x_t
        diffusion_t = jnp.sqrt(2 * diff_ratio_t) * sigma_t
        expected_x_next = x + drift_t * dt + diffusion_t * dw_t

        actual_x_next = sampler._sample_step_v_stochastic(self.idx, x, self.zs, self.ts)
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize(
        "type1, type2",
        [
            (VectorFieldType.X0, VectorFieldType.SCORE),
            (VectorFieldType.X0, VectorFieldType.EPS),
            (VectorFieldType.X0, VectorFieldType.V),
            (VectorFieldType.SCORE, VectorFieldType.EPS),
            (VectorFieldType.SCORE, VectorFieldType.V),
            (VectorFieldType.EPS, VectorFieldType.V),
        ],
    )
    def test_step_consistency_across_vf_types(self, type1, type2, use_stochastic):
        """Test that EM step is consistent when converting between VF types."""
        simple_vp_process = VariancePreservingProcess()
        x = self.x_init
        t = self.ts[self.idx]

        # Use a non-trivial base prediction (e.g., predicting v = x*cos(t))
        def predict_type1(x_in, t_in):
            return x_in * jnp.cos(t_in)

        pred1 = predict_type1(x, t)

        # Get process quantities needed for conversion
        alpha_t = simple_vp_process.alpha(t)
        sigma_t = simple_vp_process.sigma(t)
        alpha_prime_t = simple_vp_process.alpha_prime(t)
        sigma_prime_t = simple_vp_process.sigma_prime(t)

        # Convert prediction from type1 to type2
        pred2 = convert_vector_field_type(
            x, pred1, alpha_t, sigma_t, alpha_prime_t, sigma_prime_t, type1, type2
        )

        # Sampler 1 uses the original prediction and type1
        sampler1 = EulerMaruyamaSampler(
            diffusion_process=simple_vp_process,
            vector_field=predict_type1,
            vector_field_type=type1,
            use_stochastic_sampler=use_stochastic,
        )
        # Sampler 2 uses the converted prediction and type2
        sampler2 = EulerMaruyamaSampler(
            diffusion_process=simple_vp_process,
            vector_field=lambda _x, _t: pred2,  # Use the calculated pred2
            vector_field_type=type2,
            use_stochastic_sampler=use_stochastic,
        )

        # Get the appropriate step function for each sampler
        step_func1 = sampler1.get_sample_step_function()
        step_func2 = sampler2.get_sample_step_function()

        # Calculate the next step using both samplers
        x_next1 = step_func1(self.idx, x, self.zs, self.ts)
        x_next2 = step_func2(self.idx, x, self.zs, self.ts)

        # Assert the results are close
        assert_allclose(x_next1, x_next2, atol=1e-5, rtol=1e-5)

    # --- JIT / VMAP Tests ---

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("vf_type", VectorFieldType)
    def test_sample_jit(self, vf_type, use_stochastic):
        """Test EulerMaruyamaSampler.sample under JIT compilation."""
        simple_vp_process = VariancePreservingProcess()
        # Use a field that predicts zeros
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, vf_type, use_stochastic
        )

        # Prepare inputs (use class attributes)
        x_init = self.x_init
        zs = self.zs
        ts = self.ts

        # Calculate expected output (non-jitted)
        expected_final_x = sampler.sample(x_init, zs, ts)

        # JIT the sample method
        jitted_sample = jax.jit(sampler.sample)
        actual_final_x = jitted_sample(x_init, zs, ts)

        assert_allclose(actual_final_x, expected_final_x, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("vf_type", VectorFieldType)
    def test_sample_trajectory_jit(self, vf_type, use_stochastic):
        """Test EulerMaruyamaSampler.sample_trajectory under JIT compilation."""
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, vf_type, use_stochastic
        )

        # Prepare inputs (use class attributes)
        x_init = self.x_init
        zs = self.zs
        ts = self.ts

        # Calculate expected output (non-jitted)
        expected_trajectory = sampler.sample_trajectory(x_init, zs, ts)

        # JIT the sample_trajectory method
        jitted_sample_trajectory = jax.jit(sampler.sample_trajectory)
        actual_trajectory = jitted_sample_trajectory(x_init, zs, ts)

        assert_allclose(actual_trajectory, expected_trajectory, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("vf_type", VectorFieldType)
    def test_sample_vmap(self, vf_type, use_stochastic):
        """Test EulerMaruyamaSampler.sample with VMAP and JIT combinations."""
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, vf_type, use_stochastic
        )

        # Prepare batched inputs
        batch_size = 3
        key = jax.random.key(43)
        key, x0_key, zs_key = jax.random.split(key, 3)
        batched_x_init = jax.random.normal(x0_key, (batch_size,) + self.data_shape)
        # Ensure zs has batch dim matching x_init
        batched_zs = jax.random.normal(
            zs_key, (batch_size, self.num_steps) + self.data_shape
        )
        ts = self.ts  # Timesteps are usually not batched

        # 1. Calculate expected output using vmap on non-jitted function
        vmap_sample_expected = jax.vmap(sampler.sample, in_axes=(0, 0, None))
        expected_batch_final_x = vmap_sample_expected(batched_x_init, batched_zs, ts)
        assert expected_batch_final_x.shape == (batch_size,) + self.data_shape

        # 2. Vmap only
        vmap_sample = jax.vmap(sampler.sample, in_axes=(0, 0, None))
        actual_batch_final_x_vmap = vmap_sample(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_final_x_vmap, expected_batch_final_x, atol=1e-6, rtol=1e-6
        )

        # 3. Jit(Vmap)
        jit_vmap_sample = jax.jit(vmap_sample)
        actual_batch_final_x_jit_vmap = jit_vmap_sample(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_final_x_jit_vmap, expected_batch_final_x, atol=1e-5, rtol=1e-5
        )

        # 4. Vmap(Jit)
        jit_sample = jax.jit(sampler.sample)
        vmap_jit_sample = jax.vmap(jit_sample, in_axes=(0, 0, None))
        actual_batch_final_x_vmap_jit = vmap_jit_sample(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_final_x_vmap_jit, expected_batch_final_x, atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("vf_type", VectorFieldType)
    def test_sample_trajectory_vmap(self, vf_type, use_stochastic):
        """Test EulerMaruyamaSampler.sample_trajectory with VMAP and JIT combinations."""
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = EulerMaruyamaSampler(
            simple_vp_process, zero_vector_field, vf_type, use_stochastic
        )

        # Prepare batched inputs
        batch_size = 3
        key = jax.random.key(44)
        key, x0_key, zs_key = jax.random.split(key, 3)
        batched_x_init = jax.random.normal(x0_key, (batch_size,) + self.data_shape)
        batched_zs = jax.random.normal(
            zs_key, (batch_size, self.num_steps) + self.data_shape
        )
        ts = self.ts
        expected_traj_shape = (
            batch_size,
            self.num_steps + 1,
        ) + self.data_shape

        # 1. Calculate expected output using vmap on non-jitted function
        vmap_traj_expected = jax.vmap(sampler.sample_trajectory, in_axes=(0, 0, None))
        expected_batch_traj = vmap_traj_expected(batched_x_init, batched_zs, ts)
        assert expected_batch_traj.shape == expected_traj_shape

        # 2. Vmap only
        vmap_traj = jax.vmap(sampler.sample_trajectory, in_axes=(0, 0, None))
        actual_batch_traj_vmap = vmap_traj(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_traj_vmap, expected_batch_traj, atol=1e-6, rtol=1e-6
        )

        # 3. Jit(Vmap)
        jit_vmap_traj = jax.jit(vmap_traj)
        actual_batch_traj_jit_vmap = jit_vmap_traj(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_traj_jit_vmap, expected_batch_traj, atol=1e-5, rtol=1e-5
        )

        # 4. Vmap(Jit)
        jit_traj = jax.jit(sampler.sample_trajectory)
        vmap_jit_traj = jax.vmap(jit_traj, in_axes=(0, 0, None))
        actual_batch_traj_vmap_jit = vmap_jit_traj(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_traj_vmap_jit, expected_batch_traj, atol=1e-5, rtol=1e-5
        )


class TestDDMSampler:
    """Tests for the DDMSampler class. Dependencies defined internally."""

    # Define common test data as class attributes
    data_shape = (3,)
    num_steps = 3
    key = jax.random.key(123)
    x_init = jax.random.normal(key, data_shape)  # Shape: (3,)
    zs = jax.random.normal(key, (num_steps,) + data_shape)  # Shape: (3, 3,)
    ts = jnp.array([0.99, 0.6, 0.3, 0.0])  # Shape: (4,), Avoid t=1.0 for VP process
    idx = 0  # Corresponds to t=0.99 -> t1=0.6

    @pytest.mark.parametrize("use_stochastic", [False, True])
    def test_get_sample_step_function_selection(self, use_stochastic):
        """Test that the correct internal step function is selected."""
        simple_vp_process = VariancePreservingProcess()
        zero_vector_field = lambda x, t: jnp.zeros_like(x)
        sampler = DDMSampler(
            diffusion_process=simple_vp_process,
            vector_field=zero_vector_field,
            vector_field_type=VectorFieldType.X0,  # Arbitrary for this test
            use_stochastic_sampler=use_stochastic,
        )
        expected_func_name = (
            f"_sample_step_{'stochastic' if use_stochastic else 'deterministic'}"
        )
        expected_func = getattr(sampler, expected_func_name)
        assert sampler.sample_step == expected_func

    @pytest.mark.parametrize("input_vf_type", VectorFieldType)
    def test_get_x0_prediction(self, input_vf_type):
        """Test the conversion of any vector field type to X0 prediction."""
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x
        x = self.x_init
        t = self.ts[self.idx]  # 0.99

        # Create a dummy vector field output for the specific type
        # For simplicity, let's assume the field predicts 'x' itself
        f_x_t = identity_vector_field(x, t)  # Pretend field output is 'x'

        sampler = DDMSampler(
            diffusion_process=simple_vp_process,
            vector_field=lambda _x, _t: f_x_t,  # Use the fixed output
            vector_field_type=input_vf_type,
            use_stochastic_sampler=False,  # Doesn't matter for this method
        )

        # Calculate expected x0 using the standalone conversion function
        alpha_t = simple_vp_process.alpha(t)
        sigma_t = simple_vp_process.sigma(t)
        alpha_prime_t = simple_vp_process.alpha_prime(t)
        sigma_prime_t = simple_vp_process.sigma_prime(t)
        expected_x0_x_t = convert_vector_field_type(
            x,
            f_x_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            input_vf_type,
            VectorFieldType.X0,
        )

        actual_x0_x_t = sampler._get_x0_prediction(x, t)
        assert_allclose(actual_x0_x_t, expected_x0_x_t, atol=1e-6)

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize(
        "input_vf_type",
        [
            VectorFieldType.SCORE,
            VectorFieldType.EPS,
            VectorFieldType.V,
            VectorFieldType.X0,
        ],
    )
    def test_step_consistency_across_input_vf_types(
        self, input_vf_type, use_stochastic
    ):
        """Tests that DDPM/DDIM step is consistent regardless of input VF type if underlying x0 is same."""
        simple_vp_process = VariancePreservingProcess()
        x = self.x_init
        t = self.ts[self.idx]

        # Define a ground truth x0 prediction (e.g., x0 = x * 0.5)
        def predict_x0_true(x_in, t_in):
            return x_in * 0.5

        x0_true = predict_x0_true(x, t)

        # Get process quantities for conversion
        alpha_t = simple_vp_process.alpha(t)
        sigma_t = simple_vp_process.sigma(t)
        alpha_prime_t = simple_vp_process.alpha_prime(t)
        sigma_prime_t = simple_vp_process.sigma_prime(t)

        # Calculate the equivalent prediction for the input_vf_type
        input_pred = convert_vector_field_type(
            x,
            x0_true,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            VectorFieldType.X0,
            input_vf_type,
        )

        # Sampler using the converted input prediction and its type
        sampler_input = DDMSampler(
            diffusion_process=simple_vp_process,
            vector_field=lambda _x, _t: input_pred,
            vector_field_type=input_vf_type,
            use_stochastic_sampler=use_stochastic,
        )

        # Reference sampler using the ground truth x0 prediction directly
        sampler_ref = DDMSampler(
            diffusion_process=simple_vp_process,
            vector_field=predict_x0_true,
            vector_field_type=VectorFieldType.X0,
            use_stochastic_sampler=use_stochastic,
        )

        # Get the appropriate step function (same for both due to internal conversion)
        step_func = sampler_input.get_sample_step_function()
        step_func_ref = sampler_ref.get_sample_step_function()
        assert step_func.__name__ == step_func_ref.__name__  # Sanity check

        # Calculate the next step using both
        x_next_input = step_func(self.idx, x, self.zs, self.ts)
        x_next_ref = step_func_ref(self.idx, x, self.zs, self.ts)

        # Assert results are close
        assert_allclose(x_next_input, x_next_ref, atol=1e-5, rtol=1e-5)

    def test_sample_step_deterministic(self):
        """Test the DDIM step."""
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x
        # Use identity field predicting X0 = x for simplicity
        sampler = DDMSampler(
            simple_vp_process, identity_vector_field, VectorFieldType.X0, False
        )
        x = self.x_init
        t = self.ts[self.idx]  # 0.99
        t1 = self.ts[self.idx + 1]  # 0.6

        # Calculate expected x0 (which is just x in this case)
        x0_x_t = sampler._get_x0_prediction(x, t)
        assert_allclose(x0_x_t, x, atol=1e-6)  # Verify assumption

        alpha_t = simple_vp_process.alpha(t)
        sigma_t = simple_vp_process.sigma(t)
        alpha_t1 = simple_vp_process.alpha(t1)
        sigma_t1 = simple_vp_process.sigma(t1)

        # Use the actual alpha_t, sigma_t for t=0.99
        r01 = sigma_t1 / sigma_t
        # Avoid division by zero if alpha_t1 is zero (e.g. if t1=1.0)
        r11 = (alpha_t / alpha_t1) * r01 if alpha_t1 != 0 else 0.0
        expected_x_next = r01 * x + alpha_t1 * (1 - r11) * x0_x_t

        actual_x_next = sampler._sample_step_deterministic(
            self.idx, x, self.zs, self.ts
        )
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    def test_sample_step_stochastic(self):
        """Test the DDPM step."""
        simple_vp_process = VariancePreservingProcess()
        identity_vector_field = lambda x, t: x
        # Use identity field predicting X0 = x for simplicity
        sampler = DDMSampler(
            simple_vp_process, identity_vector_field, VectorFieldType.X0, True
        )
        x = self.x_init
        t = self.ts[self.idx]  # 0.99
        t1 = self.ts[self.idx + 1]  # 0.6
        z_t = self.zs[self.idx]

        x0_x_t = sampler._get_x0_prediction(x, t)
        assert_allclose(x0_x_t, x, atol=1e-6)  # Verify assumption

        alpha_t = simple_vp_process.alpha(t)
        sigma_t = simple_vp_process.sigma(t)
        alpha_t1 = simple_vp_process.alpha(t1)
        sigma_t1 = simple_vp_process.sigma(t1)

        # Use the actual alpha_t, sigma_t for t=0.99
        # Avoid division by zero if alpha_t1 or sigma_t is zero
        r11 = (
            (alpha_t / alpha_t1) * (sigma_t1 / sigma_t)
            if alpha_t1 != 0 and sigma_t != 0
            else 0.0
        )
        r12 = r11 * (sigma_t1 / sigma_t) if sigma_t != 0 else 0.0
        r22 = (alpha_t / alpha_t1) * r12 if alpha_t1 != 0 else 0.0

        expected_mean = r12 * x + alpha_t1 * (1 - r22) * x0_x_t
        # Clamp value inside sqrt to avoid potential small negative numbers due to precision
        variance_term = jnp.maximum(1 - (r11**2), 0.0)
        expected_std = sigma_t1 * jnp.sqrt(variance_term)
        expected_x_next = expected_mean + expected_std * z_t

        actual_x_next = sampler._sample_step_stochastic(self.idx, x, self.zs, self.ts)
        assert_allclose(actual_x_next, expected_x_next, atol=1e-6)

    # --- JIT / VMAP Tests ---

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("input_vf_type", VectorFieldType)
    def test_sample_jit(self, input_vf_type, use_stochastic):
        """Test DDMSampler.sample under JIT compilation."""
        simple_vp_process = VariancePreservingProcess()

        # Use a field predicting x0 = x * 0.5
        def predict_x0(x, t):
            return x * 0.5

        # Convert this to the input_vf_type prediction dynamically
        def get_input_pred_field(x, t):
            x0_true = predict_x0(x, t)
            alpha_t = simple_vp_process.alpha(t)
            sigma_t = simple_vp_process.sigma(t)
            alpha_prime_t = simple_vp_process.alpha_prime(t)
            sigma_prime_t = simple_vp_process.sigma_prime(t)
            return convert_vector_field_type(
                x,
                x0_true,
                alpha_t,
                sigma_t,
                alpha_prime_t,
                sigma_prime_t,
                VectorFieldType.X0,
                input_vf_type,
            )

        sampler = DDMSampler(
            simple_vp_process, get_input_pred_field, input_vf_type, use_stochastic
        )

        # Prepare inputs (use class attributes)
        x_init = self.x_init
        zs = self.zs
        ts = self.ts

        # Calculate expected output (non-jitted)
        expected_final_x = sampler.sample(x_init, zs, ts)

        # JIT the sample method
        jitted_sample = jax.jit(sampler.sample)
        actual_final_x = jitted_sample(x_init, zs, ts)

        assert_allclose(actual_final_x, expected_final_x, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("input_vf_type", VectorFieldType)
    def test_sample_trajectory_jit(self, input_vf_type, use_stochastic):
        """Test DDMSampler.sample_trajectory under JIT compilation."""
        simple_vp_process = VariancePreservingProcess()

        def predict_x0(x, t):
            return x * 0.5

        def get_input_pred_field(x, t):
            x0_true = predict_x0(x, t)
            alpha_t = simple_vp_process.alpha(t)
            sigma_t = simple_vp_process.sigma(t)
            alpha_prime_t = simple_vp_process.alpha_prime(t)
            sigma_prime_t = simple_vp_process.sigma_prime(t)
            return convert_vector_field_type(
                x,
                x0_true,
                alpha_t,
                sigma_t,
                alpha_prime_t,
                sigma_prime_t,
                VectorFieldType.X0,
                input_vf_type,
            )

        sampler = DDMSampler(
            simple_vp_process, get_input_pred_field, input_vf_type, use_stochastic
        )

        x_init = self.x_init
        zs = self.zs
        ts = self.ts

        expected_trajectory = sampler.sample_trajectory(x_init, zs, ts)

        jitted_sample_trajectory = jax.jit(sampler.sample_trajectory)
        actual_trajectory = jitted_sample_trajectory(x_init, zs, ts)

        assert_allclose(actual_trajectory, expected_trajectory, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("input_vf_type", VectorFieldType)
    def test_sample_vmap(self, input_vf_type, use_stochastic):
        """Test DDMSampler.sample with VMAP and JIT combinations."""
        simple_vp_process = VariancePreservingProcess()
        # Use a simple field predicting x0=zeros for easier batch testing
        lambda x, t: jnp.zeros_like(x)

        # Convert to the required input type
        def get_zero_pred_field(x, t):
            alpha_t = simple_vp_process.alpha(t)
            sigma_t = simple_vp_process.sigma(t)
            alpha_prime_t = simple_vp_process.alpha_prime(t)
            sigma_prime_t = simple_vp_process.sigma_prime(t)
            return convert_vector_field_type(
                x,
                jnp.zeros_like(x),
                alpha_t,
                sigma_t,
                alpha_prime_t,
                sigma_prime_t,
                VectorFieldType.X0,
                input_vf_type,
            )

        sampler = DDMSampler(
            simple_vp_process, get_zero_pred_field, input_vf_type, use_stochastic
        )

        # Prepare batched inputs
        batch_size = 3
        key = jax.random.key(124)
        key, x0_key, zs_key = jax.random.split(key, 3)
        batched_x_init = jax.random.normal(x0_key, (batch_size,) + self.data_shape)
        batched_zs = jax.random.normal(
            zs_key, (batch_size, self.num_steps) + self.data_shape
        )
        ts = self.ts

        sample_fn = sampler.sample

        # 1. Calculate expected output using vmap on non-jitted partial function
        vmap_sample_expected = jax.vmap(sample_fn, in_axes=(0, 0, None))
        expected_batch_final_x = vmap_sample_expected(batched_x_init, batched_zs, ts)
        assert expected_batch_final_x.shape == (batch_size,) + self.data_shape

        # 2. Vmap only
        vmap_sample = jax.vmap(sample_fn, in_axes=(0, 0, None))
        actual_batch_final_x_vmap = vmap_sample(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_final_x_vmap, expected_batch_final_x, atol=1e-6, rtol=1e-6
        )

        # 3. Jit(Vmap)
        jit_vmap_sample = jax.jit(vmap_sample)
        actual_batch_final_x_jit_vmap = jit_vmap_sample(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_final_x_jit_vmap, expected_batch_final_x, atol=1e-5, rtol=1e-5
        )

        # 4. Vmap(Jit)
        jit_sample = jax.jit(sample_fn)
        vmap_jit_sample = jax.vmap(jit_sample, in_axes=(0, 0, None))
        actual_batch_final_x_vmap_jit = vmap_jit_sample(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_final_x_vmap_jit, expected_batch_final_x, atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("use_stochastic", [False, True])
    @pytest.mark.parametrize("input_vf_type", VectorFieldType)
    def test_sample_trajectory_vmap(self, input_vf_type, use_stochastic):
        """Test DDMSampler.sample_trajectory with VMAP and JIT combinations."""
        simple_vp_process = VariancePreservingProcess()
        lambda x, t: jnp.zeros_like(x)

        def get_zero_pred_field(x, t):
            alpha_t = simple_vp_process.alpha(t)
            sigma_t = simple_vp_process.sigma(t)
            alpha_prime_t = simple_vp_process.alpha_prime(t)
            sigma_prime_t = simple_vp_process.sigma_prime(t)
            return convert_vector_field_type(
                x,
                jnp.zeros_like(x),
                alpha_t,
                sigma_t,
                alpha_prime_t,
                sigma_prime_t,
                VectorFieldType.X0,
                input_vf_type,
            )

        sampler = DDMSampler(
            simple_vp_process, get_zero_pred_field, input_vf_type, use_stochastic
        )

        batch_size = 3
        key = jax.random.key(125)
        key, x0_key, zs_key = jax.random.split(key, 3)
        batched_x_init = jax.random.normal(x0_key, (batch_size,) + self.data_shape)
        batched_zs = jax.random.normal(
            zs_key, (batch_size, self.num_steps) + self.data_shape
        )
        ts = self.ts
        expected_traj_shape = (
            batch_size,
            self.num_steps + 1,
        ) + self.data_shape

        traj_fn = sampler.sample_trajectory

        # 1. Expected
        vmap_traj_expected = jax.vmap(traj_fn, in_axes=(0, 0, None))
        expected_batch_traj = vmap_traj_expected(batched_x_init, batched_zs, ts)
        assert expected_batch_traj.shape == expected_traj_shape

        # 2. Vmap only
        vmap_traj = jax.vmap(traj_fn, in_axes=(0, 0, None))
        actual_batch_traj_vmap = vmap_traj(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_traj_vmap, expected_batch_traj, atol=1e-6, rtol=1e-6
        )

        # 3. Jit(Vmap)
        jit_vmap_traj = jax.jit(vmap_traj)
        actual_batch_traj_jit_vmap = jit_vmap_traj(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_traj_jit_vmap, expected_batch_traj, atol=1e-5, rtol=1e-5
        )

        # 4. Vmap(Jit)
        jit_traj = jax.jit(traj_fn)
        vmap_jit_traj = jax.vmap(jit_traj, in_axes=(0, 0, None))
        actual_batch_traj_vmap_jit = vmap_jit_traj(batched_x_init, batched_zs, ts)
        assert_allclose(
            actual_batch_traj_vmap_jit, expected_batch_traj, atol=1e-5, rtol=1e-5
        )


# Consider adding tests with VE/FM processes as well for broader coverage
# Consider adding tests where vector_field is not identity/zero if conversions are complex
