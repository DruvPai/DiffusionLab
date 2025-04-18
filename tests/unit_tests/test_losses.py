import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
import pytest  # Using pytest as per rules

from diffusionlab.losses import DiffusionLoss, VectorFieldType
from diffusionlab.dynamics import (
    DiffusionProcess,
    VariancePreservingProcess,
    FlowMatchingProcess,
    VarianceExplodingProcess,
)


class TestDiffusionLoss:
    """Tests for the DiffusionLoss class."""

    @pytest.fixture(
        params=[
            VariancePreservingProcess(),
            FlowMatchingProcess(),
            VarianceExplodingProcess(sigma=lambda t: t),
        ]
    )
    def process(self, request):
        """Provides different DiffusionProcess instances for testing."""
        return request.param

    @pytest.mark.parametrize(
        "vector_field_type",
        [VectorFieldType.X0, VectorFieldType.EPS, VectorFieldType.V],
    )
    def test_init_target_functions(
        self, process: DiffusionProcess, vector_field_type: VectorFieldType
    ):
        """Test that the correct target function is set during init."""
        loss_instance = DiffusionLoss(process, vector_field_type, 1)
        key = jax.random.key(0)
        key, x0_key, eps_key = jax.random.split(key, 3)
        x_0 = jax.random.normal(x0_key, (3,))  # (data_dim,)
        eps = jax.random.normal(eps_key, (3,))  # (data_dim,)
        t = jnp.array(0.4)  # scalar
        x_t = process.forward(x_0, t, eps)  # (data_dim,)
        f_x_t = jnp.zeros_like(x_t)  # Dummy prediction (data_dim,)

        target = loss_instance.target(x_t, f_x_t, x_0, eps, t)

        if vector_field_type == VectorFieldType.X0:
            expected_target = x_0
        elif vector_field_type == VectorFieldType.EPS:
            expected_target = eps
        elif vector_field_type == VectorFieldType.V:
            expected_target = (
                process.alpha_prime(t) * x_0 + process.sigma_prime(t) * eps
            )
        else:
            pytest.fail(f"Unexpected vector_field_type: {vector_field_type}")

        assert_allclose(target, expected_target, atol=1e-6)

    def test_init_raises_for_score(self, process: DiffusionProcess):
        """Test that initializing with SCORE target type raises ValueError."""
        with pytest.raises(ValueError, match="Direct score matching is not supported"):
            DiffusionLoss(process, VectorFieldType.SCORE, 1)

    def test_init_raises_for_undefined_enum(self, process: DiffusionProcess):
        """Test that initializing with undefined type raises ValueError."""
        # Use an invalid string which will fail the enum lookup
        with pytest.raises(ValueError, match="Invalid target type"):
            DiffusionLoss(process, "invalid_type", 1)  # type: ignore

    @pytest.mark.parametrize(
        "vector_field_type",
        [VectorFieldType.X0, VectorFieldType.EPS, VectorFieldType.V],
    )
    def test_prediction_loss(
        self, process: DiffusionProcess, vector_field_type: VectorFieldType
    ):
        """Test prediction_loss with different targets."""
        loss_instance = DiffusionLoss(process, vector_field_type, 1)
        key = jax.random.key(
            vector_field_type.value
        )  # Use enum value for different key per type
        key, x0_key, eps_key, fx_key = jax.random.split(key, 4)
        x_0 = jax.random.normal(x0_key, (3,))  # (data_dim,)
        eps = jax.random.normal(eps_key, (3,))  # (data_dim,)
        t = jnp.array(0.4)  # scalar
        x_t = process.forward(x_0, t, eps)  # (data_dim,)
        f_x_t = jax.random.normal(fx_key, (3,))  # Dummy prediction (data_dim,)

        if vector_field_type == VectorFieldType.X0:
            target = x_0
        elif vector_field_type == VectorFieldType.EPS:
            target = eps
        elif vector_field_type == VectorFieldType.V:
            target = process.alpha_prime(t) * x_0 + process.sigma_prime(t) * eps
        else:
            pytest.fail(f"Unexpected vector_field_type: {vector_field_type}")

        expected_loss = jnp.sum((f_x_t - target) ** 2)
        actual_loss = loss_instance.prediction_loss(x_t, f_x_t, x_0, eps, t)
        assert_allclose(actual_loss, expected_loss, atol=1e-6)

    @pytest.mark.parametrize(
        "vector_field_type",
        [VectorFieldType.X0, VectorFieldType.EPS, VectorFieldType.V],
    )
    def test_call_single_noise_draw(
        self, process: DiffusionProcess, vector_field_type: VectorFieldType
    ):
        """Test __call__ with num_noise_draws_per_sample=1."""
        loss_instance = DiffusionLoss(process, vector_field_type, 1)
        key = jax.random.key(4 + vector_field_type.value)  # Different key per type
        key, x0_key, call_key = jax.random.split(key, 3)
        x_0 = jax.random.normal(x0_key, (3,))  # (data_dim,)
        t = jnp.array(0.7)  # scalar

        # Define a simple vector field model
        def dummy_vector_field(xt, time):
            # Example: predict zeros
            return jnp.zeros_like(xt)

        # Generate the single noise sample that __call__ will use internally
        eps_single = jax.random.normal(call_key, x_0.shape)  # (data_dim,)
        x_t_single = process.forward(x_0, t, eps_single)  # (data_dim,)
        f_x_t_single = dummy_vector_field(x_t_single, t)  # (data_dim,)

        # Calculate expected loss using the internal prediction_loss method
        expected_loss = loss_instance.prediction_loss(
            x_t_single, f_x_t_single, x_0, eps_single, t
        )
        actual_loss = loss_instance.loss(call_key, dummy_vector_field, x_0, t)
        assert_allclose(actual_loss, expected_loss, atol=1e-6)

    @pytest.mark.parametrize(
        "vector_field_type",
        [VectorFieldType.X0, VectorFieldType.EPS, VectorFieldType.V],
    )
    def test_call_multiple_noise_draws(
        self, process: DiffusionProcess, vector_field_type: VectorFieldType
    ):
        """Test __call__ with num_noise_draws_per_sample > 1."""
        num_draws = 5
        loss_instance = DiffusionLoss(process, vector_field_type, num_draws)
        key = jax.random.key(6 + vector_field_type.value)  # Different key per type
        # Use a batch dim in x_0
        batch_size = 2
        data_dim = 3
        key, x0_key, call_key = jax.random.split(key, 3)
        x_0 = jax.random.normal(x0_key, (batch_size, data_dim))  # (batch, data_dim)
        t = jnp.array(0.2)  # scalar, applied to all batch elements

        # Vector field that predicts zeros (must handle batch dims potentially added by noise draws)
        def zero_vector_field(xt_maybe_batch, t_maybe_batch):
            return jnp.zeros_like(xt_maybe_batch)

        # --- Manually calculate expected average loss over noise draws ---
        # Replicate the batching and noise generation inside __call__
        # x_0 is expanded along axis 0, t is broadcasted
        x_0_expanded = x_0[None, ...].repeat(
            num_draws, axis=0
        )  # (num_draws, batch, data_dim)
        t_expanded = t[None].repeat(num_draws, axis=0)  # (num_draws,)
        eps_batch = jax.random.normal(
            call_key, x_0_expanded.shape
        )  # (num_draws, batch, data_dim)

        # Vmap forward over the noise draw dimension
        batch_diffusion_forward = jax.vmap(
            process.forward, in_axes=(0, 0, 0)
        )  # Expects (N, *data), (N,), (N, *data) -> (N, *data)
        # We need to vmap over noise draws, but apply the same t to each sample in the original batch dim of x_0
        # So, we vmap x_0_expanded (draws, batch, data), t_expanded (draws,), eps_batch (draws, batch, data)
        x_t_batch = batch_diffusion_forward(
            x_0_expanded, t_expanded, eps_batch
        )  # (num_draws, batch, data_dim)

        # The vector field gets the time input potentially batched by num_draws
        f_x_t_batch = zero_vector_field(
            x_t_batch, t_expanded[:, None]
        )  # Predict zeros (num_draws, batch, data_dim)

        # Calculate target based on vector_field_type
        if vector_field_type == VectorFieldType.X0:
            target_batch = x_0_expanded
        elif vector_field_type == VectorFieldType.EPS:
            target_batch = eps_batch
        elif vector_field_type == VectorFieldType.V:
            # Vmap the target function calculation over the noise draw batch
            batch_target_v_fn = jax.vmap(
                loss_instance.target, in_axes=(0, 0, 0, 0, 0)
            )  # Expects (N, *data), (N, *data), (N, *data), (N, *data), (N,) -> (N, *data)
            target_batch = batch_target_v_fn(
                x_t_batch, f_x_t_batch, x_0_expanded, eps_batch, t_expanded
            )
        else:
            pytest.fail(f"Unexpected vector_field_type: {vector_field_type}")

        # prediction_loss sums over all sample dims (batch, data_dim in this case)
        # We need to sum over the original batch and data dimensions (axes 1 and 2)
        data_axes = tuple(range(1, x_0_expanded.ndim))
        squared_residuals_batch = jnp.sum(
            (f_x_t_batch - target_batch) ** 2, axis=data_axes
        )  # (num_draws,)
        expected_mean_loss = jnp.mean(squared_residuals_batch)  # Mean over noise draws

        # --- Call the actual function ---
        actual_loss = loss_instance.loss(
            call_key, zero_vector_field, x_0, t
        )  # Takes original x_0 and t

        assert_allclose(actual_loss, expected_mean_loss, atol=1e-6)

    @pytest.mark.parametrize(
        "vector_field_type",
        [VectorFieldType.X0, VectorFieldType.EPS, VectorFieldType.V],
    )
    @pytest.mark.parametrize("num_noise_draws", [1, 3])
    def test_call_jit(
        self,
        process: DiffusionProcess,
        vector_field_type: VectorFieldType,
        num_noise_draws: int,
    ):
        """Test __call__ under JIT compilation."""
        loss_instance = DiffusionLoss(process, vector_field_type, num_noise_draws)
        key = jax.random.key(7)
        key, x0_key, call_key = jax.random.split(key, 3)
        x_0 = jax.random.normal(x0_key, (2, 3))  # Example data point
        t = jnp.array(0.6)

        # Simple vector field (predicts zeros)
        def zero_vector_field(xt, time):
            return jnp.zeros_like(xt)

        # Calculate expected loss without JIT
        expected_loss = loss_instance.loss(call_key, zero_vector_field, x_0, t)

        # JIT the call method. We need to make `vector_field` static.
        # Note: Jitting loss_instance directly might work if the vector_field is simple enough or hashable,
        # but using partial and static_argnums is safer.
        jitted_loss_call = jax.jit(loss_instance.loss, static_argnums=(1,))

        # Run JITted version
        actual_loss = jitted_loss_call(call_key, zero_vector_field, x_0, t)

        assert_allclose(
            actual_loss, expected_loss, atol=1e-5
        )  # Slightly higher tolerance for JIT

    @pytest.mark.parametrize(
        "vector_field_type",
        [VectorFieldType.X0, VectorFieldType.EPS, VectorFieldType.V],
    )
    @pytest.mark.parametrize("num_noise_draws", [1, 3])
    @pytest.mark.parametrize("vmap_axes", [(0, None), (None, 0), (0, 0)])
    def test_call_vmap_combinations(
        self,
        process: DiffusionProcess,
        vector_field_type: VectorFieldType,
        num_noise_draws: int,
        vmap_axes,
    ):
        """Test __call__ with VMAP and JIT combinations over x_0 and t."""
        loss_instance = DiffusionLoss(process, vector_field_type, num_noise_draws)
        batch_size = 4
        data_shape = (2, 3)
        key = jax.random.key(8)
        key, base_x0_key, base_t_key, base_call_key = jax.random.split(key, 4)

        # Simple vector field (predicts zeros)
        def zero_vector_field(xt, time):
            # Ensure it handles potential batch dims from internal noise draws
            return jnp.zeros_like(xt)

        # --- Prepare inputs based on vmap_axes ---
        x0_vmap_axis, t_vmap_axis = vmap_axes
        key_vmap_axis = 0 if (x0_vmap_axis == 0 or t_vmap_axis == 0) else None

        x0_shape = (batch_size,) + data_shape if x0_vmap_axis == 0 else data_shape
        t_shape = (batch_size,) if t_vmap_axis == 0 else ()

        x0 = jax.random.normal(base_x0_key, x0_shape)
        t = jax.random.uniform(base_t_key, shape=t_shape) * 0.9 + 0.05  # Avoid 0 or 1
        keys = (
            jax.random.split(base_call_key, batch_size)
            if key_vmap_axis == 0
            else base_call_key
        )

        # --- Calculate expected loss (using vmap on the non-jitted function) ---
        # Define the function signature for vmap
        call_fn = loss_instance.loss

        # Vmap the scalar function to get expected batch output
        vmapped_call_expected = jax.vmap(
            call_fn, in_axes=(key_vmap_axis, None, x0_vmap_axis, t_vmap_axis)
        )
        expected_loss_batch = vmapped_call_expected(keys, zero_vector_field, x0, t)
        expected_loss = jnp.mean(
            expected_loss_batch
        )  # Loss is typically averaged over batch

        # --- Test Vmap variations (expecting mean loss over batch) ---
        # 1. Vmap only
        vmapped_call = jax.vmap(
            call_fn,
            in_axes=(key_vmap_axis, None, x0_vmap_axis, t_vmap_axis),
            out_axes=0,
        )
        actual_loss_batch_vmap = vmapped_call(keys, zero_vector_field, x0, t)
        assert actual_loss_batch_vmap.shape == (batch_size,)
        assert_allclose(jnp.mean(actual_loss_batch_vmap), expected_loss, atol=1e-6)

        # 2. Jit(Vmap)
        jit_vmapped_call = jax.jit(vmapped_call, static_argnums=(1,))
        actual_loss_batch_jit_vmap = jit_vmapped_call(keys, zero_vector_field, x0, t)
        assert actual_loss_batch_jit_vmap.shape == (batch_size,)
        assert_allclose(
            jnp.mean(actual_loss_batch_jit_vmap), expected_loss, atol=1e-5
        )  # Higher tol for JIT

        # 3. Vmap(Jit)
        # Need static_argnums for the jitted function before vmapping
        jitted_call_scalar = jax.jit(call_fn, static_argnums=(1,))
        vmap_jitted_call = jax.vmap(
            jitted_call_scalar,
            in_axes=(key_vmap_axis, None, x0_vmap_axis, t_vmap_axis),
            out_axes=0,
        )
        actual_loss_batch_vmap_jit = vmap_jitted_call(keys, zero_vector_field, x0, t)
        assert actual_loss_batch_vmap_jit.shape == (batch_size,)
        assert_allclose(
            jnp.mean(actual_loss_batch_vmap_jit), expected_loss, atol=1e-5
        )  # Higher tol for JIT
