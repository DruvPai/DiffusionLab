import jax
import jax.numpy as jnp
from numpy.testing import assert_allclose
import pytest

from diffusionlab.dynamics import (
    DiffusionProcess,
    FlowMatchingProcess,
    VarianceExplodingProcess,
    VariancePreservingProcess,
)


class TestDiffusionProcess:
    """Tests for the base DiffusionProcess class."""

    def test_init_and_derivatives(self) -> None:
        """Test initialization and automatic derivative calculation."""
        alpha_fn = lambda t: t**2
        sigma_fn = lambda t: jnp.sin(t)
        process = DiffusionProcess(alpha=alpha_fn, sigma=sigma_fn)

        t_val = jnp.array(0.3)
        expected_alpha = t_val**2
        expected_sigma = jnp.sin(t_val)
        expected_alpha_prime = 2 * t_val
        expected_sigma_prime = jnp.cos(t_val)

        assert_allclose(process.alpha(t_val), expected_alpha, atol=1e-6)
        assert_allclose(process.sigma(t_val), expected_sigma, atol=1e-6)
        assert_allclose(process.alpha_prime(t_val), expected_alpha_prime, atol=1e-6)
        assert_allclose(process.sigma_prime(t_val), expected_sigma_prime, atol=1e-6)

    def test_forward(self) -> None:
        """Test the forward diffusion calculation."""
        alpha_fn = lambda t: 1.0 - t
        sigma_fn = lambda t: t
        process = DiffusionProcess(alpha=alpha_fn, sigma=sigma_fn)

        key = jax.random.key(0)
        key, subkey = jax.random.split(key)
        x_shape = (3, 4)
        x0 = jax.random.normal(key, x_shape)  # Shape: (3, 4)
        eps = jax.random.normal(subkey, x_shape)  # Shape: (3, 4)
        t = jnp.array(0.3)  # Shape: []

        expected_xt = (1.0 - t) * x0 + t * eps
        actual_xt = process.forward(x0, t, eps)  # Shape: (3, 4)

        assert actual_xt.shape == x_shape
        assert_allclose(actual_xt, expected_xt, atol=1e-6)


class TestVarianceExplodingProcess:
    """Tests for the VarianceExplodingProcess class."""

    def test_init_and_derivatives(self) -> None:
        """Test VE process initialization and derivatives."""
        sigma_fn = lambda t: t**0.5 * 5.0  # Example sigma(t) = 5 * sqrt(t)
        process = VarianceExplodingProcess(sigma=sigma_fn)

        t_val = jnp.array(0.4)
        expected_alpha = jnp.array(1.0)
        expected_sigma = 5.0 * jnp.sqrt(t_val)
        expected_alpha_prime = jnp.array(0.0)
        expected_sigma_prime = 5.0 * 0.5 * t_val ** (-0.5)  # d(5*t^0.5)/dt = 2.5*t^-0.5

        assert_allclose(process.alpha(t_val), expected_alpha)
        assert_allclose(process.sigma(t_val), expected_sigma, atol=1e-6)
        assert_allclose(process.alpha_prime(t_val), expected_alpha_prime)
        assert_allclose(process.sigma_prime(t_val), expected_sigma_prime, atol=1e-6)

    def test_forward(self) -> None:
        """Test the VE forward diffusion calculation."""
        sigma_fn = lambda t: t * 2.0  # Example sigma(t) = 2t
        process = VarianceExplodingProcess(sigma=sigma_fn)

        key = jax.random.key(1)
        key, subkey = jax.random.split(key)
        x_shape = (5,)
        x0 = jax.random.normal(key, x_shape)  # Shape: (5,)
        eps = jax.random.normal(subkey, x_shape)  # Shape: (5,)
        t = jnp.array(0.7)  # Shape: []

        expected_xt = x0 + (2.0 * t) * eps
        actual_xt = process.forward(x0, t, eps)  # Shape: (5,)

        assert actual_xt.shape == x_shape
        assert_allclose(actual_xt, expected_xt, atol=1e-6)

    def test_forward_jit(self) -> None:
        """Test the VE forward diffusion calculation with JIT."""
        sigma_fn = lambda t: t * 2.0
        process = VarianceExplodingProcess(sigma=sigma_fn)
        jitted_forward = jax.jit(process.forward)

        key = jax.random.key(11)
        key, subkey = jax.random.split(key)
        x_shape = (5,)
        x0 = jax.random.normal(key, x_shape)
        eps = jax.random.normal(subkey, x_shape)
        t = jnp.array(0.7)

        expected_xt = x0 + (2.0 * t) * eps
        actual_xt = jitted_forward(x0, t, eps)

        assert actual_xt.shape == x_shape
        assert_allclose(actual_xt, expected_xt, atol=1e-6)

    @pytest.mark.parametrize(
        "vmap_axes",
        [
            (0, None, None),  # vmap over x0
            (None, 0, None),  # vmap over t
            (None, None, 0),  # vmap over eps
            (0, 0, None),  # vmap over x0, t
            (0, None, 0),  # vmap over x0, eps
            (None, 0, 0),  # vmap over t, eps
            (0, 0, 0),  # vmap over x0, t, eps
        ],
    )
    def test_forward_vmap_combinations(self, vmap_axes) -> None:
        """Test the VE forward diffusion calculation with VMAP and JIT combinations."""
        sigma_fn = lambda t: t * 3.0
        process = VarianceExplodingProcess(sigma=sigma_fn)
        batch_size = 4
        key = jax.random.key(12)

        # --- Prepare inputs based on vmap_axes ---
        key, x0_key, eps_key, t_key = jax.random.split(key, 4)
        x_dim = 5
        x0 = jax.random.normal(
            x0_key, (batch_size, x_dim) if vmap_axes[0] == 0 else (x_dim,)
        )
        eps = jax.random.normal(
            eps_key, (batch_size, x_dim) if vmap_axes[2] == 0 else (x_dim,)
        )
        t = (
            jax.random.uniform(t_key, (batch_size,) if vmap_axes[1] == 0 else ()) * 0.9
            + 0.05
        )  # Avoid t=0 or t=1 issues

        # --- Define expected output function ---
        def expected_fn(x, time, noise):
            return x + sigma_fn(time) * noise

        # --- Calculate expected output (manually handling broadcasting/vmap) ---
        if vmap_axes == (0, None, None):
            expected_xt = expected_fn(x0, t, eps)  # Auto-broadcasts t, eps
        elif vmap_axes == (None, 0, None):
            expected_xt = jax.vmap(lambda time: expected_fn(x0, time, eps))(t)
        elif vmap_axes == (None, None, 0):
            expected_xt = jax.vmap(lambda noise: expected_fn(x0, t, noise))(eps)
        elif vmap_axes == (0, 0, None):
            expected_xt = jax.vmap(lambda x, time: expected_fn(x, time, eps))(x0, t)
        elif vmap_axes == (0, None, 0):
            expected_xt = jax.vmap(lambda x, noise: expected_fn(x, t, noise))(x0, eps)
        elif vmap_axes == (None, 0, 0):
            expected_xt = jax.vmap(lambda time, noise: expected_fn(x0, time, noise))(
                t, eps
            )
        elif vmap_axes == (0, 0, 0):
            expected_xt = jax.vmap(expected_fn)(x0, t, eps)
        else:
            raise ValueError("Invalid vmap_axes")

        # --- Test vmap variations ---
        # 1. Vmap only
        vmapped_forward = jax.vmap(process.forward, in_axes=vmap_axes)
        actual_xt_vmap = vmapped_forward(x0, t, eps)
        assert actual_xt_vmap.shape == expected_xt.shape
        assert_allclose(actual_xt_vmap, expected_xt, atol=1e-6)

        # 2. Jit(Vmap)
        jit_vmapped_forward = jax.jit(vmapped_forward)
        actual_xt_jit_vmap = jit_vmapped_forward(x0, t, eps)
        assert actual_xt_jit_vmap.shape == expected_xt.shape
        assert_allclose(actual_xt_jit_vmap, expected_xt, atol=1e-6)

        # 3. Vmap(Jit)
        jitted_forward_scalar = jax.jit(process.forward)
        vmap_jit_forward = jax.vmap(jitted_forward_scalar, in_axes=vmap_axes)
        actual_xt_vmap_jit = vmap_jit_forward(x0, t, eps)
        assert actual_xt_vmap_jit.shape == expected_xt.shape
        assert_allclose(actual_xt_vmap_jit, expected_xt, atol=1e-6)


class TestVariancePreservingProcess:
    """Tests for the VariancePreservingProcess class."""

    def test_init_and_derivatives(self) -> None:
        """Test VP process initialization and derivatives."""
        process = VariancePreservingProcess()

        t_val = jnp.array(0.6)  # 0 < t < 1
        expected_alpha = jnp.sqrt(1.0 - t_val**2)
        expected_sigma = t_val
        # d(sqrt(1-t^2))/dt = 0.5 * (1-t^2)^(-0.5) * (-2t) = -t / sqrt(1-t^2)
        expected_alpha_prime = -t_val / jnp.sqrt(1.0 - t_val**2)
        expected_sigma_prime = jnp.array(1.0)  # d(t)/dt = 1

        assert_allclose(process.alpha(t_val), expected_alpha, atol=1e-6)
        assert_allclose(process.sigma(t_val), expected_sigma, atol=1e-6)
        assert_allclose(process.alpha_prime(t_val), expected_alpha_prime, atol=1e-6)
        assert_allclose(process.sigma_prime(t_val), expected_sigma_prime, atol=1e-6)

        # Test edge case t=0
        t_zero = jnp.array(0.0)
        assert_allclose(process.alpha(t_zero), 1.0)
        assert_allclose(process.sigma(t_zero), 0.0)
        assert_allclose(
            process.alpha_prime(t_zero), 0.0
        )  # lim t->0 [-t/sqrt(1-t^2)] = 0
        assert_allclose(process.sigma_prime(t_zero), 1.0)

        # Test edge case t approaches 1 (derivative goes to -inf)
        t_almost_one = jnp.array(1.0 - 1e-7)
        assert_allclose(
            process.alpha(t_almost_one), jnp.sqrt(1.0 - t_almost_one**2), atol=1e-6
        )
        assert_allclose(process.sigma(t_almost_one), t_almost_one, atol=1e-6)
        # alpha_prime approaches -inf, sigma_prime is 1
        assert process.alpha_prime(t_almost_one) < -1e3  # Check it's large negative
        assert_allclose(process.sigma_prime(t_almost_one), 1.0)

    def test_forward(self) -> None:
        """Test the VP forward diffusion calculation."""
        process = VariancePreservingProcess()

        key = jax.random.key(2)
        key, subkey = jax.random.split(key)
        x_shape = (2, 2)
        x0 = jax.random.normal(key, x_shape)  # Shape: (2, 2)
        eps = jax.random.normal(subkey, x_shape)  # Shape: (2, 2)
        t = jnp.array(0.8)  # Shape: []

        alpha_t = jnp.sqrt(1.0 - t**2)
        sigma_t = t
        expected_xt = alpha_t * x0 + sigma_t * eps
        actual_xt = process.forward(x0, t, eps)  # Shape: (2, 2)

        assert actual_xt.shape == x_shape
        assert_allclose(actual_xt, expected_xt, atol=1e-6)

    def test_forward_jit(self) -> None:
        """Test the VP forward diffusion calculation with JIT."""
        process = VariancePreservingProcess()
        jitted_forward = jax.jit(process.forward)

        key = jax.random.key(21)
        key, subkey = jax.random.split(key)
        x_shape = (2, 2)
        x0 = jax.random.normal(key, x_shape)
        eps = jax.random.normal(subkey, x_shape)
        t = jnp.array(0.8)

        alpha_t = jnp.sqrt(1.0 - t**2)
        sigma_t = t
        expected_xt = alpha_t * x0 + sigma_t * eps
        actual_xt = jitted_forward(x0, t, eps)

        assert actual_xt.shape == x_shape
        assert_allclose(actual_xt, expected_xt, atol=1e-6)

    @pytest.mark.parametrize(
        "vmap_axes",
        [
            (0, None, None),
            (None, 0, None),
            (None, None, 0),
            (0, 0, None),
            (0, None, 0),
            (None, 0, 0),
            (0, 0, 0),
        ],
    )
    def test_forward_vmap_combinations(self, vmap_axes) -> None:
        """Test the VP forward diffusion calculation with VMAP and JIT combinations."""
        process = VariancePreservingProcess()
        batch_size = 4
        key = jax.random.key(22)

        # --- Prepare inputs ---
        key, x0_key, eps_key, t_key = jax.random.split(key, 4)
        x_dim = (2, 2)
        x0_shape = (batch_size,) + x_dim if vmap_axes[0] == 0 else x_dim
        eps_shape = (batch_size,) + x_dim if vmap_axes[2] == 0 else x_dim
        t_shape = (batch_size,) if vmap_axes[1] == 0 else ()

        x0 = jax.random.normal(x0_key, x0_shape)
        eps = jax.random.normal(eps_key, eps_shape)
        t = jax.random.uniform(t_key, t_shape) * 0.9 + 0.05  # Avoid t=0 or t=1 issues

        # --- Define expected output function (scalar version) ---
        def expected_fn(x, time, noise):
            alpha_t = jnp.sqrt(1.0 - time**2)
            sigma_t = time
            return alpha_t * x + sigma_t * noise

        # --- Calculate expected output ---
        if vmap_axes == (0, None, None):
            expected_xt = expected_fn(x0, t, eps)  # Auto-broadcasts t, eps
        elif vmap_axes == (None, 0, None):
            # Need to reshape t for broadcasting with x0/eps shape (2, 2)
            t_reshaped = t.reshape((-1,) + (1,) * len(x_dim))
            expected_xt = expected_fn(x0, t_reshaped, eps)
        elif vmap_axes == (None, None, 0):
            expected_xt = expected_fn(x0, t, eps)  # Auto-broadcasts t, x0
        elif vmap_axes == (0, 0, None):
            expected_xt = jax.vmap(lambda x, time: expected_fn(x, time, eps))(x0, t)
        elif vmap_axes == (0, None, 0):
            expected_xt = jax.vmap(lambda x, noise: expected_fn(x, t, noise))(x0, eps)
        elif vmap_axes == (None, 0, 0):
            t_reshaped = t.reshape((-1,) + (1,) * len(x_dim))
            expected_xt = expected_fn(x0, t_reshaped, eps)
        elif vmap_axes == (0, 0, 0):
            expected_xt = jax.vmap(expected_fn)(x0, t, eps)
        else:
            raise ValueError("Invalid vmap_axes")

        # --- Test vmap variations ---
        vmapped_forward = jax.vmap(process.forward, in_axes=vmap_axes)
        actual_xt_vmap = vmapped_forward(x0, t, eps)
        assert actual_xt_vmap.shape == expected_xt.shape
        assert_allclose(actual_xt_vmap, expected_xt, atol=1e-6)

        jit_vmapped_forward = jax.jit(vmapped_forward)
        actual_xt_jit_vmap = jit_vmapped_forward(x0, t, eps)
        assert actual_xt_jit_vmap.shape == expected_xt.shape
        assert_allclose(actual_xt_jit_vmap, expected_xt, atol=1e-6)

        jitted_forward_scalar = jax.jit(process.forward)
        vmap_jit_forward = jax.vmap(jitted_forward_scalar, in_axes=vmap_axes)
        actual_xt_vmap_jit = vmap_jit_forward(x0, t, eps)
        assert actual_xt_vmap_jit.shape == expected_xt.shape
        assert_allclose(actual_xt_vmap_jit, expected_xt, atol=1e-6)


class TestFlowMatchingProcess:
    """Tests for the FlowMatchingProcess class."""

    def test_init_and_derivatives(self) -> None:
        """Test FM process initialization and derivatives."""
        process = FlowMatchingProcess()

        t_val = jnp.array(0.25)
        expected_alpha = 1.0 - t_val
        expected_sigma = t_val
        expected_alpha_prime = jnp.array(-1.0)  # d(1-t)/dt = -1
        expected_sigma_prime = jnp.array(1.0)  # d(t)/dt = 1

        assert_allclose(process.alpha(t_val), expected_alpha, atol=1e-6)
        assert_allclose(process.sigma(t_val), expected_sigma, atol=1e-6)
        assert_allclose(process.alpha_prime(t_val), expected_alpha_prime)
        assert_allclose(process.sigma_prime(t_val), expected_sigma_prime)

    def test_forward(self) -> None:
        """Test the FM forward diffusion calculation."""
        process = FlowMatchingProcess()

        key = jax.random.key(3)
        key, subkey = jax.random.split(key)
        x_shape = (10,)
        x0 = jax.random.normal(key, x_shape)  # Shape: (10,)
        eps = jax.random.normal(subkey, x_shape)  # Shape: (10,)
        t = jnp.array(0.1)  # Shape: []

        alpha_t = 1.0 - t
        sigma_t = t
        expected_xt = alpha_t * x0 + sigma_t * eps
        actual_xt = process.forward(x0, t, eps)  # Shape: (10,)

        assert actual_xt.shape == x_shape
        assert_allclose(actual_xt, expected_xt, atol=1e-6)

    def test_forward_jit(self) -> None:
        """Test the FM forward diffusion calculation with JIT."""
        process = FlowMatchingProcess()
        jitted_forward = jax.jit(process.forward)

        key = jax.random.key(31)
        key, subkey = jax.random.split(key)
        x_shape = (10,)
        x0 = jax.random.normal(key, x_shape)
        eps = jax.random.normal(subkey, x_shape)
        t = jnp.array(0.1)

        alpha_t = 1.0 - t
        sigma_t = t
        expected_xt = alpha_t * x0 + sigma_t * eps
        actual_xt = jitted_forward(x0, t, eps)

        assert actual_xt.shape == x_shape
        assert_allclose(actual_xt, expected_xt, atol=1e-6)

    @pytest.mark.parametrize(
        "vmap_axes",
        [
            (0, None, None),
            (None, 0, None),
            (None, None, 0),
            (0, 0, None),
            (0, None, 0),
            (None, 0, 0),
            (0, 0, 0),
        ],
    )
    def test_forward_vmap_combinations(self, vmap_axes) -> None:
        """Test the FM forward diffusion calculation with VMAP and JIT combinations."""
        process = FlowMatchingProcess()
        batch_size = 4
        key = jax.random.key(32)

        # --- Prepare inputs ---
        key, x0_key, eps_key, t_key = jax.random.split(key, 4)
        x_dim = 10
        x0_shape = (batch_size, x_dim) if vmap_axes[0] == 0 else (x_dim,)
        eps_shape = (batch_size, x_dim) if vmap_axes[2] == 0 else (x_dim,)
        t_shape = (batch_size,) if vmap_axes[1] == 0 else ()

        x0 = jax.random.normal(x0_key, x0_shape)
        eps = jax.random.normal(eps_key, eps_shape)
        t = jax.random.uniform(t_key, t_shape) * 0.9 + 0.05  # Avoid t=0 or t=1 issues

        # --- Define expected output function (scalar version) ---
        def expected_fn(x, time, noise):
            alpha_t = 1.0 - time
            sigma_t = time
            return alpha_t * x + sigma_t * noise

        # --- Calculate expected output ---
        # Handle broadcasting/vmapping explicitly based on vmap_axes
        if vmap_axes == (0, None, None):
            expected_xt = expected_fn(x0, t, eps)
        elif vmap_axes == (None, 0, None):
            # vmap expected_fn over t
            expected_xt = jax.vmap(lambda time: expected_fn(x0, time, eps))(t)
        elif vmap_axes == (None, None, 0):
            # vmap expected_fn over eps
            expected_xt = jax.vmap(lambda noise: expected_fn(x0, t, noise))(eps)
        elif vmap_axes == (0, 0, None):
            # vmap expected_fn over x0 and t
            expected_xt = jax.vmap(lambda x, time: expected_fn(x, time, eps))(x0, t)
        elif vmap_axes == (0, None, 0):
            # vmap expected_fn over x0 and eps
            expected_xt = jax.vmap(lambda x, noise: expected_fn(x, t, noise))(x0, eps)
        elif vmap_axes == (None, 0, 0):
            # vmap expected_fn over t and eps
            expected_xt = jax.vmap(lambda time, noise: expected_fn(x0, time, noise))(
                t, eps
            )
        elif vmap_axes == (0, 0, 0):
            # vmap expected_fn over all
            expected_xt = jax.vmap(expected_fn)(x0, t, eps)
        else:
            raise ValueError("Invalid vmap_axes")

        # --- Test vmap variations ---
        vmapped_forward = jax.vmap(process.forward, in_axes=vmap_axes)
        actual_xt_vmap = vmapped_forward(x0, t, eps)
        assert actual_xt_vmap.shape == expected_xt.shape
        assert_allclose(actual_xt_vmap, expected_xt, atol=1e-6)

        jit_vmapped_forward = jax.jit(vmapped_forward)
        actual_xt_jit_vmap = jit_vmapped_forward(x0, t, eps)
        assert actual_xt_jit_vmap.shape == expected_xt.shape
        assert_allclose(actual_xt_jit_vmap, expected_xt, atol=1e-6)

        jitted_forward_scalar = jax.jit(process.forward)
        vmap_jit_forward = jax.vmap(jitted_forward_scalar, in_axes=vmap_axes)
        actual_xt_vmap_jit = vmap_jit_forward(x0, t, eps)
        assert actual_xt_vmap_jit.shape == expected_xt.shape
        assert_allclose(actual_xt_vmap_jit, expected_xt, atol=1e-6)
