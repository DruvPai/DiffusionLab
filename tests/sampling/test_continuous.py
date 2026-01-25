"""Tests for Gaussian samplers (DDIM and Euler)."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.gaussian import (
    EDM,
    VE,
    FlowMatching,
    GaussianDiffusionProcess,
    VPCosine,
    VPLinear,
)
from diffusionlab.sampling.base import Sampler, SamplerStep, sample
from diffusionlab.sampling.continuous import DDMSampler, EulerSampler
from diffusionlab.sampling.schedules import edm_schedule, uniform_schedule
from diffusionlab.typing import Prediction
from diffusionlab.utils.tree_ops import bcast_right

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
DATA_SHAPE = (8,)
NUM_STEPS = 5

GAUSSIAN_PROCESS_NAMES = ["vpcosine", "vplinear", "ve", "edm", "flow_matching"]

# Processes suitable for DDIM (all Gaussian)
DDIM_PROCESS_NAMES = GAUSSIAN_PROCESS_NAMES

# Processes suitable for Euler (all Gaussian)
EULER_PROCESS_NAMES = GAUSSIAN_PROCESS_NAMES


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_process(name: str) -> GaussianDiffusionProcess:
    if name == "vpcosine":
        return VPCosine(DATA_SHAPE)
    elif name == "vplinear":
        return VPLinear(DATA_SHAPE)
    elif name == "ve":
        return VE(DATA_SHAPE)
    elif name == "edm":
        return EDM(DATA_SHAPE)
    elif name == "flow_matching":
        return FlowMatching(DATA_SHAPE)
    else:
        raise ValueError(f"Unknown process: {name}")


def _dummy_predict(kind="eps"):
    """Predictor that returns zeros with the given kind."""

    def predict(x, s, t, cond, aux):
        return Prediction(value=jnp.zeros_like(x), kind=kind)

    return predict


def _identity_predict(kind="x_0"):
    """Predictor that returns x itself as x_0 prediction."""

    def predict(x, s, t, cond, aux):
        return Prediction(value=x, kind=kind)

    return predict


# ===========================================================================
# Tests for DDIM sampler
# ===========================================================================


class TestDDIMConstruction:
    """Tests for DDIM construction and properties."""

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_is_sampler(self, name: str):
        """DDIM is a Sampler subclass."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        assert isinstance(sampler, Sampler)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_default_eta(self, name: str):
        """Default eta is 0.0."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        assert sampler.eta == 0.0

    @pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
    def test_custom_eta(self, eta: float):
        """Custom eta is stored."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=eta)
        assert sampler.eta == eta

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_process_stored(self, name: str):
        """Process is accessible."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        assert sampler.process is process


class TestDDIMInitialize:
    """Tests for DDIM.initialize."""

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_returns_sampler_step(self, name: str):
        """initialize returns a SamplerStep."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        key = jax.random.key(0)
        x_T = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        T = jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, T)
        assert isinstance(step, SamplerStep)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_output_shapes(self, name: str):
        """Initialized step has correct shapes."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        key = jax.random.key(0)
        x_T = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        T = jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, T)
        assert step.x.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert step.t.shape == (BATCH_SIZE,)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_preserves_input(self, name: str):
        """initialize preserves x_T and T."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        key = jax.random.key(0)
        x_T = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        T = jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, T)
        assert jnp.allclose(step.x, x_T)
        assert jnp.allclose(step.t, T)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_aux_has_coefficients(self, name: str):
        """Aux contains schedule coefficients."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        key = jax.random.key(0)
        x_T = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        T = jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, T)
        assert "alpha" in step.aux
        assert "sigma" in step.aux


class TestDDIMUpdate:
    """Tests for DDIM.update step logic."""

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_output_shapes(self, name: str):
        """update returns a SamplerStep with correct shapes."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="eps")
        new_step = sampler.update(key, pred, step, s)

        assert new_step.x.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.allclose(new_step.t, s)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_time_advances(self, name: str):
        """Updated step has the target time s."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="eps")
        new_step = sampler.update(key, pred, step, s)

        assert jnp.allclose(new_step.t, s)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_eta_zero_is_deterministic(self, name: str):
        """DDIM with eta=0 gives same output for different keys."""
        process = _make_process(name)
        sampler = DDMSampler(process, eta=0.0)
        x_t = jax.random.normal(jax.random.key(0), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(jax.random.key(0), x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="eps")

        out1 = sampler.update(jax.random.key(1), pred, step, s)
        out2 = sampler.update(jax.random.key(2), pred, step, s)
        assert jnp.allclose(out1.x, out2.x)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_eta_one_is_stochastic(self, name: str):
        """DDIM with eta=1 gives different outputs for different keys."""
        process = _make_process(name)
        sampler = DDMSampler(process, eta=1.0)
        x_t = jax.random.normal(jax.random.key(0), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(jax.random.key(0), x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="eps")

        out1 = sampler.update(jax.random.key(1), pred, step, s)
        out2 = sampler.update(jax.random.key(2), pred, step, s)
        assert not jnp.allclose(out1.x, out2.x)

    def test_ddim_update_formula_eta_zero(self):
        """Verify DDIM update formula for eta=0: x_s = alpha_s * x_0_hat + sigma_s * eps_hat."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        key = jax.random.key(42)

        x_0_true = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        eps_true = jax.random.normal(jax.random.key(2), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.7 * jnp.ones((BATCH_SIZE,))
        s = 0.3 * jnp.ones((BATCH_SIZE,))

        alpha_t = bcast_right(process.alpha(t), x_0_true.ndim)
        sigma_t = bcast_right(process.sigma(t), x_0_true.ndim)
        alpha_s = bcast_right(process.alpha(s), x_0_true.ndim)
        sigma_s = bcast_right(process.sigma(s), x_0_true.ndim)

        x_t = alpha_t * x_0_true + sigma_t * eps_true

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=eps_true, kind="eps")
        new_step = sampler.update(key, pred, step, s)

        # With eta=0, sigma_tilde=0, so x_s = alpha_s * x_0_hat + sigma_s * eps_hat
        expected = alpha_s * x_0_true + sigma_s * eps_true
        assert jnp.allclose(new_step.x, expected, atol=1e-5)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_update_produces_finite(self, name: str):
        """Update step produces finite values."""
        process = _make_process(name)
        sampler = DDMSampler(process, eta=0.5)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="eps")
        new_step = sampler.update(key, pred, step, s)
        assert jnp.all(jnp.isfinite(new_step.x))


class TestDDIMFinalize:
    """Tests for DDIM.finalize."""

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_output_shape(self, name: str):
        """finalize returns data with correct shape."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.1 * jnp.ones((BATCH_SIZE,))

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=jnp.zeros_like(x_t), kind="eps")
        out = sampler.finalize(key, pred, step)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_finalize_extracts_x0(self, name: str):
        """finalize returns the x_0 prediction."""
        process = _make_process(name)
        sampler = DDMSampler(process)
        key = jax.random.key(0)

        x_0_true = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.1 * jnp.ones((BATCH_SIZE,))

        step = SamplerStep(x=x_0_true, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=x_0_true, kind="x_0")
        out = sampler.finalize(key, pred, step)
        assert jnp.allclose(out, x_0_true, atol=1e-5)


class TestDDIMEndToEnd:
    """End-to-end tests for DDIM through sample()."""

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_sample_output_shape(self, name: str):
        """sample() with DDIM returns correct shape."""
        process = _make_process(name)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        out = sample(key, sampler, _dummy_predict(), BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_sample_finite(self, name: str):
        """sample() with DDIM produces finite values."""
        process = _make_process(name)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        # FlowMatching has alpha(1)=0, so eps=0 prediction causes div-by-zero;
        # use x_0 prediction instead for FlowMatching
        kind = "x_0" if name == "flow_matching" else "eps"
        out = sample(key, sampler, _dummy_predict(kind), BATCH_SIZE, sched)
        assert jnp.all(jnp.isfinite(out))

    def test_ddim_deterministic_end_to_end(self):
        """DDIM eta=0 through sample() is deterministic."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _dummy_predict()

        out1 = sample(key, sampler, predict, BATCH_SIZE, sched)
        out2 = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.allclose(out1, out2)

    def test_ddpm_stochastic_end_to_end(self):
        """DDIM eta=1 (DDPM) through sample() produces different outputs for different keys."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=1.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        predict = _dummy_predict()

        out1 = sample(jax.random.key(0), sampler, predict, BATCH_SIZE, sched)
        out2 = sample(jax.random.key(1), sampler, predict, BATCH_SIZE, sched)
        assert not jnp.allclose(out1, out2)

    @pytest.mark.parametrize("name", DDIM_PROCESS_NAMES)
    def test_with_edm_schedule(self, name: str):
        """DDIM works with EDM schedule."""
        process = _make_process(name)
        sampler = DDMSampler(process, eta=0.0)
        sched = edm_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        out = sample(key, sampler, _dummy_predict(), BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_various_eta(self, eta: float):
        """DDIM works with various eta values."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=eta)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        out = sample(key, sampler, _dummy_predict(), BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))

    def test_prediction_kinds_eps(self):
        """DDIM correctly handles eps predictions."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        out = sample(key, sampler, _dummy_predict("eps"), BATCH_SIZE, sched)
        assert jnp.all(jnp.isfinite(out))

    def test_prediction_kinds_x0(self):
        """DDIM correctly handles x_0 predictions."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="x_0")

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.all(jnp.isfinite(out))

    def test_prediction_kinds_v(self):
        """DDIM correctly handles v predictions."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="v_t")

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.all(jnp.isfinite(out))


# ===========================================================================
# Tests for Euler sampler
# ===========================================================================


class TestEulerConstruction:
    """Tests for Euler construction and properties."""

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_is_sampler(self, name: str):
        """Euler is a Sampler subclass."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        assert isinstance(sampler, Sampler)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_process_stored(self, name: str):
        """Process is accessible."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        assert sampler.process is process


class TestEulerInitialize:
    """Tests for Euler.initialize."""

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_returns_sampler_step(self, name: str):
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_T = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        T = jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, T)
        assert isinstance(step, SamplerStep)
        assert jnp.allclose(step.x, x_T)
        assert jnp.allclose(step.t, T)


class TestEulerUpdate:
    """Tests for Euler.update step logic."""

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_output_shapes(self, name: str):
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="v_t")
        new_step = sampler.update(key, pred, step, s)

        assert new_step.x.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.allclose(new_step.t, s)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_is_deterministic(self, name: str):
        """Euler is always deterministic (no noise injection)."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        x_t = jax.random.normal(jax.random.key(0), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(jax.random.key(0), x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="v_t")

        out1 = sampler.update(jax.random.key(1), pred, step, s)
        out2 = sampler.update(jax.random.key(2), pred, step, s)
        assert jnp.allclose(out1.x, out2.x)

    def test_euler_step_formula(self):
        """Verify Euler update: x_s = x_t + (s - t) * v_hat."""
        process = FlowMatching(DATA_SHAPE)
        sampler = EulerSampler(process)
        key = jax.random.key(0)

        x_t = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        v_hat = jax.random.normal(jax.random.key(2), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))
        dt = bcast_right(s - t, x_t.ndim)

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=v_hat, kind="v_t")
        new_step = sampler.update(key, pred, step, s)

        expected = x_t + dt * v_hat
        assert jnp.allclose(new_step.x, expected, atol=1e-5)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_zero_velocity_no_change(self, name: str):
        """Zero velocity prediction means x_s = x_t."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        # Zero velocity => x_s = x_t (since v is converted from zero)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="v_t")
        new_step = sampler.update(key, pred, step, s)

        assert jnp.allclose(new_step.x, x_t, atol=1e-5)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_update_produces_finite(self, name: str):
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(
            value=jax.random.normal(jax.random.key(1), x_t.shape), kind="v_t"
        )
        new_step = sampler.update(key, pred, step, s)
        assert jnp.all(jnp.isfinite(new_step.x))


class TestEulerFinalize:
    """Tests for Euler.finalize."""

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_output_shape(self, name: str):
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.05 * jnp.ones((BATCH_SIZE,))

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=jnp.zeros_like(x_t), kind="v_t")
        out = sampler.finalize(key, pred, step)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)


class TestEulerEndToEnd:
    """End-to-end tests for Euler through sample()."""

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_sample_output_shape(self, name: str):
        process = _make_process(name)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="v_t")

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_sample_finite(self, name: str):
        process = _make_process(name)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="v_t")

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.all(jnp.isfinite(out))

    def test_euler_deterministic_end_to_end(self):
        """Euler is deterministic through sample()."""
        process = FlowMatching(DATA_SHAPE)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="v_t")

        out1 = sample(key, sampler, predict, BATCH_SIZE, sched)
        out2 = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_with_edm_schedule(self, name: str):
        """Euler works with EDM schedule."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        sched = edm_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="v_t")

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))

    def test_euler_handles_eps_prediction(self):
        """Euler converts eps predictions through process.convert_prediction."""
        # Use VPCosine (not FlowMatching) because FlowMatching has alpha(1)=0,
        # causing division by zero when converting eps at t~1
        process = VPCosine(DATA_SHAPE)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        out = sample(key, sampler, _dummy_predict("eps"), BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))

    def test_euler_handles_x0_prediction(self):
        """Euler converts x_0 predictions through process.convert_prediction."""
        process = FlowMatching(DATA_SHAPE)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="x_0")

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))


# ===========================================================================
# Tests for Euler sampler with flow-map predictions (x_s, v_st)
# ===========================================================================

FLOW_MAP_KINDS = ["x_s", "v_st"]


class TestEulerFlowMapUpdate:
    """Tests for Euler.update with flow-map prediction kinds (x_s, v_st)."""

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    @pytest.mark.parametrize("kind", FLOW_MAP_KINDS)
    def test_output_shapes(self, name: str, kind: str):
        """Flow-map prediction returns correct shapes."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind=kind)
        new_step = sampler.update(key, pred, step, s)

        assert new_step.x.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.allclose(new_step.t, s)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    @pytest.mark.parametrize("kind", FLOW_MAP_KINDS)
    def test_is_deterministic(self, name: str, kind: str):
        """Euler with flow-map predictions is deterministic."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        x_t = jax.random.normal(jax.random.key(0), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(jax.random.key(0), x_t, t)
        pred = Prediction(value=jnp.ones_like(x_t), kind=kind)

        out1 = sampler.update(jax.random.key(1), pred, step, s)
        out2 = sampler.update(jax.random.key(2), pred, step, s)
        assert jnp.allclose(out1.x, out2.x)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    @pytest.mark.parametrize("kind", FLOW_MAP_KINDS)
    def test_produces_finite(self, name: str, kind: str):
        """Flow-map predictions produce finite values."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(
            value=jax.random.normal(jax.random.key(1), x_t.shape), kind=kind
        )
        new_step = sampler.update(key, pred, step, s)
        assert jnp.all(jnp.isfinite(new_step.x))

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    def test_x_s_prediction_recovers_predicted_value(self, name: str):
        """x_s prediction through Euler returns exactly the predicted x_s.

        The Euler step converts x_s → v_st = (x_t - x_s) / (t - s), then
        x_new = x_t + (s - t) * v_st = x_s.  This identity holds for all
        processes.
        """
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)

        x_t = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        x_s_pred = jax.random.normal(jax.random.key(2), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=x_s_pred, kind="x_s")
        new_step = sampler.update(key, pred, step, s)

        assert jnp.allclose(new_step.x, x_s_pred, atol=1e-5)

    def test_v_st_step_formula(self):
        """Verify Euler step with v_st: x_s = x_t + (s - t) * v_st."""
        process = FlowMatching(DATA_SHAPE)
        sampler = EulerSampler(process)
        key = jax.random.key(0)

        x_t = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        v_st = jax.random.normal(jax.random.key(2), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=v_st, kind="v_st")
        new_step = sampler.update(key, pred, step, s)

        dt = bcast_right(s - t, x_t.ndim)
        expected = x_t + dt * v_st
        assert jnp.allclose(new_step.x, expected, atol=1e-5)

    def test_zero_v_st_no_change(self):
        """Zero v_st prediction means x_s = x_t."""
        process = FlowMatching(DATA_SHAPE)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        pred = Prediction(value=jnp.zeros_like(x_t), kind="v_st")
        new_step = sampler.update(key, pred, step, s)

        assert jnp.allclose(new_step.x, x_t, atol=1e-5)

    def test_flow_map_exact_single_step_v_st(self):
        """Perfect v_st oracle recovers exact x_s for VPCosine.

        Given ground-truth quantities, compute exact v(s,t) = (x_t - x_s) / (t - s).
        Euler step with v_st prediction gives exact x_s.
        """
        process = VPCosine(DATA_SHAPE)
        sampler = EulerSampler(process)
        key = jax.random.key(0)

        x_0 = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        z = jax.random.normal(jax.random.key(2), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.3 * jnp.ones((BATCH_SIZE,))

        alpha_t = bcast_right(process.alpha(t), x_0.ndim)
        sigma_t = bcast_right(process.sigma(t), x_0.ndim)
        alpha_s = bcast_right(process.alpha(s), x_0.ndim)
        sigma_s = bcast_right(process.sigma(s), x_0.ndim)
        dt = bcast_right(t - s, x_0.ndim)

        x_t = alpha_t * x_0 + sigma_t * z
        x_s_true = alpha_s * x_0 + sigma_s * z
        v_st_true = (x_t - x_s_true) / dt

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=v_st_true, kind="v_st")
        new_step = sampler.update(key, pred, step, s)

        assert jnp.allclose(new_step.x, x_s_true, atol=1e-5)

    def test_velocity_kind_selection_uses_v_st(self):
        """Euler uses v_st (not v) when prediction kind is x_s.

        For nonlinear schedules (VPCosine), x_s → v_st and x_s → v give
        different Euler steps.  The identity property (x_new = x_s_pred)
        only holds when v_st is used, confirming the correct branch.
        """
        process = VPCosine(DATA_SHAPE)
        sampler = EulerSampler(process)
        key = jax.random.key(0)

        x_0 = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        z = jax.random.normal(jax.random.key(2), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.3 * jnp.ones((BATCH_SIZE,))

        alpha_t = bcast_right(process.alpha(t), x_0.ndim)
        sigma_t = bcast_right(process.sigma(t), x_0.ndim)
        alpha_s = bcast_right(process.alpha(s), x_0.ndim)
        sigma_s = bcast_right(process.sigma(s), x_0.ndim)

        x_t = alpha_t * x_0 + sigma_t * z
        x_s_true = alpha_s * x_0 + sigma_s * z

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=x_s_true, kind="x_s")
        new_step = sampler.update(key, pred, step, s)

        # v_st route gives back x_s_true exactly (identity property)
        assert jnp.allclose(new_step.x, x_s_true, atol=1e-5)

        # If Euler had used "v_t" instead, the result would differ
        # (VPCosine is nonlinear, so the first-order approximation is inexact)
        v_from_x_s = process.convert_prediction(pred, x_t, s, t, "v_t").value
        dt = bcast_right(s - t, x_0.ndim)
        x_s_via_v = x_t + dt * v_from_x_s
        assert not jnp.allclose(x_s_via_v, x_s_true, atol=1e-5)


class TestEulerFlowMapFinalize:
    """Tests for Euler.finalize with flow-map prediction kinds."""

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    @pytest.mark.parametrize("kind", FLOW_MAP_KINDS)
    def test_output_shape(self, name: str, kind: str):
        """finalize with flow-map prediction returns correct shape."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.normal(key, (BATCH_SIZE, *DATA_SHAPE))
        t = 0.1 * jnp.ones((BATCH_SIZE,))
        t_final = 0.02 * jnp.ones((BATCH_SIZE,))

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=jnp.zeros_like(x_t), kind=kind)
        out = sampler.finalize(key, pred, step, t_final)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)

    def test_finalize_x_s_extracts_x0(self):
        """finalize with perfect x_s prediction correctly recovers x_0."""
        process = VPCosine(DATA_SHAPE)
        sampler = EulerSampler(process)
        key = jax.random.key(0)

        x_0 = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        z = jax.random.normal(jax.random.key(2), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.1 * jnp.ones((BATCH_SIZE,))
        t_final = 0.02 * jnp.ones((BATCH_SIZE,))

        alpha_t = bcast_right(process.alpha(t), x_0.ndim)
        sigma_t = bcast_right(process.sigma(t), x_0.ndim)
        alpha_s = bcast_right(process.alpha(t_final), x_0.ndim)
        sigma_s = bcast_right(process.sigma(t_final), x_0.ndim)

        x_t = alpha_t * x_0 + sigma_t * z
        x_s_true = alpha_s * x_0 + sigma_s * z

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=x_s_true, kind="x_s")
        out = sampler.finalize(key, pred, step, t_final)
        assert jnp.allclose(out, x_0, atol=1e-4)

    def test_finalize_v_st_extracts_x0(self):
        """finalize with perfect v_st prediction correctly recovers x_0."""
        process = VPCosine(DATA_SHAPE)
        sampler = EulerSampler(process)
        key = jax.random.key(0)

        x_0 = jax.random.normal(jax.random.key(1), (BATCH_SIZE, *DATA_SHAPE))
        z = jax.random.normal(jax.random.key(2), (BATCH_SIZE, *DATA_SHAPE))
        t = 0.1 * jnp.ones((BATCH_SIZE,))
        t_final = 0.02 * jnp.ones((BATCH_SIZE,))

        alpha_t = bcast_right(process.alpha(t), x_0.ndim)
        sigma_t = bcast_right(process.sigma(t), x_0.ndim)
        alpha_s = bcast_right(process.alpha(t_final), x_0.ndim)
        sigma_s = bcast_right(process.sigma(t_final), x_0.ndim)
        dt = bcast_right(t - t_final, x_0.ndim)

        x_t = alpha_t * x_0 + sigma_t * z
        x_s_true = alpha_s * x_0 + sigma_s * z
        v_st_true = (x_t - x_s_true) / dt

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=v_st_true, kind="v_st")
        out = sampler.finalize(key, pred, step, t_final)
        assert jnp.allclose(out, x_0, atol=1e-4)


class TestEulerFlowMapEndToEnd:
    """End-to-end tests for Euler with flow-map predictions through sample()."""

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    @pytest.mark.parametrize("kind", FLOW_MAP_KINDS)
    def test_sample_output_shape(self, name: str, kind: str):
        """sample() with flow-map prediction returns correct shape."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind=kind)

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    @pytest.mark.parametrize("kind", FLOW_MAP_KINDS)
    def test_sample_finite(self, name: str, kind: str):
        """sample() with flow-map prediction produces finite values."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind=kind)

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("kind", FLOW_MAP_KINDS)
    def test_deterministic_end_to_end(self, kind: str):
        """Euler with flow-map prediction through sample() is deterministic."""
        process = FlowMatching(DATA_SHAPE)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind=kind)

        out1 = sample(key, sampler, predict, BATCH_SIZE, sched)
        out2 = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.allclose(out1, out2)

    @pytest.mark.parametrize("name", EULER_PROCESS_NAMES)
    @pytest.mark.parametrize("kind", FLOW_MAP_KINDS)
    def test_with_edm_schedule(self, name: str, kind: str):
        """Euler with flow-map prediction works with EDM schedule."""
        process = _make_process(name)
        sampler = EulerSampler(process)
        sched = edm_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind=kind)

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))
