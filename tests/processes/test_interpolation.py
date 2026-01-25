"""Tests for InterpolationProcess (diffusionlab.processes.interpolation)."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.interpolation import InterpolationProcess
from diffusionlab.typing import (
    ContinuousData,
    Prediction,
    PRNGKey,
    Scalar,
    Time,
)
from diffusionlab.utils.tree_ops import bcast_right

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
DATA_DIM = 8


# ---------------------------------------------------------------------------
# Concrete reference implementations
# ---------------------------------------------------------------------------


class FlowMatchingProcess(InterpolationProcess):
    """Flow matching: alpha(t) = 1 - t, sigma(t) = t, source = N(0, I).

    Attributes:
        data_dim: Dimensionality of data (excluding batch).
    """

    def __init__(self, data_dim: int):
        self.data_dim = data_dim

    def alpha(self, t: Time) -> Scalar:
        return 1.0 - t

    def alpha_prime(self, t: Time) -> Scalar:
        return -jnp.ones_like(t)

    def sigma(self, t: Time) -> Scalar:
        return t

    def sigma_prime(self, t: Time) -> Scalar:
        return jnp.ones_like(t)

    def logsnr_inverse(self, lam: Scalar) -> Time:
        # logsnr = 2*(log(1-t) - log(t)) => t = 1/(1 + exp(lam/2))
        return 1.0 / (1.0 + jnp.exp(lam / 2.0))

    def _sample_from_source(self, key: PRNGKey, batch_size: int) -> ContinuousData:
        return jax.random.normal(key, (batch_size, self.data_dim))


class VPProcess(InterpolationProcess):
    """VP: alpha(t) = cos(pi/2 t), sigma(t) = sin(pi/2 t), source = N(0, I).

    Attributes:
        data_dim: Dimensionality of data (excluding batch).
    """

    def __init__(self, data_dim: int):
        self.data_dim = data_dim

    def alpha(self, t: Time) -> Scalar:
        return jnp.cos(jnp.pi / 2 * t)

    def alpha_prime(self, t: Time) -> Scalar:
        return -jnp.pi / 2 * jnp.sin(jnp.pi / 2 * t)

    def sigma(self, t: Time) -> Scalar:
        return jnp.sin(jnp.pi / 2 * t)

    def sigma_prime(self, t: Time) -> Scalar:
        return jnp.pi / 2 * jnp.cos(jnp.pi / 2 * t)

    def logsnr_inverse(self, lam: Scalar) -> Time:
        # logsnr = -2*log(tan(pi/2*t)) => t = 2/pi * arctan(exp(-lam/2))
        return 2.0 / jnp.pi * jnp.arctan(jnp.exp(-lam / 2.0))

    def _sample_from_source(self, key: PRNGKey, batch_size: int) -> ContinuousData:
        return jax.random.normal(key, (batch_size, self.data_dim))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROCESS_NAMES = ["flow_matching", "vp"]


def _make_process(name: str) -> InterpolationProcess:
    """Factory for test processes."""
    if name == "flow_matching":
        return FlowMatchingProcess(DATA_DIM)
    elif name == "vp":
        return VPProcess(DATA_DIM)
    else:
        raise ValueError(f"Unknown process: {name}")


def _reference_predictions(
    proc: InterpolationProcess,
    x_0: ContinuousData,
    z: ContinuousData,
    t: Time,
    s: Time | None = None,
) -> tuple[ContinuousData, dict[str, ContinuousData]]:
    """Compute all prediction kinds from known x_0 and z (reference impl).

    Args:
        proc: The interpolation process.
        x_0: Original data, shape (batch, *data_shape).
        z: Source noise, shape (batch, *data_shape).
        t: Time, shape (batch,).
        s: Optional intermediate time, shape (batch,).

    Returns:
        A tuple of (x_t, dict mapping kind -> reference value).
    """
    ndim = x_0.ndim
    alpha_t = bcast_right(proc.alpha(t), ndim)
    sigma_t = bcast_right(proc.sigma(t), ndim)
    alpha_prime_t = bcast_right(proc.alpha_prime(t), ndim)
    sigma_prime_t = bcast_right(proc.sigma_prime(t), ndim)

    x_t = alpha_t * x_0 + sigma_t * z
    v_t = alpha_prime_t * x_0 + sigma_prime_t * z

    preds: dict[str, ContinuousData] = {
        "x_0": x_0,
        "z": z,
        "v_t": v_t,
    }

    if s is not None:
        alpha_s = bcast_right(proc.alpha(s), ndim)
        sigma_s = bcast_right(proc.sigma(s), ndim)
        x_s = alpha_s * x_0 + sigma_s * z
        dt = bcast_right(t - s, ndim)
        v_st = (x_t - x_s) / dt
        preds["x_s"] = x_s
        preds["v_st"] = v_st

    return x_t, preds


# Conversion pairs that do NOT require s
BASIC_CONVERSION_PAIRS = [
    ("x_0", "z"),
    ("x_0", "v_t"),
    ("z", "x_0"),
    ("z", "v_t"),
    ("v_t", "x_0"),
    ("v_t", "z"),
]

# Conversion pairs that DO require s
S_CONVERSION_PAIRS = [
    ("x_0", "x_s"),
    ("x_0", "v_st"),
    ("z", "x_s"),
    ("z", "v_st"),
    ("v_t", "x_s"),
    ("v_t", "v_st"),
    ("x_s", "x_0"),
    ("x_s", "z"),
    ("x_s", "v_t"),
    ("x_s", "v_st"),
    ("v_st", "x_0"),
    ("v_st", "z"),
    ("v_st", "v_t"),
    ("v_st", "x_s"),
]

# Identity conversions
IDENTITY_KINDS = ["x_0", "z", "v_t", "x_s", "v_st"]


# ===========================================================================
# Tests for InterpolationProcess schedule functions
# ===========================================================================


class TestInterpolationSchedule:
    """Tests for schedule functions of InterpolationProcess."""

    def test_flow_matching_alpha_sigma_boundary(self):
        """Flow matching: alpha(0) = 1, alpha(1) = 0, sigma(0) = 0, sigma(1) = 1."""
        proc = FlowMatchingProcess(DATA_DIM)
        t0 = jnp.zeros((BATCH_SIZE,))
        t1 = jnp.ones((BATCH_SIZE,))

        assert jnp.allclose(proc.alpha(t0), jnp.ones((BATCH_SIZE,)))
        assert jnp.allclose(proc.alpha(t1), jnp.zeros((BATCH_SIZE,)))
        assert jnp.allclose(proc.sigma(t0), jnp.zeros((BATCH_SIZE,)))
        assert jnp.allclose(proc.sigma(t1), jnp.ones((BATCH_SIZE,)))

    def test_flow_matching_derivatives_are_constant(self):
        """Flow matching derivatives: alpha'(t) = -1, sigma'(t) = 1 for all t."""
        proc = FlowMatchingProcess(DATA_DIM)
        t = jnp.array([0.1, 0.3, 0.5, 0.9])

        assert jnp.allclose(proc.alpha_prime(t), -jnp.ones(4))
        assert jnp.allclose(proc.sigma_prime(t), jnp.ones(4))

    def test_vp_alpha_sigma_boundary(self):
        """VP: alpha(0) = 1, alpha(1) ~ 0, sigma(0) ~ 0, sigma(1) = 1."""
        proc = VPProcess(DATA_DIM)
        t0 = jnp.zeros((BATCH_SIZE,))
        t1 = jnp.ones((BATCH_SIZE,))

        assert jnp.allclose(proc.alpha(t0), jnp.ones((BATCH_SIZE,)), atol=1e-6)
        assert jnp.allclose(proc.alpha(t1), jnp.zeros((BATCH_SIZE,)), atol=1e-6)
        assert jnp.allclose(proc.sigma(t0), jnp.zeros((BATCH_SIZE,)), atol=1e-6)
        assert jnp.allclose(proc.sigma(t1), jnp.ones((BATCH_SIZE,)), atol=1e-6)

    def test_vp_midpoint_values(self):
        """VP: alpha(0.5) = cos(pi/4) ~ sqrt(2)/2, sigma(0.5) = sin(pi/4) ~ sqrt(2)/2."""
        proc = VPProcess(DATA_DIM)
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        sqrt2_over_2 = jnp.sqrt(2.0) / 2.0

        assert jnp.allclose(proc.alpha(t), sqrt2_over_2, atol=1e-6)
        assert jnp.allclose(proc.sigma(t), sqrt2_over_2, atol=1e-6)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_logsnr_matches_definition(self, proc_name: str):
        """logsnr(t) = 2 * (log alpha(t) - log sigma(t))."""
        proc = _make_process(proc_name)
        t = jnp.array([0.1, 0.3, 0.5, 0.7])

        expected = 2.0 * (jnp.log(proc.alpha(t)) - jnp.log(proc.sigma(t)))
        assert jnp.allclose(proc.logsnr(t), expected, atol=1e-6)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_logsnr_inverse_roundtrip(self, proc_name: str):
        """logsnr_inverse(logsnr(t)) ~ t."""
        proc = _make_process(proc_name)
        t = jnp.array([0.1, 0.3, 0.5, 0.7])

        lam = proc.logsnr(t)
        t_recovered = proc.logsnr_inverse(lam)
        assert jnp.allclose(t_recovered, t, atol=1e-5)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_logsnr_monotonically_decreasing(self, proc_name: str):
        """logsnr should decrease as t increases."""
        proc = _make_process(proc_name)
        t = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])

        logsnr_vals = proc.logsnr(t)
        diffs = jnp.diff(logsnr_vals)
        assert jnp.all(diffs < 0)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_get_corruption_coeffs_keys_and_values(self, proc_name: str):
        """get_schedule_info returns all expected keys with matching values."""
        proc = _make_process(proc_name)
        t = jnp.array([0.2, 0.4, 0.6, 0.8])

        coeffs = proc.get_coefficients(t)

        assert set(coeffs.keys()) == {
            "alpha",
            "alpha_prime",
            "sigma",
            "sigma_prime",
            "logsnr",
        }
        assert jnp.allclose(coeffs["alpha"], proc.alpha(t))
        assert jnp.allclose(coeffs["alpha_prime"], proc.alpha_prime(t))
        assert jnp.allclose(coeffs["sigma"], proc.sigma(t))
        assert jnp.allclose(coeffs["sigma_prime"], proc.sigma_prime(t))
        assert jnp.allclose(coeffs["logsnr"], proc.logsnr(t))


# ===========================================================================
# Tests for InterpolationProcess forward pass
# ===========================================================================


class TestInterpolationForward:
    """Tests for forward pass of InterpolationProcess."""

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_output_shapes(self, proc_name: str):
        """forward returns (x_t, aux) with correct shapes."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))

        x_t, aux = proc.forward(key, x_0, t)

        assert x_t.shape == (BATCH_SIZE, DATA_DIM)
        assert aux["x"].shape == (BATCH_SIZE, DATA_DIM)
        assert aux["z"].shape == (BATCH_SIZE, DATA_DIM)
        assert aux["t"].shape == (BATCH_SIZE,)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_aux_contains_schedule_info(self, proc_name: str):
        """forward aux dict includes schedule info keys."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))

        _, aux = proc.forward(key, x_0, t)

        for k in ("alpha", "alpha_prime", "sigma", "sigma_prime", "logsnr"):
            assert k in aux

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_forward_correctness(self, proc_name: str):
        """forward computes x_t = alpha(t)*x_0 + sigma(t)*z correctly."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(42)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))

        x_t, aux = proc.forward(key, x_0, t)
        z = aux["z"]

        alpha_t = bcast_right(proc.alpha(t), x_0.ndim)
        sigma_t = bcast_right(proc.sigma(t), x_0.ndim)
        expected_x_t = alpha_t * x_0 + sigma_t * z
        assert jnp.allclose(x_t, expected_x_t, atol=1e-6)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_near_zero_is_clean(self, proc_name: str):
        """At t ~ 0, x_t should be close to x_0 (alpha(0)=1, sigma(0)=0)."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 1e-6 * jnp.ones((BATCH_SIZE,))

        x_t, _ = proc.forward(key, x_0, t)
        assert jnp.allclose(x_t, x_0, atol=1e-3)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_deterministic_given_key(self, proc_name: str):
        """Calling forward with the same key produces the same result."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(123)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        x_t_a, aux_a = proc.forward(key, x_0, t)
        x_t_b, aux_b = proc.forward(key, x_0, t)

        assert jnp.allclose(x_t_a, x_t_b)
        assert jnp.allclose(aux_a["z"], aux_b["z"])

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_forward_multiple_count_and_shapes(self, proc_name: str):
        """forward_multiple returns correct number of (x_t, aux) pairs."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]

        results = proc.forward_multiple(key, x_0, ts)

        assert len(results) == 3
        for x_t, aux in results:
            assert x_t.shape == (BATCH_SIZE, DATA_DIM)
            assert aux["z"].shape == (BATCH_SIZE, DATA_DIM)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_forward_multiple_shared_noise(self, proc_name: str):
        """forward_multiple uses the same z for all times."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]

        results = proc.forward_multiple(key, x_0, ts)
        z_0 = results[0][1]["z"]
        for _, aux in results[1:]:
            assert jnp.allclose(aux["z"], z_0)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_forward_multiple_correctness(self, proc_name: str):
        """Each x_t from forward_multiple matches the interpolation formula."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]

        results = proc.forward_multiple(key, x_0, ts)
        z = results[0][1]["z"]

        for (x_t, _), t_val in zip(results, ts):
            alpha_t = bcast_right(proc.alpha(t_val), x_0.ndim)
            sigma_t = bcast_right(proc.sigma(t_val), x_0.ndim)
            expected = alpha_t * x_0 + sigma_t * z
            assert jnp.allclose(x_t, expected, atol=1e-6)


# ===========================================================================
# Tests for InterpolationProcess sample_from_terminal
# ===========================================================================


class TestInterpolationTerminal:
    """Tests for sample_from_terminal of InterpolationProcess."""

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_shape(self, proc_name: str):
        """sample_from_terminal returns (x_1, aux) with correct shapes."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)

        x_1, aux = proc.sample_from_terminal(key, BATCH_SIZE)

        assert x_1.shape == (BATCH_SIZE, DATA_DIM)
        assert aux["t"].shape == (BATCH_SIZE,)
        assert aux["z"].shape == (BATCH_SIZE, DATA_DIM)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_time_is_one(self, proc_name: str):
        """Aux time from sample_from_terminal should be 1.0."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)

        _, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        assert jnp.allclose(aux["t"], jnp.ones((BATCH_SIZE,)))

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_is_scaled_noise(self, proc_name: str):
        """At t=1, x_1 = sigma(1) * z (since alpha(1) = 0 for both schedules)."""
        proc = _make_process(proc_name)
        key = jax.random.PRNGKey(0)

        x_1, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        z = aux["z"]
        sigma_1 = proc.sigma(jnp.ones((BATCH_SIZE,)))
        expected = bcast_right(sigma_1, z.ndim) * z
        assert jnp.allclose(x_1, expected, atol=1e-6)


# ===========================================================================
# Tests for InterpolationProcess convert_prediction
# ===========================================================================


class TestInterpolationConversion:
    """Tests for convert_prediction of InterpolationProcess."""

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    @pytest.mark.parametrize("kind", IDENTITY_KINDS)
    def test_identity_conversion(self, proc_name: str, kind: str):
        """Converting a prediction to its own kind returns the same value."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))
        s = 0.1 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t, s)
        prediction = Prediction(value=preds[kind], kind=kind)
        result = proc.convert_prediction(prediction, x_t, s, t, kind)

        assert result.kind == kind
        assert jnp.allclose(result.value, preds[kind], atol=1e-5)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    @pytest.mark.parametrize("from_kind,to_kind", BASIC_CONVERSION_PAIRS)
    def test_basic_conversion(self, proc_name: str, from_kind: str, to_kind: str):
        """Test conversions among x_0, z, v (no intermediate time s)."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t)
        prediction = Prediction(value=preds[from_kind], kind=from_kind)
        result = proc.convert_prediction(prediction, x_t, None, t, to_kind)

        assert result.kind == to_kind
        assert jnp.allclose(result.value, preds[to_kind], atol=1e-5)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    @pytest.mark.parametrize("from_kind,to_kind", S_CONVERSION_PAIRS)
    def test_s_conversion(self, proc_name: str, from_kind: str, to_kind: str):
        """Test conversions involving x_s or v_st (requires intermediate time s)."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t, s)
        prediction = Prediction(value=preds[from_kind], kind=from_kind)
        result = proc.convert_prediction(prediction, x_t, s, t, to_kind)

        assert result.kind == to_kind
        assert jnp.allclose(result.value, preds[to_kind], atol=1e-4)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_roundtrip_x0_z_x0(self, proc_name: str):
        """x_0 -> z -> x_0 round-trip returns the original x_0."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t)
        pred_x0 = Prediction(value=preds["x_0"], kind="x_0")
        pred_z = proc.convert_prediction(pred_x0, x_t, None, t, "z")
        pred_x0_back = proc.convert_prediction(pred_z, x_t, None, t, "x_0")

        assert jnp.allclose(pred_x0_back.value, x_0, atol=1e-5)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_roundtrip_x0_v_x0(self, proc_name: str):
        """x_0 -> v -> x_0 round-trip returns the original x_0."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t)
        pred_x0 = Prediction(value=preds["x_0"], kind="x_0")
        pred_v = proc.convert_prediction(pred_x0, x_t, None, t, "v_t")
        pred_x0_back = proc.convert_prediction(pred_v, x_t, None, t, "x_0")

        assert jnp.allclose(pred_x0_back.value, x_0, atol=1e-5)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_roundtrip_z_v_z(self, proc_name: str):
        """z -> v -> z round-trip returns the original z."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t)
        pred_z = Prediction(value=preds["z"], kind="z")
        pred_v = proc.convert_prediction(pred_z, x_t, None, t, "v_t")
        pred_z_back = proc.convert_prediction(pred_v, x_t, None, t, "z")

        assert jnp.allclose(pred_z_back.value, z, atol=1e-5)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_roundtrip_x0_xs_x0(self, proc_name: str):
        """x_0 -> x_s -> x_0 round-trip returns the original x_0."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t, s)
        pred_x0 = Prediction(value=preds["x_0"], kind="x_0")
        pred_xs = proc.convert_prediction(pred_x0, x_t, s, t, "x_s")
        pred_x0_back = proc.convert_prediction(pred_xs, x_t, s, t, "x_0")

        assert jnp.allclose(pred_x0_back.value, x_0, atol=1e-4)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_roundtrip_x0_vst_x0(self, proc_name: str):
        """x_0 -> v_st -> x_0 round-trip returns the original x_0."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t, s)
        pred_x0 = Prediction(value=preds["x_0"], kind="x_0")
        pred_vst = proc.convert_prediction(pred_x0, x_t, s, t, "v_st")
        pred_x0_back = proc.convert_prediction(pred_vst, x_t, s, t, "x_0")

        assert jnp.allclose(pred_x0_back.value, x_0, atol=1e-4)

    @pytest.mark.parametrize("proc_name", PROCESS_NAMES)
    def test_unknown_conversion_raises(self, proc_name: str):
        """Converting to an unknown kind raises NotImplementedError."""
        proc = _make_process(proc_name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        z = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))

        x_t, preds = _reference_predictions(proc, x_0, z, t)
        pred_x0 = Prediction(value=preds["x_0"], kind="x_0")
        with pytest.raises(NotImplementedError):
            proc.convert_prediction(pred_x0, x_t, None, t, "unknown_kind")
