"""Tests for GaussianDiffusion (diffusionlab.processes.gaussian)."""

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
from diffusionlab.typing import ContinuousData, Prediction, Time
from diffusionlab.utils.tree_ops import bcast_right

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
DATA_DIM = 8
GAUSSIAN_DATA_SHAPE = (DATA_DIM,)

# Process names for parametrised tests
GAUSSIAN_PROCESS_NAMES = [
    "vpcosine",
    "vplinear",
    "ve",
    "edm",
    "flow_matching_g",
]

# VP-type: alpha(0) = 1, sigma(0) = 0, alpha(1) ~ 0, sigma(1) = 1
VP_PROCESS_NAMES = ["vpcosine", "vplinear", "flow_matching_g"]

# Strict VP: alpha^2 + sigma^2 = 1 for all t
STRICT_VP_NAMES = ["vpcosine", "vplinear"]

# VE-type: alpha(t) = 1 for all t
ALPHA_ONE_NAMES = ["ve", "edm"]


# ---------------------------------------------------------------------------
# Factories and helpers
# ---------------------------------------------------------------------------


def _make_gaussian(name: str) -> GaussianDiffusionProcess:
    """Factory for concrete Gaussian process instances."""
    if name == "vpcosine":
        return VPCosine(GAUSSIAN_DATA_SHAPE)
    elif name == "vplinear":
        return VPLinear(GAUSSIAN_DATA_SHAPE)
    elif name == "ve":
        return VE(GAUSSIAN_DATA_SHAPE)
    elif name == "edm":
        return EDM(GAUSSIAN_DATA_SHAPE)
    elif name == "flow_matching_g":
        return FlowMatching(GAUSSIAN_DATA_SHAPE)
    else:
        raise ValueError(f"Unknown Gaussian process: {name}")


def _gaussian_ref_predictions(
    proc: GaussianDiffusionProcess,
    x_0: ContinuousData,
    eps: ContinuousData,
    t: Time,
    s: Time | None = None,
) -> tuple[ContinuousData, dict[str, ContinuousData]]:
    """Reference predictions for Gaussian processes, including eps and score.

    Args:
        proc: A GaussianDiffusion instance.
        x_0: Original data, shape (batch, *data_shape).
        eps: Gaussian noise, shape (batch, *data_shape).
        t: Time, shape (batch,).
        s: Optional intermediate time, shape (batch,).

    Returns:
        Tuple of (x_t, dict mapping kind -> reference value).
    """
    ndim = x_0.ndim
    alpha_t = bcast_right(proc.alpha(t), ndim)
    sigma_t = bcast_right(proc.sigma(t), ndim)
    alpha_prime_t = bcast_right(proc.alpha_prime(t), ndim)
    sigma_prime_t = bcast_right(proc.sigma_prime(t), ndim)

    x_t = alpha_t * x_0 + sigma_t * eps
    v_t = alpha_prime_t * x_0 + sigma_prime_t * eps

    preds: dict[str, ContinuousData] = {
        "x_0": x_0,
        "eps": eps,
        "v_t": v_t,
        "score": -eps / sigma_t,
    }

    if s is not None:
        alpha_s = bcast_right(proc.alpha(s), ndim)
        sigma_s = bcast_right(proc.sigma(s), ndim)
        x_s = alpha_s * x_0 + sigma_s * eps
        dt = bcast_right(t - s, ndim)
        preds["x_s"] = x_s
        preds["v_st"] = (x_t - x_s) / dt

    return x_t, preds


# Gaussian conversion pairs (no s required)
GAUSS_BASIC_PAIRS = [
    ("x_0", "eps"),
    ("x_0", "score"),
    ("x_0", "v_t"),
    ("eps", "x_0"),
    ("eps", "score"),
    ("eps", "v_t"),
    ("score", "x_0"),
    ("score", "eps"),
    ("score", "v_t"),
    ("v_t", "x_0"),
    ("v_t", "eps"),
    ("v_t", "score"),
]

# Gaussian conversion pairs requiring s
GAUSS_S_PAIRS = [
    ("x_0", "x_s"),
    ("x_0", "v_st"),
    ("eps", "x_s"),
    ("eps", "v_st"),
    ("score", "x_s"),
    ("score", "v_st"),
    ("x_s", "x_0"),
    ("x_s", "eps"),
    ("x_s", "score"),
    ("x_s", "v_t"),
    ("x_s", "v_st"),
    ("v_st", "x_0"),
    ("v_st", "eps"),
    ("v_st", "score"),
    ("v_st", "v_t"),
    ("v_st", "x_s"),
    ("v_t", "x_s"),
    ("v_t", "v_st"),
]

# All identity kinds for Gaussian
GAUSS_IDENTITY_KINDS = ["x_0", "eps", "v_t", "score", "x_s", "v_st"]


# ===========================================================================
# Tests for Gaussian schedule functions
# ===========================================================================


class TestGaussianSchedule:
    """Tests for schedule functions of concrete Gaussian diffusion processes."""

    @pytest.mark.parametrize("name", VP_PROCESS_NAMES)
    def test_boundary_alpha_one_sigma_zero(self, name: str):
        """alpha(0) = 1, sigma(0) = 0 for VP-type processes."""
        proc = _make_gaussian(name)
        t0 = jnp.zeros((BATCH_SIZE,))
        assert jnp.allclose(proc.alpha(t0), 1.0, atol=1e-6)
        assert jnp.allclose(proc.sigma(t0), 0.0, atol=1e-6)

    @pytest.mark.parametrize("name", VP_PROCESS_NAMES)
    def test_boundary_alpha_zero_sigma_one(self, name: str):
        """alpha(1) ~ 0, sigma(1) ~ 1 for VP-type processes."""
        proc = _make_gaussian(name)
        t1 = jnp.ones((BATCH_SIZE,))
        # VPLinear with default betas has alpha(1) ~ 0.007, so relaxed atol
        assert jnp.allclose(proc.alpha(t1), 0.0, atol=1e-1)
        assert jnp.allclose(proc.sigma(t1), 1.0, atol=1e-2)

    @pytest.mark.parametrize("name", STRICT_VP_NAMES)
    def test_vp_property_alpha_sq_plus_sigma_sq(self, name: str):
        """alpha(t)^2 + sigma(t)^2 = 1 for variance-preserving processes."""
        proc = _make_gaussian(name)
        t = jnp.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
        assert jnp.allclose(proc.alpha(t) ** 2 + proc.sigma(t) ** 2, 1.0, atol=1e-6)

    @pytest.mark.parametrize("name", ALPHA_ONE_NAMES)
    def test_alpha_is_constant_one(self, name: str):
        """alpha(t) = 1 for all t in VE-type processes."""
        proc = _make_gaussian(name)
        t = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])
        assert jnp.allclose(proc.alpha(t), 1.0)

    @pytest.mark.parametrize("name", ALPHA_ONE_NAMES)
    def test_alpha_prime_is_zero(self, name: str):
        """alpha'(t) = 0 for VE-type processes."""
        proc = _make_gaussian(name)
        t = jnp.array([0.0, 0.2, 0.5, 0.8, 1.0])
        assert jnp.allclose(proc.alpha_prime(t), 0.0)

    def test_ve_sigma_boundaries(self):
        """VE: sigma(0) = sigma_min, sigma(1) = sigma_max."""
        proc = VE(GAUSSIAN_DATA_SHAPE, sigma_min=0.01, sigma_max=50.0)
        assert jnp.allclose(proc.sigma(jnp.zeros(1)), 0.01, atol=1e-6)
        assert jnp.allclose(proc.sigma(jnp.ones(1)), 50.0, atol=1e-3)

    def test_edm_sigma_boundaries(self):
        """EDM: sigma(0) = sigma_min, sigma(1) = sigma_max."""
        proc = EDM(GAUSSIAN_DATA_SHAPE, sigma_min=0.002, sigma_max=80.0)
        assert jnp.allclose(proc.sigma(jnp.zeros(1)), 0.002, atol=1e-6)
        assert jnp.allclose(proc.sigma(jnp.ones(1)), 80.0, atol=1e-2)

    def test_ve_sigma_is_geometric(self):
        """VE: sigma(t) = sigma_min * (sigma_max/sigma_min)^t."""
        proc = VE(GAUSSIAN_DATA_SHAPE, sigma_min=0.01, sigma_max=50.0)
        t = jnp.array([0.25, 0.5, 0.75])
        expected = 0.01 * (50.0 / 0.01) ** t
        assert jnp.allclose(proc.sigma(t), expected, atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_derivatives_match_jax_grad(self, name: str):
        """alpha'(t) and sigma'(t) match jax.grad of the scalar functions."""
        proc = _make_gaussian(name)
        t = jnp.array([0.2, 0.4, 0.6, 0.8])
        alpha_grad_fn = jax.vmap(jax.grad(lambda ti: proc.alpha(ti[None])[0]))
        sigma_grad_fn = jax.vmap(jax.grad(lambda ti: proc.sigma(ti[None])[0]))
        alpha_prime_ref = alpha_grad_fn(t)
        sigma_prime_ref = sigma_grad_fn(t)
        assert jnp.allclose(proc.alpha_prime(t), alpha_prime_ref, atol=1e-5)
        assert jnp.allclose(proc.sigma_prime(t), sigma_prime_ref, atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_logsnr_inverse_roundtrip(self, name: str):
        """logsnr_inverse(logsnr(t)) ~ t."""
        proc = _make_gaussian(name)
        t = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        recovered = proc.logsnr_inverse(proc.logsnr(t))
        assert jnp.allclose(recovered, t, atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_logsnr_monotonically_decreasing(self, name: str):
        """logsnr should decrease as t increases."""
        proc = _make_gaussian(name)
        t = jnp.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
        diffs = jnp.diff(proc.logsnr(t))
        assert jnp.all(diffs < 0)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_get_coefficients_keys_and_values(self, name: str):
        """get_coefficients matches individual method calls."""
        proc = _make_gaussian(name)
        t = jnp.array([0.2, 0.5, 0.8])
        coeffs = proc.get_coefficients(t)
        assert jnp.allclose(coeffs["alpha"], proc.alpha(t))
        assert jnp.allclose(coeffs["sigma"], proc.sigma(t))
        assert jnp.allclose(coeffs["alpha_prime"], proc.alpha_prime(t))
        assert jnp.allclose(coeffs["sigma_prime"], proc.sigma_prime(t))
        assert jnp.allclose(coeffs["logsnr"], proc.logsnr(t))

    def test_vplinear_custom_betas(self):
        """VPLinear with non-default betas produces different schedule."""
        a = VPLinear(GAUSSIAN_DATA_SHAPE, beta_min=0.1, beta_max=20.0)
        b = VPLinear(GAUSSIAN_DATA_SHAPE, beta_min=1.0, beta_max=30.0)
        t = jnp.array([0.5])
        assert not jnp.allclose(a.sigma(t), b.sigma(t))

    def test_edm_custom_rho(self):
        """EDM with different rho gives different schedule."""
        a = EDM(GAUSSIAN_DATA_SHAPE, rho=7.0)
        b = EDM(GAUSSIAN_DATA_SHAPE, rho=3.0)
        t = jnp.array([0.5])
        assert not jnp.allclose(a.sigma(t), b.sigma(t))

    def test_vpcosine_midpoint(self):
        """VPCosine: alpha(0.5) = cos(pi/4) = sigma(0.5) = sin(pi/4) = sqrt(2)/2."""
        proc = VPCosine(GAUSSIAN_DATA_SHAPE)
        t = 0.5 * jnp.ones(1)
        sqrt2_over_2 = jnp.sqrt(2.0) / 2.0
        assert jnp.allclose(proc.alpha(t), sqrt2_over_2, atol=1e-6)
        assert jnp.allclose(proc.sigma(t), sqrt2_over_2, atol=1e-6)


# ===========================================================================
# Tests for GaussianDiffusion forward pass
# ===========================================================================


class TestGaussianForward:
    """Tests for forward pass of GaussianDiffusion subclasses."""

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_output_shapes(self, name: str):
        """Forward output shapes are correct."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))
        x_t, aux = proc.forward(key, x_0, t)
        assert x_t.shape == (BATCH_SIZE, DATA_DIM)
        assert aux["eps"].shape == (BATCH_SIZE, DATA_DIM)
        assert aux["x"].shape == (BATCH_SIZE, DATA_DIM)
        assert aux["t"].shape == (BATCH_SIZE,)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_forward_has_eps_not_z(self, name: str):
        """GaussianDiffusion forward returns 'eps' in aux, not 'z'."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))
        _, aux = proc.forward(key, x_0, t)
        assert "eps" in aux
        assert "z" not in aux

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_forward_correctness(self, name: str):
        """x_t = alpha(t) x_0 + sigma(t) eps."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(42)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))
        x_t, aux = proc.forward(key, x_0, t)
        eps = aux["eps"]
        alpha_t = bcast_right(proc.alpha(t), x_0.ndim)
        sigma_t = bcast_right(proc.sigma(t), x_0.ndim)
        expected = alpha_t * x_0 + sigma_t * eps
        assert jnp.allclose(x_t, expected, atol=1e-6)

    @pytest.mark.parametrize("name", VP_PROCESS_NAMES)
    def test_near_zero_is_clean(self, name: str):
        """At t ~ 0, x_t ~ x_0 for VP-type processes."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 1e-6 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)
        assert jnp.allclose(x_t, x_0, atol=1e-3)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_deterministic_given_key(self, name: str):
        """Same PRNG key produces same result."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(123)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t_a, aux_a = proc.forward(key, x_0, t)
        x_t_b, aux_b = proc.forward(key, x_0, t)
        assert jnp.allclose(x_t_a, x_t_b)
        assert jnp.allclose(aux_a["eps"], aux_b["eps"])

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_forward_multiple_count_and_shapes(self, name: str):
        """forward_multiple returns correct count and shapes, with 'eps' in aux."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]
        results = proc.forward_multiple(key, x_0, ts)
        assert len(results) == 3
        for x_t, aux in results:
            assert "eps" in aux
            assert "z" not in aux
            assert x_t.shape == (BATCH_SIZE, DATA_DIM)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_forward_multiple_shared_noise(self, name: str):
        """forward_multiple uses the same eps for all times."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]
        results = proc.forward_multiple(key, x_0, ts)
        eps_0 = results[0][1]["eps"]
        for _, aux in results[1:]:
            assert jnp.allclose(aux["eps"], eps_0)


# ===========================================================================
# Tests for GaussianDiffusion sample_from_terminal
# ===========================================================================


class TestGaussianTerminal:
    """Tests for sample_from_terminal of GaussianDiffusion."""

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_shape(self, name: str):
        """sample_from_terminal returns correct shapes with 'eps' in aux."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(0)
        x_1, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        assert "eps" in aux
        assert "z" not in aux
        assert x_1.shape == (BATCH_SIZE, DATA_DIM)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_time_is_one(self, name: str):
        """Terminal sample aux time should be 1."""
        proc = _make_gaussian(name)
        key = jax.random.PRNGKey(0)
        _, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        assert jnp.allclose(aux["t"], 1.0)


# ===========================================================================
# Tests for GaussianDiffusion convert_prediction
# ===========================================================================


class TestGaussianConversion:
    """Tests for convert_prediction with eps and score kinds."""

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    @pytest.mark.parametrize("kind", GAUSS_IDENTITY_KINDS)
    def test_identity_conversion(self, name: str, kind: str):
        """Converting a prediction to its own kind returns the same value."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))
        s = 0.1 * jnp.ones((BATCH_SIZE,))
        x_t, preds = _gaussian_ref_predictions(proc, x_0, eps, t, s)
        result = proc.convert_prediction(
            Prediction(value=preds[kind], kind=kind), x_t, s, t, kind
        )
        assert result.kind == kind
        assert jnp.allclose(result.value, preds[kind], atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    @pytest.mark.parametrize("from_kind,to_kind", GAUSS_BASIC_PAIRS)
    def test_basic_conversion(self, name: str, from_kind: str, to_kind: str):
        """Conversions among x_0, eps, score, v (no intermediate time s)."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))
        x_t, preds = _gaussian_ref_predictions(proc, x_0, eps, t)
        result = proc.convert_prediction(
            Prediction(value=preds[from_kind], kind=from_kind),
            x_t,
            None,
            t,
            to_kind,
        )
        assert result.kind == to_kind
        assert jnp.allclose(result.value, preds[to_kind], atol=1e-4)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    @pytest.mark.parametrize("from_kind,to_kind", GAUSS_S_PAIRS)
    def test_s_conversion(self, name: str, from_kind: str, to_kind: str):
        """Conversions involving x_s or v_st (requires intermediate time s)."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, preds = _gaussian_ref_predictions(proc, x_0, eps, t, s)
        result = proc.convert_prediction(
            Prediction(value=preds[from_kind], kind=from_kind),
            x_t,
            s,
            t,
            to_kind,
        )
        assert result.kind == to_kind
        assert jnp.allclose(result.value, preds[to_kind], atol=1e-4)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_roundtrip_x0_eps_x0(self, name: str):
        """x_0 -> eps -> x_0 roundtrip preserves original."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))
        x_t, _ = _gaussian_ref_predictions(proc, x_0, eps, t)
        pred = Prediction(value=x_0, kind="x_0")
        via_eps = proc.convert_prediction(pred, x_t, None, t, "eps")
        back = proc.convert_prediction(via_eps, x_t, None, t, "x_0")
        assert jnp.allclose(back.value, x_0, atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_roundtrip_x0_score_x0(self, name: str):
        """x_0 -> score -> x_0 roundtrip preserves original."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))
        x_t, _ = _gaussian_ref_predictions(proc, x_0, eps, t)
        pred = Prediction(value=x_0, kind="x_0")
        via_score = proc.convert_prediction(pred, x_t, None, t, "score")
        back = proc.convert_prediction(via_score, x_t, None, t, "x_0")
        assert jnp.allclose(back.value, x_0, atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_roundtrip_eps_score_eps(self, name: str):
        """eps -> score -> eps roundtrip preserves original."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))
        x_t, _ = _gaussian_ref_predictions(proc, x_0, eps, t)
        pred = Prediction(value=eps, kind="eps")
        via_score = proc.convert_prediction(pred, x_t, None, t, "score")
        back = proc.convert_prediction(via_score, x_t, None, t, "eps")
        assert jnp.allclose(back.value, eps, atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_score_matches_tweedie(self, name: str):
        """score = -eps / sigma(t) (Tweedie's formula)."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))
        x_t, _ = _gaussian_ref_predictions(proc, x_0, eps, t)
        sigma_t = bcast_right(proc.sigma(t), x_0.ndim)
        expected_score = -eps / sigma_t
        pred_eps = Prediction(value=eps, kind="eps")
        result = proc.convert_prediction(pred_eps, x_t, None, t, "score")
        assert jnp.allclose(result.value, expected_score, atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_score_to_x0_via_tweedie(self, name: str):
        """x_0 = (x_t + sigma^2 * score) / alpha from Tweedie's formula."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.4 * jnp.ones((BATCH_SIZE,))
        x_t, preds = _gaussian_ref_predictions(proc, x_0, eps, t)
        alpha_t = bcast_right(proc.alpha(t), x_0.ndim)
        sigma_t = bcast_right(proc.sigma(t), x_0.ndim)
        expected_x0 = (x_t + sigma_t**2 * preds["score"]) / alpha_t
        assert jnp.allclose(expected_x0, x_0, atol=1e-5)

    @pytest.mark.parametrize("name", GAUSSIAN_PROCESS_NAMES)
    def test_unknown_conversion_raises(self, name: str):
        """Converting to an unknown kind raises NotImplementedError."""
        proc = _make_gaussian(name)
        x_0 = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM))
        eps = jax.random.normal(jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM))
        t = 0.3 * jnp.ones((BATCH_SIZE,))
        x_t, _ = _gaussian_ref_predictions(proc, x_0, eps, t)
        pred = Prediction(value=x_0, kind="x_0")
        with pytest.raises(NotImplementedError):
            proc.convert_prediction(pred, x_t, None, t, "nonexistent_kind")
