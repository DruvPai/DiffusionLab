"""Tests for CFG (classifier-free guidance) transform."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.gaussian import FlowMatching, VPCosine
from diffusionlab.sampling.base import SamplerStep, SamplingTransform, sample
from diffusionlab.sampling.continuous import DDMSampler, EulerSampler
from diffusionlab.sampling.guidance import CFG
from diffusionlab.sampling.schedules import uniform_schedule
from diffusionlab.typing import Prediction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
DATA_SHAPE = (8,)
NUM_STEPS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cond_uncond_predictor(cond_val, uncond_val, kind="eps"):
    """Predictor returning cond_val when cond is not None, uncond_val otherwise."""

    def predict(x, s, t, cond, aux):
        if cond is None:
            return Prediction(value=uncond_val, kind=kind)
        return Prediction(value=cond_val, kind=kind)

    return predict


# ===========================================================================
# Tests for CFG construction
# ===========================================================================


class TestCFGConstruction:
    """Tests for CFG initialization with different guidance_scale types."""

    def test_float_scale(self):
        """Float guidance_scale is accepted."""
        cfg = CFG(guidance_scale=2.0)
        assert hasattr(cfg, "guidance_scale_fn")
        assert callable(cfg.guidance_scale_fn)

    def test_callable_scale(self):
        """Callable guidance_scale is accepted."""
        cfg = CFG(guidance_scale=lambda t: 2.0 * t)
        assert callable(cfg.guidance_scale_fn)

    def test_is_sampling_transform(self):
        """CFG is a SamplingTransform."""
        cfg = CFG(guidance_scale=1.0)
        assert isinstance(cfg, SamplingTransform)

    def test_float_scale_produces_correct_values(self):
        """Float scale wraps into a callable that broadcasts correctly."""
        cfg = CFG(guidance_scale=3.0)
        t = jnp.array([0.5, 0.5, 0.5, 0.5])
        result = cfg.guidance_scale_fn(t)
        assert jnp.allclose(result, 3.0)
        assert result.shape == (4,)

    def test_scalar_tree_constant_jax_array(self):
        """PyTree constant (JAX array) as guidance_scale is accepted and used."""
        scale = jnp.array(2.0)
        cfg = CFG(guidance_scale=scale)
        t = jnp.array([0.5, 0.5])
        result = cfg.guidance_scale_fn(t)
        assert jnp.allclose(result, 2.0)
        # Same effective behavior as float 2.0 in transform
        cond_val = jnp.array([[3.0, 4.0], [1.0, 2.0]])
        uncond_val = jnp.array([[1.0, 2.0], [1.0, 2.0]])
        predict = _make_cond_uncond_predictor(cond_val, uncond_val)
        step = SamplerStep(
            x=jnp.zeros((2, 2)),
            t=jnp.array([0.5, 0.5]),
            aux={},
        )
        pred = predict(step.x, None, step.t, "cond", step.aux)
        out = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4, 0.4]), "cond"
        )
        expected = uncond_val + 2.0 * (cond_val - uncond_val)
        assert jnp.allclose(out.value, expected)

    def test_scalar_tree_constant_dict(self):
        """PyTree constant (dict) as guidance_scale is accepted for multimodal."""
        scale_tree = {"image": jnp.array(1.5), "text": jnp.array(2.0)}
        cfg = CFG(guidance_scale=scale_tree)
        cond_img = jnp.array([[1.0, 2.0]])
        uncond_img = jnp.array([[0.0, 0.0]])
        cond_txt = jnp.array([[3.0]])
        uncond_txt = jnp.array([[1.0]])

        def predict(x, s, t, cond, aux):
            if cond is None:
                return {
                    "image": Prediction(value=uncond_img, kind="eps"),
                    "text": Prediction(value=uncond_txt, kind="v_t"),
                }
            return {
                "image": Prediction(value=cond_img, kind="eps"),
                "text": Prediction(value=cond_txt, kind="v_t"),
            }

        step = SamplerStep(
            x={"image": jnp.zeros((1, 2)), "text": jnp.zeros((1, 1))},
            t={"image": jnp.array([0.5]), "text": jnp.array([0.5])},
            aux={},
        )
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )
        expected_img = uncond_img + 1.5 * (cond_img - uncond_img)
        expected_txt = uncond_txt + 2.0 * (cond_txt - uncond_txt)
        assert jnp.allclose(result["image"].value, expected_img)
        assert jnp.allclose(result["text"].value, expected_txt)

    def test_callable_scale_evaluated_at_t(self):
        """Callable scale is evaluated at the given time."""
        cfg = CFG(guidance_scale=lambda t: 2 * t)
        t = jnp.array([0.5, 0.5])
        result = cfg.guidance_scale_fn(t)
        assert jnp.allclose(result, 1.0)


# ===========================================================================
# Tests for CFG transform_prediction
# ===========================================================================


class TestCFGTransformPrediction:
    """Tests for the CFG.transform_prediction method."""

    def test_linear_combination_formula(self):
        """Verify: result = pred_uncond + w * (pred_cond - pred_uncond)."""
        cond_val = jnp.array([[3.0, 4.0]])
        uncond_val = jnp.array([[1.0, 2.0]])
        w = 2.0

        predict = _make_cond_uncond_predictor(cond_val, uncond_val)
        cfg = CFG(guidance_scale=w)

        step = SamplerStep(x=jnp.zeros((1, 2)), t=jnp.array([0.5]), aux={})
        pred_cond = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0),
            predict,
            pred_cond,
            step,
            jnp.array([0.4]),
            "cond",
        )

        expected = uncond_val + w * (cond_val - uncond_val)
        assert jnp.allclose(result.value, expected)

    def test_preserves_prediction_kind(self):
        """Output prediction has same kind as input."""
        cond_val = jnp.array([[1.0]])
        uncond_val = jnp.array([[0.0]])
        predict = _make_cond_uncond_predictor(cond_val, uncond_val, kind="v_t")
        cfg = CFG(guidance_scale=2.0)

        step = SamplerStep(x=jnp.zeros((1, 1)), t=jnp.array([0.5]), aux={})
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )
        assert result.kind == "v_t"

    def test_scale_one_is_conditional(self):
        """CFG with w=1 returns the conditional prediction."""
        cond_val = jnp.array([[5.0, 6.0]])
        uncond_val = jnp.array([[1.0, 2.0]])
        predict = _make_cond_uncond_predictor(cond_val, uncond_val)
        cfg = CFG(guidance_scale=1.0)

        step = SamplerStep(x=jnp.zeros((1, 2)), t=jnp.array([0.5]), aux={})
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )

        # w=1: uncond + 1*(cond - uncond) = cond
        assert jnp.allclose(result.value, cond_val)

    def test_scale_zero_is_unconditional(self):
        """CFG with w=0 returns the unconditional prediction."""
        cond_val = jnp.array([[5.0, 6.0]])
        uncond_val = jnp.array([[1.0, 2.0]])
        predict = _make_cond_uncond_predictor(cond_val, uncond_val)
        cfg = CFG(guidance_scale=0.0)

        step = SamplerStep(x=jnp.zeros((1, 2)), t=jnp.array([0.5]), aux={})
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )

        # w=0: uncond + 0*(cond - uncond) = uncond
        assert jnp.allclose(result.value, uncond_val)

    @pytest.mark.parametrize("w", [0.5, 1.0, 2.0, 5.0, 10.0])
    def test_various_scales(self, w: float):
        """CFG formula works for various guidance weights."""
        cond_val = jnp.array([[3.0, 4.0]])
        uncond_val = jnp.array([[1.0, 2.0]])
        predict = _make_cond_uncond_predictor(cond_val, uncond_val)
        cfg = CFG(guidance_scale=w)

        step = SamplerStep(x=jnp.zeros((1, 2)), t=jnp.array([0.5]), aux={})
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )

        expected = uncond_val + w * (cond_val - uncond_val)
        assert jnp.allclose(result.value, expected, atol=1e-5)

    def test_negative_scale(self):
        """Negative guidance scale (anti-guidance) works."""
        cond_val = jnp.array([[3.0]])
        uncond_val = jnp.array([[1.0]])
        w = -1.0
        predict = _make_cond_uncond_predictor(cond_val, uncond_val)
        cfg = CFG(guidance_scale=w)

        step = SamplerStep(x=jnp.zeros((1, 1)), t=jnp.array([0.5]), aux={})
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )

        expected = uncond_val + w * (cond_val - uncond_val)
        assert jnp.allclose(result.value, expected)

    def test_batched_data(self):
        """CFG works with batched data."""
        cond_val = jnp.array([[3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        uncond_val = jnp.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
        w = 2.0
        predict = _make_cond_uncond_predictor(cond_val, uncond_val)
        cfg = CFG(guidance_scale=w)

        step = SamplerStep(
            x=jnp.zeros((4, 2)),
            t=jnp.array([0.5, 0.5, 0.5, 0.5]),
            aux={},
        )
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4] * 4), "cond"
        )

        expected = uncond_val + w * (cond_val - uncond_val)
        assert jnp.allclose(result.value, expected)
        assert result.value.shape == (4, 2)

    def test_higher_dimensional_data(self):
        """CFG works with higher-dimensional data (e.g. images)."""
        cond_val = jnp.ones((2, 3, 4, 4))
        uncond_val = jnp.zeros((2, 3, 4, 4))
        w = 3.0
        predict = _make_cond_uncond_predictor(cond_val, uncond_val)
        cfg = CFG(guidance_scale=w)

        step = SamplerStep(
            x=jnp.zeros((2, 3, 4, 4)),
            t=jnp.array([0.5, 0.5]),
            aux={},
        )
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0),
            predict,
            pred,
            step,
            jnp.array([0.4, 0.4]),
            "cond",
        )

        expected = uncond_val + w * (cond_val - uncond_val)
        assert jnp.allclose(result.value, expected)
        assert result.value.shape == (2, 3, 4, 4)


# ===========================================================================
# Tests for CFG with callable guidance scale
# ===========================================================================


class TestCFGCallableScale:
    """Tests for time-dependent guidance scale."""

    def test_callable_scale_varies_with_t(self):
        """Callable scale evaluated at current time t."""
        cond_val = jnp.array([[4.0]])
        uncond_val = jnp.array([[0.0]])

        predict = _make_cond_uncond_predictor(cond_val, uncond_val)

        # scale = 2*t, so at t=0.5 => scale=1.0
        cfg = CFG(guidance_scale=lambda t: 2 * t)
        step = SamplerStep(x=jnp.zeros((1, 1)), t=jnp.array([0.5]), aux={})
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )

        # w=1.0: uncond + 1*(cond - uncond) = cond = 4.0
        assert jnp.allclose(result.value, cond_val, atol=1e-5)

    def test_callable_at_different_times(self):
        """Different t values produce different effective scales."""
        cond_val = jnp.array([[4.0]])
        uncond_val = jnp.array([[0.0]])
        predict = _make_cond_uncond_predictor(cond_val, uncond_val)

        cfg = CFG(guidance_scale=lambda t: 2 * t)

        # t=0.25 => w=0.5
        step1 = SamplerStep(x=jnp.zeros((1, 1)), t=jnp.array([0.25]), aux={})
        pred1 = predict(step1.x, None, step1.t, "cond", step1.aux)
        result1 = cfg.transform_prediction(
            jax.random.key(0), predict, pred1, step1, jnp.array([0.2]), "cond"
        )

        # t=0.75 => w=1.5
        step2 = SamplerStep(x=jnp.zeros((1, 1)), t=jnp.array([0.75]), aux={})
        pred2 = predict(step2.x, None, step2.t, "cond", step2.aux)
        result2 = cfg.transform_prediction(
            jax.random.key(0), predict, pred2, step2, jnp.array([0.6]), "cond"
        )

        expected1 = uncond_val + 0.5 * (cond_val - uncond_val)
        expected2 = uncond_val + 1.5 * (cond_val - uncond_val)

        assert jnp.allclose(result1.value, expected1, atol=1e-5)
        assert jnp.allclose(result2.value, expected2, atol=1e-5)

    def test_constant_callable_matches_float(self):
        """A constant callable produces same result as the equivalent float."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=x * 0.1, kind="eps")

        out_float = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=2.0)],
        )
        out_callable = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[
                CFG(
                    guidance_scale=lambda t: jnp.broadcast_to(jnp.asarray(2.0), t.shape)
                )
            ],
        )
        assert jnp.allclose(out_float, out_callable, atol=1e-5)


# ===========================================================================
# Tests for CFG with multimodal (dict-of-Predictions)
# ===========================================================================


class TestCFGMultimodal:
    """Tests for CFG with dictionary-structured predictions (multimodal)."""

    def test_multimodal_linear_combination(self):
        """CFG applies the linear combination per modality."""
        cond_img = jnp.array([[1.0, 2.0]])
        uncond_img = jnp.array([[0.0, 0.0]])
        cond_txt = jnp.array([[3.0]])
        uncond_txt = jnp.array([[1.0]])
        w = 3.0

        def predict(x, s, t, cond, aux):
            if cond is None:
                return {
                    "image": Prediction(value=uncond_img, kind="eps"),
                    "text": Prediction(value=uncond_txt, kind="v_t"),
                }
            return {
                "image": Prediction(value=cond_img, kind="eps"),
                "text": Prediction(value=cond_txt, kind="v_t"),
            }

        cfg = CFG(guidance_scale=w)
        step = SamplerStep(
            x={"image": jnp.zeros((1, 2)), "text": jnp.zeros((1, 1))},
            t={"image": jnp.array([0.5]), "text": jnp.array([0.5])},
            aux={},
        )
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )

        expected_img = uncond_img + w * (cond_img - uncond_img)
        expected_txt = uncond_txt + w * (cond_txt - uncond_txt)
        assert jnp.allclose(result["image"].value, expected_img)
        assert jnp.allclose(result["text"].value, expected_txt)

    def test_multimodal_preserves_kinds(self):
        """Each modality's prediction kind is preserved."""

        def predict(x, s, t, cond, aux):
            return {
                "a": Prediction(value=jnp.ones((1, 2)), kind="eps"),
                "b": Prediction(value=jnp.ones((1, 3)), kind="v_t"),
            }

        cfg = CFG(guidance_scale=2.0)
        step = SamplerStep(
            x={"a": jnp.zeros((1, 2)), "b": jnp.zeros((1, 3))},
            t={"a": jnp.array([0.5]), "b": jnp.array([0.5])},
            aux={},
        )
        pred = predict(step.x, None, step.t, "cond", step.aux)
        result = cfg.transform_prediction(
            jax.random.key(0), predict, pred, step, jnp.array([0.4]), "cond"
        )

        assert result["a"].kind == "eps"
        assert result["b"].kind == "v_t"


# ===========================================================================
# Tests for CFG end-to-end through sample()
# ===========================================================================


class TestCFGEndToEnd:
    """Integration tests for CFG through the sample() loop."""

    def test_cfg_scale_one_is_identity(self):
        """CFG with scale=1.0 equals unguided sampling."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=x * 0.1, kind="eps")

        out_plain = sample(key, sampler, predict, BATCH_SIZE, sched)
        out_cfg = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=1.0)],
        )
        assert jnp.allclose(out_plain, out_cfg, atol=1e-5)

    def test_cfg_changes_output(self):
        """CFG with scale != 1.0 changes the sampling output."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            if cond is None:
                return Prediction(value=jnp.zeros_like(x), kind="eps")
            return Prediction(value=x * 0.1, kind="eps")

        out_plain = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            cond=jnp.ones((BATCH_SIZE, 4)),
        )
        out_cfg = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            cond=jnp.ones((BATCH_SIZE, 4)),
            transforms=[CFG(guidance_scale=5.0)],
        )
        assert not jnp.allclose(out_plain, out_cfg)

    def test_cfg_output_shape(self):
        """CFG through sample() produces correct output shape."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="eps")

        out = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=2.0)],
        )
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)

    def test_cfg_produces_finite(self):
        """CFG output is finite."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="eps")

        out = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=7.5)],
        )
        assert jnp.all(jnp.isfinite(out))

    def test_cfg_with_euler(self):
        """CFG works with Euler sampler."""
        process = FlowMatching(DATA_SHAPE)
        sampler = EulerSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="v_t")

        out = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=2.0)],
        )
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))

    @pytest.mark.parametrize("w", [0.0, 0.5, 1.0, 2.0, 7.5])
    def test_cfg_various_scales_end_to_end(self, w: float):
        """CFG runs without error for various scales."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="eps")

        out = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=w)],
        )
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))

    def test_cfg_calls_predictor_twice_per_step(self):
        """CFG calls the predictor once for cond and once for uncond per step."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        call_count = [0]

        def counting_predict(x, s, t, cond, aux):
            call_count[0] += 1
            return Prediction(value=jnp.zeros_like(x), kind="eps")

        sample(
            key,
            sampler,
            counting_predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=2.0)],
        )

        # NUM_STEPS normal calls + NUM_STEPS unconditional calls from CFG
        assert call_count[0] == 2 * NUM_STEPS

    def test_cfg_deterministic(self):
        """CFG sampling is deterministic given same key."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return Prediction(value=x * 0.1, kind="eps")

        out1 = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=3.0)],
        )
        out2 = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=3.0)],
        )
        assert jnp.allclose(out1, out2)
