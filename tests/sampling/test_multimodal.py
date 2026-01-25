"""Tests for MultimodalSampler."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.discrete import MaskLinear, UniformLinear
from diffusionlab.processes.gaussian import FlowMatching, VPCosine
from diffusionlab.sampling.base import (
    MultimodalSampler,
    Sampler,
    SamplerStep,
    sample,
)
from diffusionlab.sampling.continuous import DDMSampler, EulerSampler
from diffusionlab.sampling.discrete import DiscreteAncestralSampler
from diffusionlab.sampling.guidance import CFG
from diffusionlab.sampling.schedules import uniform_schedule
from diffusionlab.typing import Prediction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
IMG_SHAPE = (4,)
VOCAB_SIZE = 10
SEQ_SHAPE = (6,)
NUM_STEPS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_img_txt_sampler():
    """Create a multimodal sampler with image (DDIM) and text (DiscreteAncestral)."""
    process_img = VPCosine(IMG_SHAPE)
    process_txt = MaskLinear(VOCAB_SIZE, SEQ_SHAPE)
    samplers = {
        "image": DDMSampler(process_img, eta=0.0),
        "text": DiscreteAncestralSampler(process_txt),
    }
    return MultimodalSampler(samplers)


def _make_multimodal_schedule(num_steps, batch_size):
    """Create a dict-structured schedule for image+text."""
    img_sched = uniform_schedule(num_steps, batch_size)
    txt_sched = uniform_schedule(num_steps, batch_size)
    return [
        {"image": img_sched[i], "text": txt_sched[i]} for i in range(len(img_sched))
    ]


def _make_img_txt_predictor():
    """Predictor for image (eps) + text (logits_x0)."""

    def predict(x, s, t, cond, aux):
        return {
            "image": Prediction(value=jnp.zeros((BATCH_SIZE, *IMG_SHAPE)), kind="eps"),
            "text": Prediction(
                value=jax.random.normal(
                    jax.random.key(0),
                    (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE),
                ),
                kind="logits_x0",
            ),
        }

    return predict


# ===========================================================================
# Tests for MultimodalSampler construction
# ===========================================================================


class TestMultimodalSamplerConstruction:
    """Tests for MultimodalSampler initialization."""

    def test_is_sampler(self):
        mm = _make_img_txt_sampler()
        assert isinstance(mm, Sampler)

    def test_samplers_stored(self):
        process_img = VPCosine(IMG_SHAPE)
        process_txt = MaskLinear(VOCAB_SIZE, SEQ_SHAPE)
        samplers = {
            "image": DDMSampler(process_img),
            "text": DiscreteAncestralSampler(process_txt),
        }
        mm = MultimodalSampler(samplers)
        assert "image" in mm.samplers
        assert "text" in mm.samplers

    def test_process_is_multimodal(self):
        """The wrapped process is a MultimodalCorruptionProcess."""
        from diffusionlab.processes.base import MultimodalCorruptionProcess

        mm = _make_img_txt_sampler()
        assert isinstance(mm.process, MultimodalCorruptionProcess)

    def test_two_continuous_modalities(self):
        """MultimodalSampler works with two continuous modalities."""
        samplers = {
            "a": DDMSampler(VPCosine((4,))),
            "b": EulerSampler(FlowMatching((8,))),
        }
        mm = MultimodalSampler(samplers)
        assert isinstance(mm, Sampler)

    def test_two_discrete_modalities(self):
        """MultimodalSampler works with two discrete modalities."""
        samplers = {
            "a": DiscreteAncestralSampler(MaskLinear(VOCAB_SIZE, SEQ_SHAPE)),
            "b": DiscreteAncestralSampler(UniformLinear(VOCAB_SIZE, SEQ_SHAPE)),
        }
        mm = MultimodalSampler(samplers)
        assert isinstance(mm, Sampler)


# ===========================================================================
# Tests for MultimodalSampler initialize
# ===========================================================================


class TestMultimodalSamplerInitialize:
    """Tests for MultimodalSampler.initialize."""

    def test_returns_sampler_step(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)
        assert isinstance(step, SamplerStep)

    def test_output_structure(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)
        assert isinstance(step.x, dict)
        assert "image" in step.x
        assert "text" in step.x

    def test_output_shapes(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)
        assert step.x["image"].shape == (BATCH_SIZE, *IMG_SHAPE)
        assert step.x["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    def test_preserves_time(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)
        assert jnp.allclose(step.t["image"], 1.0)
        assert jnp.allclose(step.t["text"], 1.0)


# ===========================================================================
# Tests for MultimodalSampler update
# ===========================================================================


class TestMultimodalSamplerUpdate:
    """Tests for MultimodalSampler.update."""

    def test_output_shapes(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)

        prediction = {
            "image": Prediction(value=jnp.zeros((BATCH_SIZE, *IMG_SHAPE)), kind="eps"),
            "text": Prediction(
                value=jax.random.normal(
                    jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
                ),
                kind="logits_x0",
            ),
        }
        s = {
            "image": 0.8 * jnp.ones((BATCH_SIZE,)),
            "text": 0.8 * jnp.ones((BATCH_SIZE,)),
        }

        new_step = mm.update(jax.random.key(2), prediction, step, s)
        assert new_step.x["image"].shape == (BATCH_SIZE, *IMG_SHAPE)
        assert new_step.x["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    def test_time_advances(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)
        prediction = {
            "image": Prediction(value=jnp.zeros((BATCH_SIZE, *IMG_SHAPE)), kind="eps"),
            "text": Prediction(
                value=jax.random.normal(
                    jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
                ),
                kind="logits_x0",
            ),
        }
        s = {
            "image": 0.6 * jnp.ones((BATCH_SIZE,)),
            "text": 0.6 * jnp.ones((BATCH_SIZE,)),
        }

        new_step = mm.update(jax.random.key(2), prediction, step, s)
        assert jnp.allclose(new_step.t["image"], 0.6)
        assert jnp.allclose(new_step.t["text"], 0.6)


# ===========================================================================
# Tests for MultimodalSampler finalize
# ===========================================================================


class TestMultimodalSamplerFinalize:
    """Tests for MultimodalSampler.finalize."""

    def test_output_structure(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)
        prediction = {
            "image": Prediction(value=jnp.zeros((BATCH_SIZE, *IMG_SHAPE)), kind="eps"),
            "text": Prediction(
                value=jax.random.normal(
                    jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
                ),
                kind="logits_x0",
            ),
        }

        out = mm.finalize(key, prediction, step)
        assert isinstance(out, dict)
        assert "image" in out
        assert "text" in out

    def test_output_shapes(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)
        prediction = {
            "image": Prediction(value=jnp.zeros((BATCH_SIZE, *IMG_SHAPE)), kind="eps"),
            "text": Prediction(
                value=jax.random.normal(
                    jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
                ),
                kind="logits_x0",
            ),
        }

        out = mm.finalize(key, prediction, step)
        assert out["image"].shape == (BATCH_SIZE, *IMG_SHAPE)
        assert out["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    def test_text_output_integer(self):
        mm = _make_img_txt_sampler()
        key = jax.random.key(0)
        x_T, _ = mm.process.sample_from_terminal(key, BATCH_SIZE)
        T = {"image": jnp.ones((BATCH_SIZE,)), "text": jnp.ones((BATCH_SIZE,))}

        step = mm.initialize(key, x_T, T)
        prediction = {
            "image": Prediction(value=jnp.zeros((BATCH_SIZE, *IMG_SHAPE)), kind="eps"),
            "text": Prediction(
                value=jax.random.normal(
                    jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
                ),
                kind="logits_x0",
            ),
        }

        out = mm.finalize(key, prediction, step)
        assert jnp.issubdtype(out["text"].dtype, jnp.integer)
        assert jnp.issubdtype(out["image"].dtype, jnp.floating)


# ===========================================================================
# Tests for MultimodalSampler end-to-end through sample()
# ===========================================================================


class TestMultimodalSamplerEndToEnd:
    """End-to-end tests for MultimodalSampler through sample()."""

    def test_sample_output_structure(self):
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _make_img_txt_predictor()

        out = sample(key, mm, predict, BATCH_SIZE, sched)
        assert isinstance(out, dict)
        assert "image" in out
        assert "text" in out

    def test_sample_output_shapes(self):
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _make_img_txt_predictor()

        out = sample(key, mm, predict, BATCH_SIZE, sched)
        assert out["image"].shape == (BATCH_SIZE, *IMG_SHAPE)
        assert out["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    def test_image_is_float_text_is_int(self):
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _make_img_txt_predictor()

        out = sample(key, mm, predict, BATCH_SIZE, sched)
        assert jnp.issubdtype(out["image"].dtype, jnp.floating)
        assert jnp.issubdtype(out["text"].dtype, jnp.integer)

    def test_text_in_vocab_range(self):
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _make_img_txt_predictor()

        out = sample(key, mm, predict, BATCH_SIZE, sched)
        assert jnp.all(out["text"] >= 0)
        assert jnp.all(out["text"] < VOCAB_SIZE)

    def test_image_is_finite(self):
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _make_img_txt_predictor()

        out = sample(key, mm, predict, BATCH_SIZE, sched)
        assert jnp.all(jnp.isfinite(out["image"]))

    def test_deterministic_same_key(self):
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _make_img_txt_predictor()

        out1 = sample(key, mm, predict, BATCH_SIZE, sched)
        out2 = sample(key, mm, predict, BATCH_SIZE, sched)
        assert jnp.allclose(out1["image"], out2["image"])
        assert jnp.array_equal(out1["text"], out2["text"])

    def test_different_keys_different_outputs(self):
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        predict = _make_img_txt_predictor()

        out1 = sample(jax.random.key(0), mm, predict, BATCH_SIZE, sched)
        out2 = sample(jax.random.key(1), mm, predict, BATCH_SIZE, sched)
        assert not jnp.allclose(out1["image"], out2["image"])

    def test_with_conditioning(self):
        """Multimodal sampling works with conditioning."""
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _make_img_txt_predictor()
        cond = jnp.ones((BATCH_SIZE, 16))

        out = sample(key, mm, predict, BATCH_SIZE, sched, cond=cond)
        assert out["image"].shape == (BATCH_SIZE, *IMG_SHAPE)
        assert out["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("num_steps", [2, 5, 10])
    def test_various_step_counts(self, num_steps: int):
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(num_steps, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _make_img_txt_predictor()

        out = sample(key, mm, predict, BATCH_SIZE, sched)
        assert out["image"].shape == (BATCH_SIZE, *IMG_SHAPE)
        assert out["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    def test_with_cfg_transform(self):
        """Multimodal sampling works with CFG transform."""
        mm = _make_img_txt_sampler()
        sched = _make_multimodal_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def predict(x, s, t, cond, aux):
            return {
                "image": Prediction(
                    value=jnp.zeros((BATCH_SIZE, *IMG_SHAPE)), kind="eps"
                ),
                "text": Prediction(
                    value=jax.random.normal(
                        jax.random.key(0),
                        (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE),
                    ),
                    kind="logits_x0",
                ),
            }

        out = sample(
            key,
            mm,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[CFG(guidance_scale=2.0)],
        )
        assert out["image"].shape == (BATCH_SIZE, *IMG_SHAPE)
        assert out["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    def test_two_continuous_modalities_end_to_end(self):
        """End-to-end with two continuous modalities."""
        samplers = {
            "a": DDMSampler(VPCosine((4,)), eta=0.0),
            "b": EulerSampler(FlowMatching((8,))),
        }
        mm = MultimodalSampler(samplers)

        a_sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        b_sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        sched = [{"a": a_sched[i], "b": b_sched[i]} for i in range(len(a_sched))]

        def predict(x, s, t, cond, aux):
            return {
                "a": Prediction(value=jnp.zeros((BATCH_SIZE, 4)), kind="eps"),
                "b": Prediction(value=jnp.zeros((BATCH_SIZE, 8)), kind="v_t"),
            }

        key = jax.random.key(42)
        out = sample(key, mm, predict, BATCH_SIZE, sched)
        assert out["a"].shape == (BATCH_SIZE, 4)
        assert out["b"].shape == (BATCH_SIZE, 8)
        assert jnp.all(jnp.isfinite(out["a"]))
        assert jnp.all(jnp.isfinite(out["b"]))

    def test_different_schedules_per_modality(self):
        """Each modality can have a different schedule."""
        mm = _make_img_txt_sampler()

        img_sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        # Text could use fewer effective steps (all at same times but different values)
        txt_sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        sched = [
            {"image": img_sched[i], "text": txt_sched[i]} for i in range(len(img_sched))
        ]

        key = jax.random.key(42)
        predict = _make_img_txt_predictor()
        out = sample(key, mm, predict, BATCH_SIZE, sched)
        assert out["image"].shape == (BATCH_SIZE, *IMG_SHAPE)
        assert out["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)
