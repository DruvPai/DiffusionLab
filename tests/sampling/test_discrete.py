"""Tests for DiscreteAncestral sampler."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.discrete import (
    DiscreteCorruptionProcess,
    MaskCosine,
    MaskLinear,
    UniformCosine,
    UniformLinear,
)
from diffusionlab.sampling.base import Sampler, SamplerStep, sample
from diffusionlab.sampling.discrete import DiscreteAncestralSampler
from diffusionlab.sampling.schedules import uniform_schedule
from diffusionlab.typing import Prediction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
VOCAB_SIZE = 10
SEQ_SHAPE = (6,)
NUM_STEPS = 5

MASK_PROCESS_NAMES = ["mask_linear", "mask_cosine"]
UNIFORM_PROCESS_NAMES = ["uniform_linear", "uniform_cosine"]
ALL_PROCESS_NAMES = MASK_PROCESS_NAMES + UNIFORM_PROCESS_NAMES


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_process(name: str) -> DiscreteCorruptionProcess:
    if name == "mask_linear":
        return MaskLinear(VOCAB_SIZE, SEQ_SHAPE)
    elif name == "mask_cosine":
        return MaskCosine(VOCAB_SIZE, SEQ_SHAPE)
    elif name == "uniform_linear":
        return UniformLinear(VOCAB_SIZE, SEQ_SHAPE)
    elif name == "uniform_cosine":
        return UniformCosine(VOCAB_SIZE, SEQ_SHAPE)
    else:
        raise ValueError(f"Unknown process: {name}")


def _logits_predict(batch_size, seq_shape, vocab_size, seed=0):
    """Return a predictor that gives random logits."""
    logits = jax.random.normal(
        jax.random.key(seed), (batch_size, *seq_shape, vocab_size)
    )

    def predict(x, s, t, cond, aux):
        return Prediction(value=logits, kind="logits_x0")

    return predict


# ===========================================================================
# Tests for DiscreteAncestral construction
# ===========================================================================


class TestDiscreteAncestralConstruction:
    """Tests for DiscreteAncestral construction."""

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_is_sampler(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        assert isinstance(sampler, Sampler)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_process_stored(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        assert sampler.process is process

    def test_rejects_non_discrete_process(self):
        """DiscreteAncestral only accepts DiscreteCorruptionProcess."""
        from diffusionlab.processes.gaussian import VPCosine

        with pytest.raises(AssertionError):
            DiscreteAncestralSampler(VPCosine((8,)))  # type: ignore


# ===========================================================================
# Tests for DiscreteAncestral initialize
# ===========================================================================


class TestDiscreteAncestralInitialize:
    """Tests for DiscreteAncestral.initialize."""

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_returns_sampler_step(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_T = jnp.full((BATCH_SIZE, *SEQ_SHAPE), VOCAB_SIZE, dtype=jnp.int32)
        T = jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, T)
        assert isinstance(step, SamplerStep)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_preserves_input(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_T = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        T = jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, T)
        assert jnp.array_equal(step.x, x_T)
        assert jnp.allclose(step.t, T)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_aux_has_coefficients(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_T = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        T = jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, T)
        assert "beta" in step.aux
        assert "beta_prime" in step.aux


# ===========================================================================
# Tests for DiscreteAncestral update
# ===========================================================================


class TestDiscreteAncestralUpdate:
    """Tests for DiscreteAncestral.update step logic."""

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_output_shapes(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.6 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        logits = jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")
        new_step = sampler.update(jax.random.key(2), pred, step, s)

        assert new_step.x.shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert jnp.allclose(new_step.t, s)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_output_in_valid_range(self, name: str):
        """Updated tokens are in valid vocabulary range."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        logits = jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")
        new_step = sampler.update(jax.random.key(2), pred, step, s)

        # For mask processes, output can include mask index
        from diffusionlab.processes.discrete import MaskDiffusion

        if isinstance(process, MaskDiffusion):
            assert jnp.all(new_step.x >= 0)
            assert jnp.all(new_step.x <= VOCAB_SIZE)  # mask_index = VOCAB_SIZE
        else:
            assert jnp.all(new_step.x >= 0)
            assert jnp.all(new_step.x < VOCAB_SIZE)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_is_stochastic(self, name: str):
        """DiscreteAncestral sampling is stochastic (different keys, different outputs)."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)

        # Initialize from terminal distribution so mask processes have mask tokens
        # and uniform processes have random tokens — both with stochastic posteriors
        x_T, _ = process.sample_from_terminal(key, BATCH_SIZE)
        t = jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_T, t)
        # Use nearly-uniform logits so sampling has high entropy
        logits = 0.01 * jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")

        out1 = sampler.update(jax.random.key(10), pred, step, s)
        out2 = sampler.update(jax.random.key(20), pred, step, s)
        assert not jnp.array_equal(out1.x, out2.x)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_deterministic_same_key(self, name: str):
        """Same key produces same output."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        logits = jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")

        update_key = jax.random.key(42)
        out1 = sampler.update(update_key, pred, step, s)
        out2 = sampler.update(update_key, pred, step, s)
        assert jnp.array_equal(out1.x, out2.x)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_integer_output(self, name: str):
        """Output tokens are integers."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.8 * jnp.ones((BATCH_SIZE,))
        s = 0.4 * jnp.ones((BATCH_SIZE,))

        step = sampler.initialize(key, x_t, t)
        logits = jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")
        new_step = sampler.update(jax.random.key(2), pred, step, s)

        assert jnp.issubdtype(new_step.x.dtype, jnp.integer)


# ===========================================================================
# Tests for DiscreteAncestral finalize
# ===========================================================================


class TestDiscreteAncestralFinalize:
    """Tests for DiscreteAncestral.finalize."""

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_output_shape(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.05 * jnp.ones((BATCH_SIZE,))

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        logits = jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")
        out = sampler.finalize(key, pred, step)
        assert out.shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_output_is_argmax(self, name: str):
        """finalize returns argmax of the predicted x_0 distribution."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.05 * jnp.ones((BATCH_SIZE,))

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        logits = jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")
        out = sampler.finalize(key, pred, step)

        # The finalize converts to x_0 probs and takes argmax
        x_0_pred = process.convert_prediction(pred, x_t, None, t, "x_0")
        expected = jnp.argmax(x_0_pred.value, axis=-1)
        assert jnp.array_equal(out, expected)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_output_in_vocab_range(self, name: str):
        """Finalized output is in {0, ..., vocab_size - 1}."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.05 * jnp.ones((BATCH_SIZE,))

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        logits = jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")
        out = sampler.finalize(key, pred, step)

        assert jnp.all(out >= 0)
        assert jnp.all(out < VOCAB_SIZE)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_integer_output(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.05 * jnp.ones((BATCH_SIZE,))

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        logits = jax.random.normal(
            jax.random.key(1), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
        )
        pred = Prediction(value=logits, kind="logits_x0")
        out = sampler.finalize(key, pred, step)
        assert jnp.issubdtype(out.dtype, jnp.integer)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_confident_prediction_recovers_mode(self, name: str):
        """When logits strongly favour one class, finalize returns that class."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        key = jax.random.key(0)
        x_t = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE)
        t = 0.05 * jnp.ones((BATCH_SIZE,))

        target_class = 3
        logits = -100.0 * jnp.ones((BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE))
        logits = logits.at[..., target_class].set(100.0)

        step = SamplerStep(x=x_t, t=t, aux=process.get_coefficients(t))
        pred = Prediction(value=logits, kind="logits_x0")
        out = sampler.finalize(key, pred, step)

        assert jnp.all(out == target_class)


# ===========================================================================
# Tests for DiscreteAncestral end-to-end through sample()
# ===========================================================================


class TestDiscreteAncestralEndToEnd:
    """End-to-end tests for DiscreteAncestral through sample()."""

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_sample_output_shape(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _logits_predict(BATCH_SIZE, SEQ_SHAPE, VOCAB_SIZE)

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_sample_in_vocab_range(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _logits_predict(BATCH_SIZE, SEQ_SHAPE, VOCAB_SIZE)

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.all(out >= 0)
        assert jnp.all(out < VOCAB_SIZE)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_sample_integer_output(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _logits_predict(BATCH_SIZE, SEQ_SHAPE, VOCAB_SIZE)

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.issubdtype(out.dtype, jnp.integer)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_deterministic_same_key(self, name: str):
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _logits_predict(BATCH_SIZE, SEQ_SHAPE, VOCAB_SIZE)

        out1 = sample(key, sampler, predict, BATCH_SIZE, sched)
        out2 = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.array_equal(out1, out2)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_stochastic_different_keys(self, name: str):
        """Different keys produce different intermediate trajectories."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)

        # Predict one-hot on x_t: the model "copies" its noisy input.
        # This makes finalize's argmax depend on the stochastic trajectory.
        def echo_predict(x, s, t, cond, aux):
            logits = 100.0 * jax.nn.one_hot(x, VOCAB_SIZE)
            return Prediction(value=logits, kind="logits_x0")

        out1 = sample(jax.random.key(0), sampler, echo_predict, BATCH_SIZE, sched)
        out2 = sample(jax.random.key(1), sampler, echo_predict, BATCH_SIZE, sched)
        assert not jnp.array_equal(out1, out2)

    @pytest.mark.parametrize("name", ALL_PROCESS_NAMES)
    def test_confident_prediction_converges(self, name: str):
        """With very confident logits, output should match the predicted class."""
        process = _make_process(name)
        sampler = DiscreteAncestralSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        target_class = 5
        logits = -100.0 * jnp.ones((BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE))
        logits = logits.at[..., target_class].set(100.0)

        def predict(x, s, t, cond, aux):
            return Prediction(value=logits, kind="logits_x0")

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.all(out == target_class)

    @pytest.mark.parametrize("num_steps", [2, 5, 10, 20])
    def test_various_step_counts(self, num_steps: int):
        """DiscreteAncestral works with different step counts."""
        process = MaskLinear(VOCAB_SIZE, SEQ_SHAPE)
        sampler = DiscreteAncestralSampler(process)
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        key = jax.random.key(42)
        predict = _logits_predict(BATCH_SIZE, SEQ_SHAPE, VOCAB_SIZE)

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert jnp.all(out >= 0)
        assert jnp.all(out < VOCAB_SIZE)

    def test_with_conditioning(self):
        """DiscreteAncestral works with conditioning data."""
        process = MaskLinear(VOCAB_SIZE, SEQ_SHAPE)
        sampler = DiscreteAncestralSampler(process)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        cond = jnp.ones((BATCH_SIZE, 16))

        received_conds = []

        def predict(x, s, t, cond, aux):
            received_conds.append(cond)
            logits = jax.random.normal(
                jax.random.key(0), (BATCH_SIZE, *SEQ_SHAPE, VOCAB_SIZE)
            )
            return Prediction(value=logits, kind="logits_x0")

        out = sample(key, sampler, predict, BATCH_SIZE, sched, cond=cond)
        assert out.shape == (BATCH_SIZE, *SEQ_SHAPE)
        for c in received_conds:
            assert jnp.allclose(c, cond)
