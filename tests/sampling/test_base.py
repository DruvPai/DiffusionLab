"""Tests for diffusionlab.sampling.base (SamplerStep, SamplingTransform, sample)."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.gaussian import VPCosine
from diffusionlab.sampling.base import (
    SamplerStep,
    SamplingTransform,
    sample,
    sample_trajectory,
)
from diffusionlab.sampling.continuous import DDMSampler
from diffusionlab.sampling.schedules import uniform_schedule
from diffusionlab.typing import Prediction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
DATA_SHAPE = (8,)
NUM_STEPS = 5


# ===========================================================================
# Tests for SamplerStep
# ===========================================================================


class TestSamplerStep:
    """Tests for SamplerStep as a frozen dataclass and JAX pytree."""

    def test_fields(self):
        """SamplerStep has x, t, aux fields."""
        step = SamplerStep(
            x=jnp.ones((2, 3)),
            t=jnp.array([0.5, 0.5]),
            aux={"alpha": jnp.array([0.8, 0.8])},
        )
        assert hasattr(step, "x")
        assert hasattr(step, "t")
        assert hasattr(step, "aux")

    def test_frozen(self):
        """SamplerStep is frozen (immutable)."""
        step = SamplerStep(
            x=jnp.ones((2, 3)),
            t=jnp.array([0.5, 0.5]),
            aux={},
        )
        with pytest.raises(AttributeError):
            step.x = jnp.zeros((2, 3))  # type: ignore[misc]

    def test_pytree_roundtrip(self):
        """Flatten/unflatten preserves all data."""
        step = SamplerStep(
            x=jnp.ones((2, 3)),
            t=jnp.array([0.5, 0.5]),
            aux={"alpha": jnp.array([0.8, 0.8])},
        )
        leaves, treedef = jax.tree_util.tree_flatten(step)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.x, step.x)
        assert jnp.allclose(restored.t, step.t)
        assert jnp.allclose(restored.aux["alpha"], step.aux["alpha"])

    def test_tree_map(self):
        """jax.tree.map applies to all leaves."""
        step = SamplerStep(
            x=jnp.ones((2, 3)),
            t=jnp.array([0.5, 0.5]),
            aux={"v_t": jnp.array([1.0, 1.0])},
        )
        doubled = jax.tree_util.tree_map(lambda v: 2 * v, step)
        assert jnp.allclose(doubled.x, 2.0 * jnp.ones((2, 3)))
        assert jnp.allclose(doubled.t, jnp.array([1.0, 1.0]))
        assert jnp.allclose(doubled.aux["v_t"], jnp.array([2.0, 2.0]))

    def test_pytree_leaf_count(self):
        """Leaf count matches expected number of arrays."""
        step = SamplerStep(
            x=jnp.ones((2, 3)),
            t=jnp.array([0.5, 0.5]),
            aux={"a": jnp.array([1.0]), "b": jnp.array([2.0])},
        )
        leaves = jax.tree_util.tree_leaves(step)
        assert len(leaves) == 4  # x, t, a, b

    def test_empty_aux(self):
        """SamplerStep works with empty aux dict."""
        step = SamplerStep(
            x=jnp.ones((2, 3)),
            t=jnp.array([0.5, 0.5]),
            aux={},
        )
        leaves, treedef = jax.tree_util.tree_flatten(step)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.x, step.x)
        assert restored.aux == {}

    def test_pytree_with_nested_aux(self):
        """SamplerStep handles nested aux dictionaries."""
        step = SamplerStep(
            x=jnp.ones((2, 3)),
            t=jnp.array([0.5, 0.5]),
            aux={
                "coeffs": {
                    "alpha": jnp.array([0.8, 0.8]),
                    "sigma": jnp.array([0.6, 0.6]),
                }
            },
        )
        leaves = jax.tree_util.tree_leaves(step)
        assert len(leaves) == 4  # x, t, alpha, sigma


# ===========================================================================
# Tests for SamplingTransform base class
# ===========================================================================


class TestSamplingTransform:
    """Tests for the SamplingTransform ABC defaults."""

    def test_default_transform_prediction_is_identity(self):
        """Default transform_prediction returns prediction unchanged."""

        class NoopTransform(SamplingTransform):
            pass

        transform = NoopTransform()
        pred = Prediction(value=jnp.array([[1.0, 2.0]]), kind="eps")
        step = SamplerStep(
            x=jnp.zeros((1, 2)),
            t=jnp.array([0.5]),
            aux={},
        )

        result = transform.transform_prediction(
            jax.random.key(0),
            lambda *a: pred,
            pred,
            step,
            jnp.array([0.4]),
            None,
        )
        assert jnp.allclose(result.value, pred.value)
        assert result.kind == pred.kind

    def test_default_transform_step_is_identity(self):
        """Default transform_step returns step unchanged."""

        class NoopTransform(SamplingTransform):
            pass

        transform = NoopTransform()
        step = SamplerStep(
            x=jnp.ones((1, 2)),
            t=jnp.array([0.3]),
            aux={},
        )
        pred = Prediction(value=jnp.zeros((1, 2)), kind="eps")
        prev_step = SamplerStep(
            x=jnp.ones((1, 2)),
            t=jnp.array([0.5]),
            aux={},
        )

        result = transform.transform_step(jax.random.key(0), step, pred, prev_step)
        assert jnp.allclose(result.x, step.x)
        assert jnp.allclose(result.t, step.t)

    def test_custom_prediction_transform(self):
        """Custom transform_prediction is called correctly."""

        class ScaleTransform(SamplingTransform):
            def __init__(self, factor: float):
                super().__init__()
                self.factor: float = factor

            def transform_prediction(self, key, predictor, prediction, step, s, cond):
                return Prediction(
                    value=prediction.value * self.factor, kind=prediction.kind
                )

        transform = ScaleTransform(3.0)
        pred = Prediction(value=jnp.array([[1.0, 2.0]]), kind="eps")
        step = SamplerStep(x=jnp.zeros((1, 2)), t=jnp.array([0.5]), aux={})

        result = transform.transform_prediction(
            jax.random.key(0),
            lambda *a: pred,
            pred,
            step,
            jnp.array([0.4]),
            None,
        )
        assert jnp.allclose(result.value, jnp.array([[3.0, 6.0]]))

    def test_custom_step_transform(self):
        """Custom transform_step modifies the step."""

        class ShiftTransform(SamplingTransform):
            def __init__(self, offset: float):
                super().__init__()
                self.offset: float = offset

            def transform_step(self, key, step, prediction, prev_step):
                return SamplerStep(
                    x=step.x + self.offset,
                    t=step.t,
                    aux=step.aux,
                )

        transform = ShiftTransform(10.0)
        step = SamplerStep(x=jnp.ones((1, 2)), t=jnp.array([0.3]), aux={})
        pred = Prediction(value=jnp.zeros((1, 2)), kind="eps")
        prev = SamplerStep(x=jnp.ones((1, 2)), t=jnp.array([0.5]), aux={})

        result = transform.transform_step(jax.random.key(0), step, pred, prev)
        assert jnp.allclose(result.x, jnp.ones((1, 2)) + 10.0)


# ===========================================================================
# Tests for the sample() orchestrator
# ===========================================================================


class TestSampleOrchestrator:
    """Tests for the sample() function that drives the sampling loop."""

    def _make_dummy_predictor(self, kind="eps"):
        """Return a simple predictor that predicts zeros."""

        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind=kind)

        return predict

    def test_output_shape(self):
        """sample() returns correct output shape."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        out = sample(key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)

    def test_deterministic_same_key(self):
        """Same key produces identical outputs."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = self._make_dummy_predictor()

        out1 = sample(key, sampler, predict, BATCH_SIZE, sched)
        out2 = sample(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.allclose(out1, out2)

    def test_different_keys_different_outputs(self):
        """Different keys produce different outputs (for stochastic terminal sampling)."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        predict = self._make_dummy_predictor()

        out1 = sample(jax.random.key(0), sampler, predict, BATCH_SIZE, sched)
        out2 = sample(jax.random.key(1), sampler, predict, BATCH_SIZE, sched)
        assert not jnp.allclose(out1, out2)

    def test_conditioning_passed_to_predictor(self):
        """Conditioning data is passed through to the predictor."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        received_conds = []

        def tracking_predict(x, s, t, cond, aux):
            received_conds.append(cond)
            return Prediction(value=jnp.zeros_like(x), kind="eps")

        cond_data = jnp.ones((BATCH_SIZE, 16))
        sample(key, sampler, tracking_predict, BATCH_SIZE, sched, cond=cond_data)

        # All calls should receive the conditioning
        for c in received_conds:
            assert jnp.allclose(c, cond_data)

    def test_none_conditioning_by_default(self):
        """Without cond argument, predictor receives cond=None."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        received_conds = []

        def tracking_predict(x, s, t, cond, aux):
            received_conds.append(cond)
            return Prediction(value=jnp.zeros_like(x), kind="eps")

        sample(key, sampler, tracking_predict, BATCH_SIZE, sched)
        for c in received_conds:
            assert c is None

    def test_predictor_called_correct_number_of_times(self):
        """Predictor is called num_steps times (intermediate + final)."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        call_count = [0]

        def counting_predict(x, s, t, cond, aux):
            call_count[0] += 1
            return Prediction(value=jnp.zeros_like(x), kind="eps")

        sample(key, sampler, counting_predict, BATCH_SIZE, sched)
        # num_steps - 1 intermediate + 1 final = num_steps
        assert call_count[0] == NUM_STEPS

    def test_empty_transforms_list(self):
        """Explicit empty transforms list equals no transforms."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = self._make_dummy_predictor()

        out_none = sample(key, sampler, predict, BATCH_SIZE, sched, transforms=None)
        out_empty = sample(key, sampler, predict, BATCH_SIZE, sched, transforms=[])
        assert jnp.allclose(out_none, out_empty)

    def test_identity_transform_is_noop(self):
        """An identity SamplingTransform doesn't change the output."""

        class IdentityTransform(SamplingTransform):
            pass

        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = self._make_dummy_predictor()

        out_plain = sample(key, sampler, predict, BATCH_SIZE, sched)
        out_with = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[IdentityTransform()],
        )
        assert jnp.allclose(out_plain, out_with)

    def test_prediction_transforms_applied_in_order(self):
        """Multiple prediction transforms are applied sequentially."""

        class ScalePrediction(SamplingTransform):
            def __init__(self, factor: float):
                super().__init__()
                self.factor: float = factor

            def transform_prediction(self, key, predictor, prediction, step, s, cond):
                return Prediction(
                    value=prediction.value * self.factor,
                    kind=prediction.kind,
                )

        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def ones_predict(x, s, t, cond, aux):
            return Prediction(value=jnp.ones_like(x), kind="eps")

        # 2x then 3x = 6x
        out_composed = sample(
            key,
            sampler,
            ones_predict,
            BATCH_SIZE,
            sched,
            transforms=[ScalePrediction(2.0), ScalePrediction(3.0)],
        )
        out_single = sample(
            key,
            sampler,
            ones_predict,
            BATCH_SIZE,
            sched,
            transforms=[ScalePrediction(6.0)],
        )
        assert jnp.allclose(out_composed, out_single, atol=1e-5)

    def test_step_transform_modifies_trajectory(self):
        """A step transform that shifts x changes the output."""

        class AddOffset(SamplingTransform):
            def transform_step(self, key, step, prediction, prev_step):
                return SamplerStep(
                    x=step.x + 10.0,
                    t=step.t,
                    aux=step.aux,
                )

        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        # Use an input-dependent predictor so shifted x produces different predictions
        def predict(x, s, t, cond, aux):
            return Prediction(value=x * 0.1, kind="eps")

        out_plain = sample(key, sampler, predict, BATCH_SIZE, sched)
        out_shifted = sample(
            key,
            sampler,
            predict,
            BATCH_SIZE,
            sched,
            transforms=[AddOffset()],
        )
        assert not jnp.allclose(out_plain, out_shifted)

    def test_transforms_receive_correct_args(self):
        """Transforms receive the correct arguments from the sample loop."""
        received = []

        class TrackingTransform(SamplingTransform):
            def transform_prediction(self, key, predictor, prediction, step, s, cond):
                received.append(
                    {
                        "x_shape": step.x.shape,
                        "t_shape": step.t.shape,
                        "s_shape": s.shape,
                        "pred_kind": prediction.kind,
                        "has_cond": cond is not None,
                    }
                )
                return prediction

        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        cond = jnp.ones((BATCH_SIZE, 4))
        sample(
            key,
            sampler,
            self._make_dummy_predictor(),
            BATCH_SIZE,
            sched,
            cond=cond,
            transforms=[TrackingTransform()],
        )

        assert len(received) == NUM_STEPS  # called at each step
        for r in received:
            assert r["x_shape"] == (BATCH_SIZE, *DATA_SHAPE)
            assert r["t_shape"] == (BATCH_SIZE,)
            assert r["s_shape"] == (BATCH_SIZE,)
            assert r["pred_kind"] == "eps"
            assert r["has_cond"] is True

    @pytest.mark.parametrize("num_steps", [2, 5, 10, 20])
    def test_more_steps_generally_works(self, num_steps: int):
        """sample() works with different step counts."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        key = jax.random.key(42)

        out = sample(key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched)
        assert out.shape == (BATCH_SIZE, *DATA_SHAPE)
        assert jnp.all(jnp.isfinite(out))


# ===========================================================================
# Tests for the sample_trajectory() orchestrator
# ===========================================================================


class TestSampleTrajectoryOrchestrator:
    """Tests for sample_trajectory() that returns the full denoising path."""

    def _make_dummy_predictor(self, kind="eps"):
        def predict(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind=kind)

        return predict

    def test_final_matches_sample(self):
        """sample_trajectory final output matches sample output."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = self._make_dummy_predictor()

        out = sample(key, sampler, predict, BATCH_SIZE, sched)
        final, _traj = sample_trajectory(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.allclose(out, final)

    def test_trajectory_length(self):
        """Trajectory has num_steps entries (initial + intermediate updates)."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        _final, traj = sample_trajectory(
            key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched
        )
        # initial state + (num_steps - 1) intermediate updates = num_steps
        assert len(traj) == NUM_STEPS

    @pytest.mark.parametrize("num_steps", [2, 5, 10])
    def test_trajectory_length_varies(self, num_steps: int):
        """Trajectory length scales with num_steps."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        key = jax.random.key(42)

        _final, traj = sample_trajectory(
            key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched
        )
        assert len(traj) == num_steps

    def test_trajectory_elements_are_sampler_steps(self):
        """Every element in trajectory is a SamplerStep."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        _final, traj = sample_trajectory(
            key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched
        )
        for step in traj:
            assert isinstance(step, SamplerStep)

    def test_trajectory_shapes(self):
        """Each trajectory step has correct x and t shapes."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        _final, traj = sample_trajectory(
            key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched
        )
        for step in traj:
            assert step.x.shape == (BATCH_SIZE, *DATA_SHAPE)
            assert step.t.shape == (BATCH_SIZE,)

    def test_trajectory_times_decrease(self):
        """Times in trajectory are monotonically decreasing (t=1 → t~0)."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        _final, traj = sample_trajectory(
            key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched
        )
        times = [float(step.t[0]) for step in traj]
        for i in range(len(times) - 1):
            assert times[i] > times[i + 1]

    def test_trajectory_first_step_is_terminal(self):
        """First trajectory entry is at the terminal time t=1."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        _final, traj = sample_trajectory(
            key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched
        )
        assert jnp.allclose(traj[0].t, jnp.ones(BATCH_SIZE))

    def test_deterministic_same_key(self):
        """Same key produces identical trajectory."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)
        predict = self._make_dummy_predictor()

        final1, traj1 = sample_trajectory(key, sampler, predict, BATCH_SIZE, sched)
        final2, traj2 = sample_trajectory(key, sampler, predict, BATCH_SIZE, sched)
        assert jnp.allclose(final1, final2)
        for s1, s2 in zip(traj1, traj2):
            assert jnp.allclose(s1.x, s2.x)
            assert jnp.allclose(s1.t, s2.t)

    def test_with_transforms(self):
        """Transforms are applied and trajectory still matches sample."""

        class ScalePrediction(SamplingTransform):
            def __init__(self, factor: float):
                super().__init__()
                self.factor: float = factor

            def transform_prediction(self, key, predictor, prediction, step, s, cond):
                return Prediction(
                    value=prediction.value * self.factor,
                    kind=prediction.kind,
                )

        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        def ones_predict(x, s, t, cond, aux):
            return Prediction(value=jnp.ones_like(x), kind="eps")

        transforms = [ScalePrediction(2.0)]

        out = sample(
            key, sampler, ones_predict, BATCH_SIZE, sched, transforms=transforms
        )
        final, traj = sample_trajectory(
            key, sampler, ones_predict, BATCH_SIZE, sched, transforms=transforms
        )
        assert jnp.allclose(out, final)
        assert len(traj) == NUM_STEPS

    def test_conditioning_passed_through(self):
        """Conditioning reaches the predictor in sample_trajectory."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        received_conds = []

        def tracking_predict(x, s, t, cond, aux):
            received_conds.append(cond)
            return Prediction(value=jnp.zeros_like(x), kind="eps")

        cond_data = jnp.ones((BATCH_SIZE, 16))
        sample_trajectory(
            key, sampler, tracking_predict, BATCH_SIZE, sched, cond=cond_data
        )
        for c in received_conds:
            assert jnp.allclose(c, cond_data)

    def test_trajectory_values_finite(self):
        """All trajectory entries have finite values."""
        process = VPCosine(DATA_SHAPE)
        sampler = DDMSampler(process, eta=0.0)
        sched = uniform_schedule(NUM_STEPS, BATCH_SIZE)
        key = jax.random.key(42)

        final, traj = sample_trajectory(
            key, sampler, self._make_dummy_predictor(), BATCH_SIZE, sched
        )
        assert jnp.all(jnp.isfinite(final))
        for step in traj:
            assert jnp.all(jnp.isfinite(step.x))
            assert jnp.all(jnp.isfinite(step.t))
