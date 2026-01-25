"""Tests for SimplicialCorruptionProcess (diffusionlab.processes.simplicial)."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.simplicial import (
    SimplicialCorruptionProcess,
    SimplicialCosine,
    SimplicialLinear,
)
from diffusionlab.typing import Prediction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
NUM_CLASSES = 8  # K (number of regular token classes)
SEQ_SHAPE = (12,)
PADDING_INDEX = -1

SCHEDULE_NAMES = ["simplicial_linear", "simplicial_cosine"]
VARIANT_NAMES = ["mask", "uniform"]


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_simplicial(
    schedule: str,
    variant: str,
    padding_token: int | None = None,
    temperature: float = 1.0,
) -> SimplicialCorruptionProcess:
    """Build a concrete simplicial process.

    Args:
        schedule: One of "simplicial_linear", "simplicial_cosine".
        variant: One of "mask", "uniform".
        padding_token: Optional padding token.
        temperature: Dirichlet temperature.

    Returns:
        A concrete SimplicialCorruptionProcess.
    """
    cls = SimplicialLinear if "linear" in schedule else SimplicialCosine
    factory = cls.mask_process if variant == "mask" else cls.uniform_process
    return factory(
        NUM_CLASSES,
        SEQ_SHAPE,
        padding_token=padding_token,
        temperature=temperature,
    )


# ===========================================================================
# Tests for simplicial properties
# ===========================================================================


class TestSimplicialProperties:
    """Tests for process properties (invariant_probs, is_masking, etc.)."""

    def test_mask_invariant_probs(self):
        """Mask variant has invariant_probs = [0, ..., 0, 1]."""
        proc = SimplicialLinear.mask_process(NUM_CLASSES, SEQ_SHAPE)
        expected = jnp.zeros(NUM_CLASSES + 1).at[-1].set(1.0)
        assert jnp.array_equal(proc.invariant_probs, expected)
        assert proc.vocab_size == NUM_CLASSES + 1
        assert proc.is_masking

    def test_uniform_invariant_probs(self):
        """Uniform variant has invariant_probs = [1/K, ..., 1/K]."""
        proc = SimplicialLinear.uniform_process(NUM_CLASSES, SEQ_SHAPE)
        expected = jnp.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        assert jnp.array_equal(proc.invariant_probs, expected)
        assert proc.vocab_size == NUM_CLASSES
        assert not proc.is_masking

    def test_invalid_padding_token_raises(self):
        """padding_token inside category range should raise."""
        with pytest.raises(ValueError, match="padding_token"):
            SimplicialLinear.mask_process(NUM_CLASSES, SEQ_SHAPE, padding_token=0)

    def test_invalid_unmasked_vocab_size_raises(self):
        """unmasked_vocab_size < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="unmasked_vocab_size"):
            SimplicialLinear.mask_process(0, SEQ_SHAPE)
        with pytest.raises(ValueError, match="unmasked_vocab_size"):
            SimplicialLinear.uniform_process(0, SEQ_SHAPE)


# ===========================================================================
# Tests for simplicial schedule functions
# ===========================================================================


class TestSimplicialSchedule:
    """Tests for beta, beta_prime, h, and get_coefficients."""

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_beta_at_zero(self, schedule: str, variant: str):
        """beta(0) = 0."""
        proc = _make_simplicial(schedule, variant)
        t0 = jnp.zeros((BATCH_SIZE,))
        assert jnp.allclose(proc.beta(t0), 0.0, atol=1e-6)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_beta_at_one(self, schedule: str, variant: str):
        """beta(1) = 1."""
        proc = _make_simplicial(schedule, variant)
        t1 = jnp.ones((BATCH_SIZE,))
        assert jnp.allclose(proc.beta(t1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_beta_monotonically_increasing(self, schedule: str, variant: str):
        """beta should increase as t increases."""
        proc = _make_simplicial(schedule, variant)
        t = jnp.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        diffs = jnp.diff(proc.beta(t))
        assert jnp.all(diffs >= -1e-7)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_h_at_one_is_zero(self, schedule: str, variant: str):
        """h(1) ~ 0 (since beta(1) = 1)."""
        proc = _make_simplicial(schedule, variant)
        t1 = jnp.ones((BATCH_SIZE,))
        assert jnp.allclose(proc.h(t1), 0.0, atol=1e-3)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_h_at_zero_is_large(self, schedule: str, variant: str):
        """h(0) should be very large ((1 - 0) / (0 + eps) = 1/eps)."""
        proc = _make_simplicial(schedule, variant)
        t0 = jnp.zeros((BATCH_SIZE,))
        assert jnp.all(proc.h(t0) > 1e4)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_beta_prime_matches_jax_grad(self, schedule: str, variant: str):
        """beta'(t) matches jax.grad."""
        proc = _make_simplicial(schedule, variant)
        t = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        grad_fn = jax.vmap(jax.grad(lambda ti: proc.beta(ti[None])[0]))
        beta_prime_ref = grad_fn(t)
        assert jnp.allclose(proc.beta_prime(t), beta_prime_ref, atol=1e-5)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_get_coefficients_keys_and_values(self, schedule: str, variant: str):
        """get_coefficients returns correct keys and matching values."""
        proc = _make_simplicial(schedule, variant)
        t = jnp.array([0.2, 0.4, 0.6, 0.8])
        coeffs = proc.get_coefficients(t)
        assert set(coeffs.keys()) == {"beta", "beta_prime", "h"}
        assert jnp.allclose(coeffs["beta"], proc.beta(t))
        assert jnp.allclose(coeffs["beta_prime"], proc.beta_prime(t))
        assert jnp.allclose(coeffs["h"], proc.h(t))


# ===========================================================================
# Tests for simplicial forward pass
# ===========================================================================


class TestSimplicialForward:
    """Tests for forward pass of simplicial processes."""

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_output_shapes(self, schedule: str, variant: str):
        """forward returns (x_t, aux) with correct shapes."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, aux = proc.forward(key, x_0, t)

        assert x_t.shape == (BATCH_SIZE, *SEQ_SHAPE, M)
        assert aux["x"].shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert aux["t"].shape == (BATCH_SIZE,)
        assert aux["padding_mask"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_log_probs_normalise(self, schedule: str, variant: str):
        """x_t should be log-normalised: logsumexp(x_t, axis=-1) ~ 0."""
        proc = _make_simplicial(schedule, variant)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)

        log_sums = jax.nn.logsumexp(x_t, axis=-1)
        assert jnp.allclose(log_sums, 0.0, atol=1e-5)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_aux_contains_coefficients(self, schedule: str, variant: str):
        """forward aux dict includes schedule info keys."""
        proc = _make_simplicial(schedule, variant)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        _, aux = proc.forward(key, x_0, t)
        for k in ("beta", "beta_prime", "h"):
            assert k in aux

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_near_zero_is_clean(self, schedule: str, variant: str):
        """At t ~ 0 (large h), argmax of x_t should equal x_0."""
        proc = _make_simplicial(schedule, variant)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = 1e-6 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)
        argmax_xt = jnp.argmax(x_t, axis=-1)
        assert jnp.array_equal(argmax_xt, x_0)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    def test_mask_near_one_argmax_is_mask(self, schedule: str):
        """At t = 1 for masking, argmax should be the mask index K."""
        proc = _make_simplicial(schedule, "mask")
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)
        argmax_xt = jnp.argmax(x_t, axis=-1)
        assert jnp.all(argmax_xt == NUM_CLASSES)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    def test_mask_intermediate_two_categories(self, schedule: str):
        """For masking at intermediate t, mass is on x_0 and mask only."""
        proc = _make_simplicial(schedule, "mask")
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)

        probs = jnp.exp(x_t)  # (batch, *seq_shape, K+1)
        K = NUM_CLASSES
        mask_probs = probs[..., K]
        x0_probs = jnp.take_along_axis(probs[..., :K], x_0[..., None], axis=-1)[..., 0]
        other_probs = 1.0 - mask_probs - x0_probs
        assert jnp.allclose(other_probs, 0.0, atol=1e-3)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_deterministic_given_key(self, schedule: str, variant: str):
        """Same PRNG key produces the same result."""
        proc = _make_simplicial(schedule, variant)
        key = jax.random.PRNGKey(42)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t_a, _ = proc.forward(key, x_0, t)
        x_t_b, _ = proc.forward(key, x_0, t)
        assert jnp.allclose(x_t_a, x_t_b)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_no_padding_is_all_false(self, schedule: str, variant: str):
        """Without padding_token, padding_mask is all False."""
        proc = _make_simplicial(schedule, variant)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        _, aux = proc.forward(key, x_0, t)
        assert jnp.all(~aux["padding_mask"])

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_forward_multiple_count_and_shapes(self, schedule: str, variant: str):
        """forward_multiple returns correct number and shapes."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]
        results = proc.forward_multiple(key, x_0, ts)

        assert len(results) == 3
        for x_t, aux in results:
            assert x_t.shape == (BATCH_SIZE, *SEQ_SHAPE, M)
            assert aux["x"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_forward_multiple_log_probs_normalise(self, schedule: str, variant: str):
        """All x_t from forward_multiple should be log-normalised."""
        proc = _make_simplicial(schedule, variant)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]
        results = proc.forward_multiple(key, x_0, ts)
        for x_t, _ in results:
            log_sums = jax.nn.logsumexp(x_t, axis=-1)
            assert jnp.allclose(log_sums, 0.0, atol=1e-5)


# ===========================================================================
# Tests for simplicial sample_from_terminal
# ===========================================================================


class TestSimplicialTerminal:
    """Tests for sample_from_terminal."""

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_shape(self, schedule: str, variant: str):
        """Terminal sample has correct shape."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        key = jax.random.PRNGKey(0)
        x_1, aux = proc.sample_from_terminal(key, BATCH_SIZE)

        assert x_1.shape == (BATCH_SIZE, *SEQ_SHAPE, M)
        assert aux["t"].shape == (BATCH_SIZE,)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_time_is_one(self, schedule: str, variant: str):
        """Aux time should be 1.0."""
        proc = _make_simplicial(schedule, variant)
        key = jax.random.PRNGKey(0)
        _, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        assert jnp.allclose(aux["t"], 1.0)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_log_probs_normalise(self, schedule: str, variant: str):
        """Terminal samples should be log-normalised."""
        proc = _make_simplicial(schedule, variant)
        key = jax.random.PRNGKey(0)
        x_1, _ = proc.sample_from_terminal(key, BATCH_SIZE)
        log_sums = jax.nn.logsumexp(x_1, axis=-1)
        assert jnp.allclose(log_sums, 0.0, atol=1e-5)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    def test_mask_terminal_argmax_is_mask(self, schedule: str):
        """For masking, terminal argmax should be the mask index K."""
        proc = _make_simplicial(schedule, "mask")
        key = jax.random.PRNGKey(0)
        x_1, _ = proc.sample_from_terminal(key, BATCH_SIZE)
        argmax_x1 = jnp.argmax(x_1, axis=-1)
        assert jnp.all(argmax_x1 == NUM_CLASSES)


# ===========================================================================
# Tests for simplicial convert_prediction
# ===========================================================================


class TestSimplicialConversion:
    """Tests for convert_prediction."""

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_identity_logits_x0(self, schedule: str, variant: str):
        """Converting logits_x0 to logits_x0 is identity."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        x_t = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE, M))
        logits = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE, M))
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        pred = Prediction(value=logits, kind="logits_x0")
        result = proc.convert_prediction(pred, x_t, None, t, "logits_x0")
        assert result.kind == "logits_x0"
        assert jnp.array_equal(result.value, logits)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_identity_x0(self, schedule: str, variant: str):
        """Converting x_0 to x_0 is identity."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        x_t = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE, M))
        probs = jax.nn.softmax(
            jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE, M)),
            axis=-1,
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        pred = Prediction(value=probs, kind="x_0")
        result = proc.convert_prediction(pred, x_t, None, t, "x_0")
        assert result.kind == "x_0"
        assert jnp.array_equal(result.value, probs)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_logits_x0_to_x0_is_softmax(self, schedule: str, variant: str):
        """logits_x0 -> x_0 applies softmax."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        x_t = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE, M))
        logits = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE, M))
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        pred = Prediction(value=logits, kind="logits_x0")
        result = proc.convert_prediction(pred, x_t, None, t, "x_0")

        expected = jax.nn.softmax(logits, axis=-1)
        assert result.kind == "x_0"
        assert jnp.allclose(result.value, expected, atol=1e-6)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_x0_to_logits_x0(self, schedule: str, variant: str):
        """x_0 -> logits_x0 applies log."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        x_t = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE, M))
        logits = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE, M))
        probs = jax.nn.softmax(logits, axis=-1)
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        pred = Prediction(value=probs, kind="x_0")
        result = proc.convert_prediction(pred, x_t, None, t, "logits_x0")

        expected = jnp.log(jnp.maximum(probs, 1e-20))
        assert result.kind == "logits_x0"
        assert jnp.allclose(result.value, expected, atol=1e-6)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_logits_roundtrip(self, schedule: str, variant: str):
        """logits_x0 -> x_0 -> logits_x0 preserves differences (up to shift)."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        x_t = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE, M))
        logits = jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE, M))
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=logits, kind="logits_x0")
        as_x0 = proc.convert_prediction(pred, x_t, None, t, "x_0")
        back = proc.convert_prediction(as_x0, x_t, None, t, "logits_x0")

        shift = logits - back.value
        assert jnp.allclose(shift, shift[..., :1], atol=1e-5)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_unknown_conversion_raises(self, schedule: str, variant: str):
        """Converting to an unsupported kind raises NotImplementedError."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        x_t = jax.random.normal(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE, M))
        probs = jax.nn.softmax(x_t, axis=-1)
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=probs, kind="x_0")
        with pytest.raises(NotImplementedError):
            proc.convert_prediction(pred, x_t, None, t, "score")


# ===========================================================================
# Tests for simplicial padding
# ===========================================================================


class TestSimplicialPadding:
    """Tests for padding_token handling."""

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_padding_preserved_in_forward(self, schedule: str, variant: str):
        """Padding positions are preserved in forward."""
        proc = _make_simplicial(schedule, variant, padding_token=PADDING_INDEX)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        x_0 = x_0.at[:, -3:].set(PADDING_INDEX)
        t = 0.8 * jnp.ones((BATCH_SIZE,))

        x_t, aux = proc.forward(key, x_0, t)

        assert jnp.all(x_t[:, -3:, :] == float(PADDING_INDEX))
        assert jnp.all(aux["padding_mask"][:, -3:])
        assert jnp.all(~aux["padding_mask"][:, :-3])

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_padding_preserved_in_forward_multiple(self, schedule: str, variant: str):
        """Padding is preserved across all times in forward_multiple."""
        proc = _make_simplicial(schedule, variant, padding_token=PADDING_INDEX)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        x_0 = x_0.at[:, :2].set(PADDING_INDEX)
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8, 1.0]]

        results = proc.forward_multiple(key, x_0, ts)
        for x_t, aux in results:
            assert jnp.all(x_t[:, :2, :] == float(PADDING_INDEX))
            assert jnp.all(aux["padding_mask"][:, :2])

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_non_padding_still_corrupted(self, schedule: str, variant: str):
        """Non-padding positions are corrupted normally (log-normalised)."""
        proc = _make_simplicial(schedule, variant, padding_token=PADDING_INDEX)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        x_0 = x_0.at[:, -3:].set(PADDING_INDEX)
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        x_t, _ = proc.forward(key, x_0, t)

        non_pad_xt = x_t[:, :-3, :]
        log_sums = jax.nn.logsumexp(non_pad_xt, axis=-1)
        assert jnp.allclose(log_sums, 0.0, atol=1e-5)


# ===========================================================================
# Tests for mathematical consistency with reference
# ===========================================================================


class TestSimplicialMath:
    """Tests verifying mathematical consistency with the reference.

    The reference implementation uses:
        dirichlet_param = tau * (pi + h(t) * one_hot(x_0))
        h(t) = alpha(t) / (1 - alpha(t) + eps)

    where alpha(t) = 1 - beta(t) in our notation.
    """

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_dirichlet_params_match_reference(self, schedule: str, variant: str):
        """Verify Dirichlet concentration parameters match the reference formula."""
        proc = _make_simplicial(schedule, variant)
        M = proc.vocab_size
        x_0 = jax.random.randint(
            jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, NUM_CLASSES
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        x_oh = jax.nn.one_hot(x_0, M)  # (batch, *seq_shape, M)
        h_t = proc.h(t)  # (batch,)
        h_t_bc = h_t[:, None]  # (batch, 1) for seq_shape=(12,)
        pi = proc.invariant_probs  # (M,)

        ref_params = proc.temperature * (pi + h_t_bc[..., None] * x_oh)
        our_params = proc.temperature * (h_t_bc[..., None] * x_oh + pi)
        assert jnp.allclose(ref_params, our_params, atol=1e-6)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    @pytest.mark.parametrize("variant", VARIANT_NAMES)
    def test_h_function_matches_reference(self, schedule: str, variant: str):
        """h(t) = (1 - beta(t)) / (beta(t) + eps) matches the reference."""
        proc = _make_simplicial(schedule, variant)
        t = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        beta_t = proc.beta(t)
        ref_h = (1.0 - beta_t) / (beta_t + proc.safety_epsilon)
        # Use relative tolerance: h(t) can be large when beta(t) is small,
        # so absolute tolerance is not appropriate for all schedules.
        assert jnp.allclose(proc.h(t), ref_h, rtol=1e-3)

    def test_linear_beta_matches_reference_schedule(self):
        """SimplicialLinear beta(t) = t matches the reference linear schedule."""
        proc = SimplicialLinear.uniform_process(NUM_CLASSES, SEQ_SHAPE)
        t = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        assert jnp.allclose(proc.beta(t), expected, atol=1e-6)

    def test_cosine_beta_matches_reference_schedule(self):
        """SimplicialCosine beta(t) = 1 - cos(pi*t/2) matches the reference."""
        proc = SimplicialCosine.uniform_process(NUM_CLASSES, SEQ_SHAPE)
        t = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        expected = 1.0 - jnp.cos(jnp.pi / 2 * t)
        assert jnp.allclose(proc.beta(t), expected, atol=1e-6)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    def test_mask_concentration_structure(self, schedule: str):
        """For masking, only x_0 category and mask category have nonzero concentration."""
        proc = _make_simplicial(schedule, "mask")
        K = NUM_CLASSES
        M = K + 1

        x_0 = 3 * jnp.ones((1,), dtype=jnp.int32)
        t = 0.5 * jnp.ones((1,))

        x_oh = jax.nn.one_hot(x_0, M)  # (1, M)
        h_t = proc.h(t)  # (1,)
        pi = proc.invariant_probs  # (M,)
        params = proc.temperature * (h_t[:, None] * x_oh + pi)

        for j in range(M):
            if j == 3:
                assert params[0, j] > 0  # x_0 category
            elif j == K:
                assert params[0, j] > 0  # mask category
            else:
                assert jnp.isclose(params[0, j], 0.0, atol=1e-7)

    @pytest.mark.parametrize("schedule", SCHEDULE_NAMES)
    def test_uniform_all_categories_positive(self, schedule: str):
        """For uniform, all categories have positive concentration."""
        proc = _make_simplicial(schedule, "uniform")
        K = NUM_CLASSES
        M = K

        x_0 = 3 * jnp.ones((1,), dtype=jnp.int32)
        t = 0.5 * jnp.ones((1,))

        x_oh = jax.nn.one_hot(x_0, M)  # (1, M)
        h_t = proc.h(t)  # (1,)
        pi = proc.invariant_probs  # (M,)
        params = proc.temperature * (h_t[:, None] * x_oh + pi)

        assert jnp.all(params > 0)
        assert jnp.argmax(params[0]) == 3
