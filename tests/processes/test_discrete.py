"""Tests for DiscreteCorruptionProcess (diffusionlab.processes.discrete)."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.discrete import (
    DiscreteCorruptionProcess,
    MaskCosine,
    MaskDiffusion,
    MaskLinear,
    UniformCosine,
    UniformLinear,
)
from diffusionlab.typing import (
    DiscreteData,
    Prediction,
    ProbabilisticData,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
VOCAB_SIZE = 8  # K
SEQ_SHAPE = (12,)
PADDING_INDEX = -1

MASK_PROCESS_NAMES = ["mask_linear", "mask_cosine"]
UNIFORM_PROCESS_NAMES = ["uniform_linear", "uniform_cosine"]
ALL_DISCRETE_PROCESS_NAMES = MASK_PROCESS_NAMES + UNIFORM_PROCESS_NAMES


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_discrete_process(
    name: str, padding_token: int | None = None
) -> DiscreteCorruptionProcess:
    """Factory for test discrete process instances."""
    if name == "mask_linear":
        return MaskLinear(VOCAB_SIZE, SEQ_SHAPE, padding_token=padding_token)
    elif name == "mask_cosine":
        return MaskCosine(VOCAB_SIZE, SEQ_SHAPE, padding_token=padding_token)
    elif name == "uniform_linear":
        return UniformLinear(VOCAB_SIZE, SEQ_SHAPE, padding_token=padding_token)
    elif name == "uniform_cosine":
        return UniformCosine(VOCAB_SIZE, SEQ_SHAPE, padding_token=padding_token)
    else:
        raise ValueError(f"Unknown process: {name}")


def _is_mask_process(name: str) -> bool:
    """Return True if the process name corresponds to a masking process."""
    return name in MASK_PROCESS_NAMES


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_mask_posterior(
    probs_x0: ProbabilisticData,
    x_t: DiscreteData,
    mask_index: int,
    beta_s: float,
    beta_t: float,
    K: int,
) -> ProbabilisticData:
    """Slow per-element reference for mask denoising posterior.

    Args:
        probs_x0: Predicted x_0 distribution, shape (batch, seq_len, K).
        x_t: Corrupted tokens, shape (batch, seq_len).
        mask_index: MASK token index.
        beta_s: Scalar beta(s).
        beta_t: Scalar beta(t).
        K: Number of regular classes.

    Returns:
        Posterior over K+1 classes, shape (batch, seq_len, K+1).
    """
    batch_size, seq_len = x_t.shape
    posterior = jnp.zeros((batch_size, seq_len, K + 1))
    for b in range(batch_size):
        for i in range(seq_len):
            if x_t[b, i] == mask_index:
                for j in range(K):
                    posterior = posterior.at[b, i, j].set(
                        (beta_t - beta_s) / beta_t * probs_x0[b, i, j]
                    )
                posterior = posterior.at[b, i, K].set(beta_s / beta_t)
            else:
                posterior = posterior.at[b, i, int(x_t[b, i])].set(1.0)
    return posterior


def _ref_uniform_posterior(
    probs_x0: ProbabilisticData,
    x_t: DiscreteData,
    beta_s: float,
    beta_t: float,
    K: int,
) -> ProbabilisticData:
    """Slow per-element reference for uniform denoising posterior.

    Args:
        probs_x0: Predicted x_0 distribution, shape (batch, seq_len, K).
        x_t: Corrupted tokens, shape (batch, seq_len).
        beta_s: Scalar beta(s).
        beta_t: Scalar beta(t).
        K: Number of classes.

    Returns:
        Posterior over K classes, shape (batch, seq_len, K).
    """
    batch_size, seq_len = x_t.shape
    gamma = (1.0 - beta_t) / (1.0 - beta_s)
    posterior = jnp.zeros((batch_size, seq_len, K))
    for b in range(batch_size):
        for i in range(seq_len):
            unnorm = jnp.zeros(K)
            for j in range(K):
                q_xt = gamma * float(x_t[b, i] == j) + (1 - gamma) / K
                q_xs = (1.0 - beta_s) * probs_x0[b, i, j] + beta_s / K
                unnorm = unnorm.at[j].set(q_xt * q_xs)
            posterior = posterior.at[b, i].set(unnorm / jnp.sum(unnorm))
    return posterior


# ===========================================================================
# Tests for discrete schedule functions
# ===========================================================================


class TestDiscreteSchedule:
    """Tests for beta / beta_prime of all concrete discrete processes."""

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_beta_at_zero(self, name: str):
        """beta(0) = 0."""
        proc = _make_discrete_process(name)
        t0 = jnp.zeros((BATCH_SIZE,))
        assert jnp.allclose(proc.beta(t0), 0.0, atol=1e-6)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_beta_at_one(self, name: str):
        """beta(1) = 1."""
        proc = _make_discrete_process(name)
        t1 = jnp.ones((BATCH_SIZE,))
        assert jnp.allclose(proc.beta(t1), 1.0, atol=1e-6)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_beta_monotonically_increasing(self, name: str):
        """beta should be non-decreasing as t increases."""
        proc = _make_discrete_process(name)
        t = jnp.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        diffs = jnp.diff(proc.beta(t))
        assert jnp.all(diffs >= -1e-7)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_beta_prime_matches_jax_grad(self, name: str):
        """beta'(t) matches jax.grad of the scalar beta function."""
        proc = _make_discrete_process(name)
        t = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        grad_fn = jax.vmap(jax.grad(lambda ti: proc.beta(ti[None])[0]))
        beta_prime_ref = grad_fn(t)
        assert jnp.allclose(proc.beta_prime(t), beta_prime_ref, atol=1e-5)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_get_coefficients_keys_and_values(self, name: str):
        """get_coefficients returns correct keys and matching values."""
        proc = _make_discrete_process(name)
        t = jnp.array([0.2, 0.4, 0.6, 0.8])
        coeffs = proc.get_coefficients(t)
        assert set(coeffs.keys()) == {"beta", "beta_prime"}
        assert jnp.allclose(coeffs["beta"], proc.beta(t))
        assert jnp.allclose(coeffs["beta_prime"], proc.beta_prime(t))


# ===========================================================================
# Tests for mask forward pass
# ===========================================================================


class TestMaskForward:
    """Tests for forward pass of MaskDiffusion subclasses."""

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_output_shapes(self, name: str):
        """forward returns (x_t, aux) with correct shapes."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, aux = proc.forward(key, x_0, t)
        assert x_t.shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert aux["x"].shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert aux["t"].shape == (BATCH_SIZE,)
        assert aux["corrupted_mask"].shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert aux["padding_mask"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_values_are_valid(self, name: str):
        """x_t values are either original tokens or MASK."""
        proc = _make_discrete_process(name)
        assert isinstance(proc, MaskDiffusion)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)
        is_original = x_t == x_0
        is_mask = x_t == proc.mask_index
        assert jnp.all(is_original | is_mask)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_aux_contains_coefficients(self, name: str):
        """forward aux dict includes schedule info keys."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        _, aux = proc.forward(key, x_0, t)
        assert "beta" in aux
        assert "beta_prime" in aux

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_near_zero_is_clean(self, name: str):
        """At t ~ 0, x_t ~ x_0 (beta(0) = 0, almost nothing masked)."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 1e-6 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)
        assert jnp.array_equal(x_t, x_0)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_near_one_is_all_mask(self, name: str):
        """At t = 1, x_t should be all MASK (beta(1) = 1)."""
        proc = _make_discrete_process(name)
        assert isinstance(proc, MaskDiffusion)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)
        assert jnp.all(x_t == proc.mask_index)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_deterministic_given_key(self, name: str):
        """Same PRNG key produces same result."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(42)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t_a, aux_a = proc.forward(key, x_0, t)
        x_t_b, aux_b = proc.forward(key, x_0, t)
        assert jnp.array_equal(x_t_a, x_t_b)
        assert jnp.array_equal(aux_a["corrupted_mask"], aux_b["corrupted_mask"])

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_no_padding_is_all_false(self, name: str):
        """Without padding_token, padding_mask should be all False."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        _, aux = proc.forward(key, x_0, t)
        assert jnp.all(~aux["padding_mask"])

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_forward_multiple_count_and_shapes(self, name: str):
        """forward_multiple returns correct number and shapes."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]
        results = proc.forward_multiple(key, x_0, ts)
        assert len(results) == 3
        for x_t, aux in results:
            assert x_t.shape == (BATCH_SIZE, *SEQ_SHAPE)
            assert aux["corrupted_mask"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_forward_multiple_monotonic_corruption(self, name: str):
        """If a position is corrupted at t_i, it is also corrupted at t_j > t_i."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.1, 0.3, 0.5, 0.7, 0.9]]
        results = proc.forward_multiple(key, x_0, ts)
        for i in range(len(results) - 1):
            corrupted_earlier = results[i][1]["corrupted_mask"]
            corrupted_later = results[i + 1][1]["corrupted_mask"]
            assert jnp.all(corrupted_later | ~corrupted_earlier)


# ===========================================================================
# Tests for uniform forward pass
# ===========================================================================


class TestUniformForward:
    """Tests for forward pass of UniformDiffusion subclasses."""

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_output_shapes(self, name: str):
        """forward returns (x_t, aux) with correct shapes."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, aux = proc.forward(key, x_0, t)
        assert x_t.shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert aux["x"].shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert aux["t"].shape == (BATCH_SIZE,)
        assert aux["corrupted_mask"].shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert aux["padding_mask"].shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_values_in_range(self, name: str):
        """x_t values are in {0, ..., K-1}."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)
        assert jnp.all(x_t >= 0)
        assert jnp.all(x_t < VOCAB_SIZE)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_near_zero_is_clean(self, name: str):
        """At t ~ 0, x_t ~ x_0."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 1e-6 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(key, x_0, t)
        assert jnp.array_equal(x_t, x_0)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_deterministic_given_key(self, name: str):
        """Same PRNG key produces same result."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(42)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t_a, _ = proc.forward(key, x_0, t)
        x_t_b, _ = proc.forward(key, x_0, t)
        assert jnp.array_equal(x_t_a, x_t_b)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_forward_multiple_count_and_shapes(self, name: str):
        """forward_multiple returns correct number and shapes."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8]]
        results = proc.forward_multiple(key, x_0, ts)
        assert len(results) == 3
        for x_t, aux in results:
            assert x_t.shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_forward_multiple_monotonic_corruption(self, name: str):
        """Monotonic corruption: corrupted at t_i implies corrupted at t_j > t_i."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.1, 0.3, 0.5, 0.7, 0.9]]
        results = proc.forward_multiple(key, x_0, ts)
        for i in range(len(results) - 1):
            corrupted_earlier = results[i][1]["corrupted_mask"]
            corrupted_later = results[i + 1][1]["corrupted_mask"]
            assert jnp.all(corrupted_later | ~corrupted_earlier)


# ===========================================================================
# Tests for mask sample_from_terminal
# ===========================================================================


class TestMaskTerminal:
    """Tests for sample_from_terminal of MaskDiffusion."""

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_shape(self, name: str):
        """Terminal sample has correct shape."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_1, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        assert x_1.shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_all_mask(self, name: str):
        """All terminal tokens are MASK."""
        proc = _make_discrete_process(name)
        assert isinstance(proc, MaskDiffusion)
        key = jax.random.PRNGKey(0)
        x_1, _ = proc.sample_from_terminal(key, BATCH_SIZE)
        assert jnp.all(x_1 == proc.mask_index)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_time_is_one(self, name: str):
        """Aux time should be 1.0."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        _, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        assert jnp.allclose(aux["t"], 1.0)


# ===========================================================================
# Tests for uniform sample_from_terminal
# ===========================================================================


class TestUniformTerminal:
    """Tests for sample_from_terminal of UniformDiffusion."""

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_shape(self, name: str):
        """Terminal sample has correct shape."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_1, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        assert x_1.shape == (BATCH_SIZE, *SEQ_SHAPE)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_values_in_range(self, name: str):
        """Terminal tokens are in {0, ..., K-1}."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_1, _ = proc.sample_from_terminal(key, BATCH_SIZE)
        assert jnp.all(x_1 >= 0)
        assert jnp.all(x_1 < VOCAB_SIZE)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_time_is_one(self, name: str):
        """Aux time should be 1.0."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        _, aux = proc.sample_from_terminal(key, BATCH_SIZE)
        assert jnp.allclose(aux["t"], 1.0)


# ===========================================================================
# Tests for mask denoising posterior
# ===========================================================================


class TestMaskPosterior:
    """Tests for the denoising posterior of MaskDiffusion."""

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_reference_correctness(self, name: str):
        """Vectorised posterior matches slow per-element reference."""
        proc = _make_discrete_process(name)
        assert isinstance(proc, MaskDiffusion)
        K = VOCAB_SIZE

        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(key, (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.6 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)

        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)
        beta_s_val = float(proc.beta(s)[0])
        beta_t_val = float(proc.beta(t)[0])
        ref = _ref_mask_posterior(
            probs_x0, x_t, proc.mask_index, beta_s_val, beta_t_val, K
        )
        assert jnp.allclose(posterior, ref, atol=1e-5)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_sums_to_one(self, name: str):
        """Posterior probabilities sum to 1 along the class axis."""
        proc = _make_discrete_process(name)
        assert isinstance(proc, MaskDiffusion)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.6 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)

        assert jnp.allclose(jnp.sum(posterior, axis=-1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_unmasked_is_deterministic(self, name: str):
        """Unmasked positions should have one-hot posterior on x_t."""
        proc = _make_discrete_process(name)
        assert isinstance(proc, MaskDiffusion)
        K = VOCAB_SIZE

        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        s = 0.1 * jnp.ones((BATCH_SIZE,))

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)

        expected = jax.nn.one_hot(x_t, K + 1)
        assert jnp.allclose(posterior, expected, atol=1e-6)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_at_s_zero_recovers_x0_dist(self, name: str):
        """At s=0, masked positions: P(x_s=j) = P(x_0=j) for regular j, P(MASK)=0."""
        proc = _make_discrete_process(name)
        assert isinstance(proc, MaskDiffusion)
        K = VOCAB_SIZE

        x_t = jnp.full((BATCH_SIZE, *SEQ_SHAPE), proc.mask_index, dtype=jnp.int32)
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        s = jnp.zeros((BATCH_SIZE,))

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)

        assert jnp.allclose(posterior[..., :K], probs_x0, atol=1e-5)
        assert jnp.allclose(posterior[..., K], 0.0, atol=1e-5)

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_at_s_equals_t_is_one_hot_xt(self, name: str):
        """At s = t, posterior = one_hot(x_t)."""
        proc = _make_discrete_process(name)
        assert isinstance(proc, MaskDiffusion)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        posterior = proc._denoising_posterior_probs(probs_x0, x_t, t, t)

        expected = jax.nn.one_hot(x_t, K + 1)
        assert jnp.allclose(posterior, expected, atol=1e-5)


# ===========================================================================
# Tests for uniform denoising posterior
# ===========================================================================


class TestUniformPosterior:
    """Tests for the denoising posterior of UniformDiffusion."""

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_reference_correctness(self, name: str):
        """Vectorised posterior matches slow per-element reference."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.6 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)

        beta_s_val = float(proc.beta(s)[0])
        beta_t_val = float(proc.beta(t)[0])
        ref = _ref_uniform_posterior(probs_x0, x_t, beta_s_val, beta_t_val, K)
        assert jnp.allclose(posterior, ref, atol=1e-5)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_sums_to_one(self, name: str):
        """Posterior probabilities sum to 1."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.6 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)

        assert jnp.allclose(jnp.sum(posterior, axis=-1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_at_s_equals_t_is_one_hot_xt(self, name: str):
        """At s = t, posterior = one_hot(x_t)."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        posterior = proc._denoising_posterior_probs(probs_x0, x_t, t, t)

        expected = jax.nn.one_hot(x_t, K)
        assert jnp.allclose(posterior, expected, atol=1e-5)

    @pytest.mark.parametrize("name", UNIFORM_PROCESS_NAMES)
    def test_at_s_zero_from_terminal(self, name: str):
        """At s=0, t=1: posterior should equal the predicted x_0 distribution."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE

        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = jnp.ones((BATCH_SIZE,))
        s = jnp.zeros((BATCH_SIZE,))

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)

        assert jnp.allclose(posterior, probs_x0, atol=1e-5)


# ===========================================================================
# Tests for discrete convert_prediction
# ===========================================================================


class TestDiscreteConversion:
    """Tests for convert_prediction of discrete processes."""

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_identity_x0(self, name: str):
        """Converting x_0 to x_0 is identity."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        logits = jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=probs_x0, kind="x_0")
        result = proc.convert_prediction(pred, x_t, None, t, "x_0")
        assert result.kind == "x_0"
        assert jnp.array_equal(result.value, probs_x0)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_identity_xs(self, name: str):
        """Converting x_s to x_s is identity."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        out_classes = K + 1 if _is_mask_process(name) else K
        dummy_xs = jax.nn.softmax(
            jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, out_classes)),
            axis=-1,
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=dummy_xs, kind="x_s")
        result = proc.convert_prediction(pred, x_t, None, t, "x_s")
        assert result.kind == "x_s"
        assert jnp.array_equal(result.value, dummy_xs)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_identity_logits_xs(self, name: str):
        """Converting logits_x_s to logits_x_s is identity."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        out_classes = K + 1 if _is_mask_process(name) else K
        logits_xs = jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, out_classes))
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=logits_xs, kind="logits_x_s")
        result = proc.convert_prediction(pred, x_t, None, t, "logits_x_s")
        assert result.kind == "logits_x_s"
        assert jnp.array_equal(result.value, logits_xs)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_identity_logits_x0(self, name: str):
        """Converting logits_x0 to logits_x0 is identity."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        logits = jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, K))
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=logits, kind="logits_x0")
        result = proc.convert_prediction(pred, x_t, None, t, "logits_x0")
        assert result.kind == "logits_x0"
        assert jnp.array_equal(result.value, logits)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_x0_to_xs(self, name: str):
        """x_0 -> x_s conversion matches direct posterior computation."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.6 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)

        pred = Prediction(value=probs_x0, kind="x_0")
        result = proc.convert_prediction(pred, x_t, s, t, "x_s")

        expected = proc._denoising_posterior_probs(probs_x0, x_t, s, t)
        assert result.kind == "x_s"
        assert jnp.allclose(result.value, expected, atol=1e-6)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_logits_x0_to_x0_is_softmax(self, name: str):
        """logits_x0 -> x_0 applies softmax."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        logits = jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, K))
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=logits, kind="logits_x0")
        result = proc.convert_prediction(pred, x_t, None, t, "x_0")

        expected = jax.nn.softmax(logits, axis=-1)
        assert result.kind == "x_0"
        assert jnp.allclose(result.value, expected, atol=1e-6)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_logits_x0_to_xs(self, name: str):
        """logits_x0 -> x_s matches softmax then posterior."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.6 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)

        pred = Prediction(value=logits, kind="logits_x0")
        result = proc.convert_prediction(pred, x_t, s, t, "x_s")

        expected = proc._denoising_posterior_probs(probs_x0, x_t, s, t)
        assert result.kind == "x_s"
        assert jnp.allclose(result.value, expected, atol=1e-5)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_x0_to_logits_xs(self, name: str):
        """x_0 -> logits_x_s matches posterior then log."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.6 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)

        pred = Prediction(value=probs_x0, kind="x_0")
        result = proc.convert_prediction(pred, x_t, s, t, "logits_x_s")

        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)
        expected = jnp.log(jnp.maximum(posterior, 1e-20))
        assert result.kind == "logits_x_s"
        assert jnp.allclose(result.value, expected, atol=1e-6)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_logits_x0_to_logits_xs(self, name: str):
        """logits_x0 -> logits_x_s matches softmax, posterior, then log."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE

        x_0 = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        t = 0.6 * jnp.ones((BATCH_SIZE,))
        s = 0.2 * jnp.ones((BATCH_SIZE,))
        x_t, _ = proc.forward(jax.random.PRNGKey(1), x_0, t)

        logits = jax.random.normal(jax.random.PRNGKey(2), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)

        pred = Prediction(value=logits, kind="logits_x0")
        result = proc.convert_prediction(pred, x_t, s, t, "logits_x_s")

        posterior = proc._denoising_posterior_probs(probs_x0, x_t, s, t)
        expected = jnp.log(jnp.maximum(posterior, 1e-20))
        assert result.kind == "logits_x_s"
        assert jnp.allclose(result.value, expected, atol=1e-5)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_logits_xs_to_xs_is_softmax(self, name: str):
        """logits_x_s -> x_s applies softmax."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        out_classes = K + 1 if _is_mask_process(name) else K
        logits_xs = jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, out_classes))
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=logits_xs, kind="logits_x_s")
        result = proc.convert_prediction(pred, x_t, None, t, "x_s")

        expected = jax.nn.softmax(logits_xs, axis=-1)
        assert result.kind == "x_s"
        assert jnp.allclose(result.value, expected, atol=1e-6)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_xs_to_logits_xs(self, name: str):
        """x_s -> logits_x_s applies log."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        out_classes = K + 1 if _is_mask_process(name) else K
        probs_xs = jax.nn.softmax(
            jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, out_classes)),
            axis=-1,
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=probs_xs, kind="x_s")
        result = proc.convert_prediction(pred, x_t, None, t, "logits_x_s")

        expected = jnp.log(jnp.maximum(probs_xs, 1e-20))
        assert result.kind == "logits_x_s"
        assert jnp.allclose(result.value, expected, atol=1e-6)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_x0_to_logits_x0(self, name: str):
        """x_0 -> logits_x0 applies log."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        logits = jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=probs_x0, kind="x_0")
        result = proc.convert_prediction(pred, x_t, None, t, "logits_x0")

        expected = jnp.log(jnp.maximum(probs_x0, 1e-20))
        assert result.kind == "logits_x0"
        assert jnp.allclose(result.value, expected, atol=1e-6)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_logits_roundtrip(self, name: str):
        """logits_x0 -> x_0 -> logits_x0 roundtrip (up to constant shift)."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        logits = jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, K))
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=logits, kind="logits_x0")
        as_x0 = proc.convert_prediction(pred, x_t, None, t, "x_0")
        back = proc.convert_prediction(as_x0, x_t, None, t, "logits_x0")

        shift = logits - back.value
        assert jnp.allclose(shift, shift[..., :1], atol=1e-5), (
            "logit differences should be preserved up to a constant"
        )

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_unknown_conversion_raises(self, name: str):
        """Converting to an unsupported kind raises NotImplementedError."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        logits = jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, K))
        probs_x0 = jax.nn.softmax(logits, axis=-1)
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=probs_x0, kind="x_0")
        with pytest.raises(NotImplementedError):
            proc.convert_prediction(pred, x_t, None, t, "score")

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_xs_to_x0_raises(self, name: str):
        """Converting x_s back to x_0 is not supported and raises."""
        proc = _make_discrete_process(name)
        K = VOCAB_SIZE
        x_t = jax.random.randint(jax.random.PRNGKey(0), (BATCH_SIZE, *SEQ_SHAPE), 0, K)
        out_classes = K + 1 if _is_mask_process(name) else K
        dummy_xs = jax.nn.softmax(
            jax.random.normal(jax.random.PRNGKey(1), (*x_t.shape, out_classes)),
            axis=-1,
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))

        pred = Prediction(value=dummy_xs, kind="x_s")
        with pytest.raises(NotImplementedError):
            proc.convert_prediction(pred, x_t, None, t, "x_0")


# ===========================================================================
# Tests for discrete padding
# ===========================================================================


class TestDiscretePadding:
    """Tests for padding_token / is_padding handling."""

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_padding_preserved_in_forward(self, name: str):
        """Padding positions are never corrupted in forward."""
        proc = _make_discrete_process(name, padding_token=PADDING_INDEX)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        x_0 = x_0.at[:, -3:].set(PADDING_INDEX)
        t = 0.8 * jnp.ones((BATCH_SIZE,))

        x_t, aux = proc.forward(key, x_0, t)

        assert jnp.all(x_t[:, -3:] == PADDING_INDEX)
        assert jnp.all(~aux["corrupted_mask"][:, -3:])
        assert jnp.all(aux["padding_mask"][:, -3:])
        assert jnp.all(~aux["padding_mask"][:, :-3])

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_padding_preserved_in_forward_multiple(self, name: str):
        """Padding is preserved across all times in forward_multiple."""
        proc = _make_discrete_process(name, padding_token=PADDING_INDEX)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        x_0 = x_0.at[:, :2].set(PADDING_INDEX)
        ts = [jnp.full((BATCH_SIZE,), v) for v in [0.2, 0.5, 0.8, 1.0]]

        results = proc.forward_multiple(key, x_0, ts)
        for x_t, aux in results:
            assert jnp.all(x_t[:, :2] == PADDING_INDEX)
            assert jnp.all(~aux["corrupted_mask"][:, :2])
            assert jnp.all(aux["padding_mask"][:, :2])

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_padding_at_t_one(self, name: str):
        """Even at t=1 (full corruption), padding is preserved."""
        proc = _make_discrete_process(name, padding_token=PADDING_INDEX)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        x_0 = x_0.at[:, 5].set(PADDING_INDEX)
        t = jnp.ones((BATCH_SIZE,))

        x_t, _ = proc.forward(key, x_0, t)
        assert jnp.all(x_t[:, 5] == PADDING_INDEX)

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_no_padding_is_all_false(self, name: str):
        """Without padding_token, padding_mask is all False."""
        proc = _make_discrete_process(name)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        t = 0.5 * jnp.ones((BATCH_SIZE,))
        _, aux = proc.forward(key, x_0, t)
        assert jnp.all(~aux["padding_mask"])

    @pytest.mark.parametrize("name", ALL_DISCRETE_PROCESS_NAMES)
    def test_non_padding_still_corrupted(self, name: str):
        """Non-padding positions are still corrupted normally."""
        proc = _make_discrete_process(name, padding_token=PADDING_INDEX)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        x_0 = x_0.at[:, -3:].set(PADDING_INDEX)
        t = jnp.ones((BATCH_SIZE,))

        x_t, aux = proc.forward(key, x_0, t)
        assert jnp.all(aux["corrupted_mask"][:, :-3])

    @pytest.mark.parametrize("name", MASK_PROCESS_NAMES)
    def test_mask_non_padding_gets_mask_at_t1(self, name: str):
        """At t=1, non-padding positions become MASK."""
        proc = _make_discrete_process(name, padding_token=PADDING_INDEX)
        assert isinstance(proc, MaskDiffusion)
        key = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(
            jax.random.PRNGKey(1), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
        )
        x_0 = x_0.at[:, -3:].set(PADDING_INDEX)
        t = jnp.ones((BATCH_SIZE,))

        x_t, _ = proc.forward(key, x_0, t)
        assert jnp.all(x_t[:, :-3] == proc.mask_index)
        assert jnp.all(x_t[:, -3:] == PADDING_INDEX)
