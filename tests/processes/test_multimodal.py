"""Tests for MultimodalCorruptionProcess (diffusionlab.processes.base)."""

import jax
import jax.numpy as jnp

from diffusionlab.processes.base import MultimodalCorruptionProcess
from diffusionlab.processes.discrete import MaskLinear
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
VOCAB_SIZE = 8
SEQ_SHAPE = (12,)


# ---------------------------------------------------------------------------
# Minimal reference implementations for multimodal integration
# ---------------------------------------------------------------------------


class SimpleFlow(InterpolationProcess):
    """Minimal flow matching for multimodal integration tests.

    Attributes:
        dim: Dimensionality of data (excluding batch).
    """

    def __init__(self, dim: int):
        self.dim = dim

    def alpha(self, t: Time) -> Scalar:
        return 1.0 - t

    def alpha_prime(self, t: Time) -> Scalar:
        return -jnp.ones_like(t)

    def sigma(self, t: Time) -> Scalar:
        return t

    def sigma_prime(self, t: Time) -> Scalar:
        return jnp.ones_like(t)

    def logsnr_inverse(self, lam: Scalar) -> Time:
        return 1.0 / (1.0 + jnp.exp(lam / 2.0))

    def _sample_from_source(self, key: PRNGKey, batch_size: int) -> ContinuousData:
        return jax.random.normal(key, (batch_size, self.dim))


class SimpleVP(InterpolationProcess):
    """Minimal VP process for multimodal integration tests.

    Attributes:
        dim: Dimensionality of data (excluding batch).
    """

    def __init__(self, dim: int):
        self.dim = dim

    def alpha(self, t: Time) -> Scalar:
        return jnp.cos(jnp.pi / 2 * t)

    def alpha_prime(self, t: Time) -> Scalar:
        return -jnp.pi / 2 * jnp.sin(jnp.pi / 2 * t)

    def sigma(self, t: Time) -> Scalar:
        return jnp.sin(jnp.pi / 2 * t)

    def sigma_prime(self, t: Time) -> Scalar:
        return jnp.pi / 2 * jnp.cos(jnp.pi / 2 * t)

    def logsnr_inverse(self, lam: Scalar) -> Time:
        return 2.0 / jnp.pi * jnp.arctan(jnp.exp(-lam / 2.0))

    def _sample_from_source(self, key: PRNGKey, batch_size: int) -> ContinuousData:
        return jax.random.normal(key, (batch_size, self.dim))


# ===========================================================================
# Tests for MultimodalCorruptionProcess forward pass
# ===========================================================================


class TestMultimodalForward:
    """Tests for forward and forward_multiple of MultimodalCorruptionProcess."""

    def test_forward_shape_and_structure(self):
        """forward returns pytree-structured (x_t, aux) matching modality dict."""
        procs = {
            "image": SimpleFlow(dim=DATA_DIM),
            "text": SimpleVP(dim=DATA_DIM * 2),
        }
        multi = MultimodalCorruptionProcess(procs)
        key = jax.random.PRNGKey(0)
        x = {
            "image": jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM)),
            "text": jax.random.normal(
                jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM * 2)
            ),
        }
        t = {
            "image": 0.3 * jnp.ones((BATCH_SIZE,)),
            "text": 0.5 * jnp.ones((BATCH_SIZE,)),
        }

        x_t, aux = multi.forward(key, x, t)

        assert set(x_t.keys()) == {"image", "text"}
        assert x_t["image"].shape == (BATCH_SIZE, DATA_DIM)
        assert x_t["text"].shape == (BATCH_SIZE, DATA_DIM * 2)
        assert set(aux.keys()) == {"image", "text"}

    def test_forward_correctness(self):
        """Multimodal forward matches per-modality interpolation formula."""
        procs = {
            "image": SimpleFlow(dim=DATA_DIM),
            "text": SimpleVP(dim=DATA_DIM * 2),
        }
        multi = MultimodalCorruptionProcess(procs)
        key = jax.random.PRNGKey(0)
        x = {
            "image": jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM)),
            "text": jax.random.normal(
                jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM * 2)
            ),
        }
        t = {
            "image": 0.3 * jnp.ones((BATCH_SIZE,)),
            "text": 0.5 * jnp.ones((BATCH_SIZE,)),
        }

        x_t, aux = multi.forward(key, x, t)

        for mod in ["image", "text"]:
            z_mod = aux[mod]["z"]
            alpha_mod = bcast_right(procs[mod].alpha(t[mod]), x[mod].ndim)
            sigma_mod = bcast_right(procs[mod].sigma(t[mod]), x[mod].ndim)
            expected = alpha_mod * x[mod] + sigma_mod * z_mod
            assert jnp.allclose(x_t[mod], expected, atol=1e-6)

    def test_forward_multiple_count_and_shapes(self):
        """forward_multiple returns correct list length and shapes."""
        procs = {
            "image": SimpleFlow(dim=DATA_DIM),
            "text": SimpleVP(dim=DATA_DIM * 2),
        }
        multi = MultimodalCorruptionProcess(procs)
        key = jax.random.PRNGKey(0)
        x = {
            "image": jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM)),
            "text": jax.random.normal(
                jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM * 2)
            ),
        }
        ts = [
            {
                "image": jnp.full((BATCH_SIZE,), 0.2),
                "text": jnp.full((BATCH_SIZE,), 0.3),
            },
            {
                "image": jnp.full((BATCH_SIZE,), 0.5),
                "text": jnp.full((BATCH_SIZE,), 0.7),
            },
        ]

        results = multi.forward_multiple(key, x, ts)

        assert len(results) == 2
        for x_t, aux in results:
            assert set(x_t.keys()) == {"image", "text"}
            assert x_t["image"].shape == (BATCH_SIZE, DATA_DIM)
            assert x_t["text"].shape == (BATCH_SIZE, DATA_DIM * 2)

    def test_forward_multiple_shared_noise(self):
        """forward_multiple uses the same z within each modality across times."""
        procs = {
            "image": SimpleFlow(dim=DATA_DIM),
            "text": SimpleVP(dim=DATA_DIM * 2),
        }
        multi = MultimodalCorruptionProcess(procs)
        key = jax.random.PRNGKey(0)
        x = {
            "image": jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM)),
            "text": jax.random.normal(
                jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM * 2)
            ),
        }
        ts = [
            {
                "image": jnp.full((BATCH_SIZE,), 0.2),
                "text": jnp.full((BATCH_SIZE,), 0.3),
            },
            {
                "image": jnp.full((BATCH_SIZE,), 0.5),
                "text": jnp.full((BATCH_SIZE,), 0.7),
            },
        ]

        results = multi.forward_multiple(key, x, ts)

        for mod in ["image", "text"]:
            z_first = results[0][1][mod]["z"]
            for _, aux in results[1:]:
                assert jnp.allclose(aux[mod]["z"], z_first)

    def test_different_modalities_get_different_keys(self):
        """Different modalities should receive different PRNG keys."""
        procs = {
            "a": SimpleFlow(dim=DATA_DIM),
            "b": SimpleFlow(dim=DATA_DIM),
        }
        multi = MultimodalCorruptionProcess(procs)
        key = jax.random.PRNGKey(0)
        x = {
            "a": jnp.zeros((BATCH_SIZE, DATA_DIM)),
            "b": jnp.zeros((BATCH_SIZE, DATA_DIM)),
        }
        t = {
            "a": 0.5 * jnp.ones((BATCH_SIZE,)),
            "b": 0.5 * jnp.ones((BATCH_SIZE,)),
        }

        x_t, aux = multi.forward(key, x, t)

        # Since x=0, x_t = sigma(t)*z, so different z means different x_t
        assert not jnp.allclose(aux["a"]["z"], aux["b"]["z"])


# ===========================================================================
# Tests for MultimodalCorruptionProcess sample_from_terminal
# ===========================================================================


class TestMultimodalTerminal:
    """Tests for sample_from_terminal of MultimodalCorruptionProcess."""

    def test_shape(self):
        """sample_from_terminal returns pytree-structured (x_1, aux)."""
        procs = {
            "image": SimpleFlow(dim=DATA_DIM),
            "text": SimpleVP(dim=DATA_DIM * 2),
        }
        multi = MultimodalCorruptionProcess(procs)
        key = jax.random.PRNGKey(0)

        x_1, aux = multi.sample_from_terminal(key, BATCH_SIZE)

        assert set(x_1.keys()) == {"image", "text"}
        assert x_1["image"].shape == (BATCH_SIZE, DATA_DIM)
        assert x_1["text"].shape == (BATCH_SIZE, DATA_DIM * 2)


# ===========================================================================
# Tests for MultimodalCorruptionProcess convert_prediction
# ===========================================================================


class TestMultimodalConversion:
    """Tests for convert_prediction of MultimodalCorruptionProcess."""

    def test_convert_prediction_without_s(self):
        """convert_prediction dispatches to each modality (s=None case)."""
        procs = {
            "image": SimpleFlow(dim=DATA_DIM),
            "text": SimpleVP(dim=DATA_DIM * 2),
        }
        multi = MultimodalCorruptionProcess(procs)
        x_0 = {
            "image": jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM)),
            "text": jax.random.normal(
                jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM * 2)
            ),
        }
        z = {
            "image": jax.random.normal(jax.random.PRNGKey(3), (BATCH_SIZE, DATA_DIM)),
            "text": jax.random.normal(
                jax.random.PRNGKey(4), (BATCH_SIZE, DATA_DIM * 2)
            ),
        }
        t = {
            "image": 0.4 * jnp.ones((BATCH_SIZE,)),
            "text": 0.6 * jnp.ones((BATCH_SIZE,)),
        }

        x_t = {}
        expected_z = {}
        for mod in ["image", "text"]:
            alpha_mod = bcast_right(procs[mod].alpha(t[mod]), x_0[mod].ndim)
            sigma_mod = bcast_right(procs[mod].sigma(t[mod]), x_0[mod].ndim)
            x_t[mod] = alpha_mod * x_0[mod] + sigma_mod * z[mod]
            expected_z[mod] = z[mod]

        prediction = {
            "image": Prediction(value=x_0["image"], kind="x_0"),
            "text": Prediction(value=x_0["text"], kind="x_0"),
        }
        to_kind = {"image": "z", "text": "z"}

        result = multi.convert_prediction(prediction, x_t, None, t, to_kind)

        for mod in ["image", "text"]:
            assert result[mod].kind == "z"
            assert jnp.allclose(result[mod].value, expected_z[mod], atol=1e-5)

    def test_convert_prediction_with_s(self):
        """convert_prediction dispatches to each modality (s provided)."""
        procs = {
            "image": SimpleFlow(dim=DATA_DIM),
            "text": SimpleVP(dim=DATA_DIM * 2),
        }
        multi = MultimodalCorruptionProcess(procs)
        x_0 = {
            "image": jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, DATA_DIM)),
            "text": jax.random.normal(
                jax.random.PRNGKey(2), (BATCH_SIZE, DATA_DIM * 2)
            ),
        }
        z = {
            "image": jax.random.normal(jax.random.PRNGKey(3), (BATCH_SIZE, DATA_DIM)),
            "text": jax.random.normal(
                jax.random.PRNGKey(4), (BATCH_SIZE, DATA_DIM * 2)
            ),
        }
        t = {
            "image": 0.5 * jnp.ones((BATCH_SIZE,)),
            "text": 0.6 * jnp.ones((BATCH_SIZE,)),
        }
        s = {
            "image": 0.2 * jnp.ones((BATCH_SIZE,)),
            "text": 0.3 * jnp.ones((BATCH_SIZE,)),
        }

        x_t = {}
        expected_xs = {}
        for mod in ["image", "text"]:
            ndim = x_0[mod].ndim
            alpha_t = bcast_right(procs[mod].alpha(t[mod]), ndim)
            sigma_t = bcast_right(procs[mod].sigma(t[mod]), ndim)
            x_t[mod] = alpha_t * x_0[mod] + sigma_t * z[mod]
            alpha_s = bcast_right(procs[mod].alpha(s[mod]), ndim)
            sigma_s = bcast_right(procs[mod].sigma(s[mod]), ndim)
            expected_xs[mod] = alpha_s * x_0[mod] + sigma_s * z[mod]

        prediction = {
            "image": Prediction(value=x_0["image"], kind="x_0"),
            "text": Prediction(value=x_0["text"], kind="x_0"),
        }
        to_kind = {"image": "x_s", "text": "x_s"}

        result = multi.convert_prediction(prediction, x_t, s, t, to_kind)

        for mod in ["image", "text"]:
            assert result[mod].kind == "x_s"
            assert jnp.allclose(result[mod].value, expected_xs[mod], atol=1e-5)


# ===========================================================================
# Tests for MultimodalCorruptionProcess get_coefficients
# ===========================================================================


class TestMultimodalScheduleInfo:
    """Tests for get_coefficients of MultimodalCorruptionProcess."""

    def test_get_coefficients(self):
        """get_coefficients returns pytree matching modality structure."""
        procs = {
            "image": SimpleFlow(dim=DATA_DIM),
            "text": SimpleVP(dim=DATA_DIM * 2),
        }
        multi = MultimodalCorruptionProcess(procs)
        t = {
            "image": 0.3 * jnp.ones((BATCH_SIZE,)),
            "text": 0.5 * jnp.ones((BATCH_SIZE,)),
        }

        coeffs = multi.get_coefficients(t)

        assert set(coeffs.keys()) == {"image", "text"}
        for mod in ["image", "text"]:
            ref_coeffs = procs[mod].get_coefficients(t[mod])
            for k in ref_coeffs:
                assert jnp.allclose(coeffs[mod][k], ref_coeffs[k])


# ===========================================================================
# Tests for mixed continuous-discrete multimodal integration
# ===========================================================================


class TestMultimodalDiscreteIntegration:
    """Test that discrete processes work within MultimodalCorruptionProcess."""

    def test_mixed_continuous_discrete(self):
        """Multimodal process with one continuous and one discrete modality."""
        procs = {
            "image": SimpleFlow(dim=16),
            "text": MaskLinear(VOCAB_SIZE, SEQ_SHAPE),
        }
        multi = MultimodalCorruptionProcess(procs)
        key = jax.random.PRNGKey(0)
        x = {
            "image": jax.random.normal(jax.random.PRNGKey(1), (BATCH_SIZE, 16)),
            "text": jax.random.randint(
                jax.random.PRNGKey(2), (BATCH_SIZE, *SEQ_SHAPE), 0, VOCAB_SIZE
            ),
        }
        t = {
            "image": 0.3 * jnp.ones((BATCH_SIZE,)),
            "text": 0.5 * jnp.ones((BATCH_SIZE,)),
        }

        x_t, aux = multi.forward(key, x, t)

        assert x_t["image"].shape == (BATCH_SIZE, 16)
        assert x_t["text"].shape == (BATCH_SIZE, *SEQ_SHAPE)
        assert "x" in aux["image"]
        assert "x" in aux["text"]
        assert "corrupted_mask" in aux["text"]
        assert "padding_mask" in aux["text"]
