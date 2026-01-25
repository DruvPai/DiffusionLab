"""Tests for weighting functions."""

import jax.numpy as jnp

from diffusionlab.training.weighting import (
    inverse_sigma_squared_weight,
    make_weight_fn,
    min_snr_weight,
    p2_weight,
    snr_weight,
    truncated_snr_weight,
    uniform_weight,
)

BATCH_SIZE = 4


def _make_aux(**kwargs):
    return {k: jnp.array(v) for k, v in kwargs.items()}


class TestUniformWeight:
    def test_shape(self):
        aux = _make_aux(t=jnp.zeros(BATCH_SIZE))
        out = uniform_weight(aux)
        assert out.shape == (BATCH_SIZE,)
        assert jnp.allclose(out, 1.0)


class TestSNRWeight:
    def test_shape_and_value(self):
        logsnr = jnp.array([0.0, 1.0, -1.0, 2.0])
        aux = _make_aux(logsnr=logsnr)
        out = snr_weight(aux)
        assert out.shape == (BATCH_SIZE,)
        assert jnp.allclose(out, jnp.exp(logsnr))


class TestTruncatedSNRWeight:
    def test_truncation(self):
        logsnr = jnp.array([0.0, 10.0, -1.0, 5.0])
        aux = _make_aux(logsnr=logsnr)
        out = truncated_snr_weight(aux, max_weight=3.0)
        assert jnp.all(out <= 3.0)

    def test_default_max(self):
        logsnr = jnp.array([0.0, 10.0, -1.0, 5.0])
        aux = _make_aux(logsnr=logsnr)
        out = truncated_snr_weight(aux)
        assert jnp.all(out <= 5.0)


class TestMinSNRWeight:
    def test_bounded_above(self):
        logsnr = jnp.array([0.0, 10.0, -1.0, 5.0])
        aux = _make_aux(logsnr=logsnr)
        out = min_snr_weight(aux, gamma=5.0)
        assert jnp.all(out <= 1.0 + 1e-5)  # min(SNR, gamma)/SNR <= 1
        assert out.shape == (BATCH_SIZE,)


class TestInverseSigmaSquaredWeight:
    def test_value(self):
        sigma = jnp.array([1.0, 2.0, 0.5, 0.1])
        aux = _make_aux(sigma=sigma)
        out = inverse_sigma_squared_weight(aux)
        assert jnp.allclose(out, 1.0 / sigma**2)


class TestP2Weight:
    def test_shape(self):
        logsnr = jnp.array([0.0, 1.0, -1.0, 2.0])
        aux = _make_aux(logsnr=logsnr)
        out = p2_weight(aux, k=1.0, gamma=1.0)
        assert out.shape == (BATCH_SIZE,)
        snr = jnp.exp(logsnr)
        expected = 1.0 / (1.0 + snr)
        assert jnp.allclose(out, expected)


class TestMakeWeightFn:
    def test_partial_application(self):
        wfn = make_weight_fn(truncated_snr_weight, max_weight=2.0)
        logsnr = jnp.array([0.0, 10.0, -1.0, 5.0])
        aux = _make_aux(logsnr=logsnr)
        out = wfn(aux)
        assert jnp.all(out <= 2.0)

    def test_p2_partial(self):
        wfn = make_weight_fn(p2_weight, k=2.0, gamma=0.5)
        logsnr = jnp.zeros(BATCH_SIZE)
        aux = _make_aux(logsnr=logsnr)
        out = wfn(aux)
        assert out.shape == (BATCH_SIZE,)
