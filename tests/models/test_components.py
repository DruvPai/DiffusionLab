"""Tests for diffusionlab.models.components (SinusoidalEmbedding, FiLM, AdaLNZero)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from diffusionlab.models.components import AdaLNZero, FiLM, SinusoidalEmbedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 4
EMB_DIM = 32
HIDDEN_DIM = 16
COND_DIM = 8
MAX_PERIOD = 10_000.0


# ---------------------------------------------------------------------------
# SinusoidalEmbedding
# ---------------------------------------------------------------------------


class TestSinusoidalEmbedding:
    """Tests for the SinusoidalEmbedding module."""

    @pytest.fixture
    def embed(self):
        return SinusoidalEmbedding(emb_dim=EMB_DIM, max_period=MAX_PERIOD)

    def test_output_shape(self, embed):
        t = jnp.linspace(0, 1, BATCH)
        out = embed(t)
        assert out.shape == (BATCH, EMB_DIM)

    def test_output_finite(self, embed):
        t = jnp.linspace(0, 1, BATCH)
        out = embed(t)
        assert jnp.all(jnp.isfinite(out))

    def test_deterministic(self, embed):
        t = jnp.linspace(0, 1, BATCH)
        out1 = embed(t)
        out2 = embed(t)
        assert jnp.allclose(out1, out2)

    def test_different_times_give_different_embeddings(self, embed):
        t = jnp.array([0.0, 0.5, 1.0])
        out = embed(t)
        # Each pair should differ
        assert not jnp.allclose(out[0], out[1])
        assert not jnp.allclose(out[1], out[2])
        assert not jnp.allclose(out[0], out[2])

    def test_zero_time(self, embed):
        t = jnp.zeros(1)
        out = embed(t)
        assert out.shape == (1, EMB_DIM)
        assert jnp.all(jnp.isfinite(out))

    def test_cos_sin_split(self, embed):
        """First half should be cosines, second half sines."""
        half = EMB_DIM // 2
        # At t=0, sin part should be 0 and cos part should be 1
        t0 = jnp.array([0.0])
        out0 = embed(t0)[0]
        assert jnp.allclose(out0[:half], jnp.ones(half))
        assert jnp.allclose(out0[half:], jnp.zeros(half), atol=1e-6)

    def test_single_sample(self, embed):
        t = jnp.array([0.5])
        out = embed(t)
        assert out.shape == (1, EMB_DIM)

    def test_large_batch(self, embed):
        t = jnp.linspace(0, 1, 128)
        out = embed(t)
        assert out.shape == (128, EMB_DIM)
        assert jnp.all(jnp.isfinite(out))

    def test_bounded_output(self, embed):
        """Sinusoidal embeddings should be in [-1, 1]."""
        t = jnp.linspace(0, 1, 100)
        out = embed(t)
        assert jnp.all(out >= -1.0 - 1e-6)
        assert jnp.all(out <= 1.0 + 1e-6)

    def test_different_max_period(self):
        e1 = SinusoidalEmbedding(emb_dim=EMB_DIM, max_period=100.0)
        e2 = SinusoidalEmbedding(emb_dim=EMB_DIM, max_period=10_000.0)
        t = jnp.array([0.5])
        out1 = e1(t)
        out2 = e2(t)
        assert not jnp.allclose(out1, out2)

    def test_jit_compatible(self, embed):
        t = jnp.linspace(0, 1, BATCH)
        out = jax.jit(embed)(t)
        assert out.shape == (BATCH, EMB_DIM)
        assert jnp.all(jnp.isfinite(out))

    def test_grad_flows(self, embed):
        """Gradients should flow through the embedding w.r.t. input."""

        def loss(t):
            return jnp.sum(embed(t))

        t = jnp.array([0.5])
        g = jax.grad(loss)(t)
        assert g.shape == (1,)
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# FiLM
# ---------------------------------------------------------------------------


class TestFiLM:
    """Tests for the FiLM conditioning layer."""

    @pytest.fixture
    def film(self):
        key = jax.random.key(0)
        return FiLM(hidden_dim=HIDDEN_DIM, cond_dim=COND_DIM, key=key)

    def test_output_shape(self, film):
        h = jnp.ones(HIDDEN_DIM)
        cond = jnp.ones(COND_DIM)
        out = film(h, cond)
        assert out.shape == (HIDDEN_DIM,)

    def test_output_finite(self, film):
        h = jnp.ones(HIDDEN_DIM)
        cond = jnp.ones(COND_DIM)
        out = film(h, cond)
        assert jnp.all(jnp.isfinite(out))

    def test_deterministic(self, film):
        h = jnp.ones(HIDDEN_DIM)
        cond = jnp.ones(COND_DIM)
        out1 = film(h, cond)
        out2 = film(h, cond)
        assert jnp.allclose(out1, out2)

    def test_different_conditioning_gives_different_output(self, film):
        h = jnp.ones(HIDDEN_DIM)
        cond1 = jnp.zeros(COND_DIM)
        cond2 = jnp.ones(COND_DIM)
        out1 = film(h, cond1)
        out2 = film(h, cond2)
        assert not jnp.allclose(out1, out2)

    def test_different_input_gives_different_output(self, film):
        h1 = jnp.arange(HIDDEN_DIM, dtype=jnp.float32)
        h2 = jnp.flip(h1)
        cond = jnp.ones(COND_DIM)
        out1 = film(h1, cond)
        out2 = film(h2, cond)
        assert not jnp.allclose(out1, out2)

    def test_jit_compatible(self, film):
        h = jnp.ones(HIDDEN_DIM)
        cond = jnp.ones(COND_DIM)
        out = jax.jit(lambda hh, cc: film(hh, cc))(h, cond)
        assert out.shape == (HIDDEN_DIM,)
        assert jnp.all(jnp.isfinite(out))
        out_eqx = eqx.filter_jit(lambda m, hh, cc: m(hh, cc))(film, h, cond)
        assert jnp.allclose(out, out_eqx)

    def test_grad_flows(self, film):
        """Gradients should flow through the FiLM layer."""

        def loss(h, cond):
            return jnp.sum(film(h, cond))

        h = jnp.ones(HIDDEN_DIM)
        cond = jnp.ones(COND_DIM)
        g_h, g_cond = jax.grad(loss, argnums=(0, 1))(h, cond)
        assert g_h.shape == (HIDDEN_DIM,)
        assert g_cond.shape == (COND_DIM,)
        assert jnp.all(jnp.isfinite(g_h))
        assert jnp.all(jnp.isfinite(g_cond))

    def test_different_keys_give_different_params(self):
        f1 = FiLM(hidden_dim=HIDDEN_DIM, cond_dim=COND_DIM, key=jax.random.key(0))
        f2 = FiLM(hidden_dim=HIDDEN_DIM, cond_dim=COND_DIM, key=jax.random.key(1))
        h = jnp.ones(HIDDEN_DIM)
        cond = jnp.ones(COND_DIM)
        out1 = f1(h, cond)
        out2 = f2(h, cond)
        assert not jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# AdaLNZero
# ---------------------------------------------------------------------------


class TestAdaLNZero:
    """Tests for the AdaLN-Zero module."""

    @pytest.fixture
    def adaln(self):
        key = jax.random.key(0)
        return AdaLNZero(hidden_dim=HIDDEN_DIM, cond_dim=COND_DIM, key=key)

    def test_get_modulation_params_shapes(self, adaln):
        cond = jnp.ones(COND_DIM)
        alpha, beta, gamma = adaln.get_modulation_params(cond)
        assert alpha.shape == (HIDDEN_DIM,)
        assert beta.shape == (HIDDEN_DIM,)
        assert gamma.shape == (HIDDEN_DIM,)

    def test_get_modulation_params_finite(self, adaln):
        cond = jnp.ones(COND_DIM)
        alpha, beta, gamma = adaln.get_modulation_params(cond)
        assert jnp.all(jnp.isfinite(alpha))
        assert jnp.all(jnp.isfinite(beta))
        assert jnp.all(jnp.isfinite(gamma))

    def test_modulate_shape_1d(self, adaln):
        """modulate with a sequence of hidden vectors."""
        seq_len = 5
        h = jnp.ones((seq_len, HIDDEN_DIM))
        beta = jnp.zeros(HIDDEN_DIM)
        gamma = jnp.zeros(HIDDEN_DIM)
        out = adaln.modulate(h, beta, gamma)
        assert out.shape == (seq_len, HIDDEN_DIM)

    def test_modulate_finite(self, adaln):
        seq_len = 5
        h = jnp.ones((seq_len, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        alpha, beta, gamma = adaln.get_modulation_params(cond)
        out = adaln.modulate(h, beta, gamma)
        assert jnp.all(jnp.isfinite(out))

    def test_modulate_identity_at_zero_params(self, adaln):
        """With gamma=0 and beta=0, modulate should return norm(h)."""
        seq_len = 3
        h = jnp.ones((seq_len, HIDDEN_DIM)) * 2.0
        beta = jnp.zeros(HIDDEN_DIM)
        gamma = jnp.zeros(HIDDEN_DIM)
        out = adaln.modulate(h, beta, gamma)
        expected = jax.vmap(adaln.norm)(h)
        assert jnp.allclose(out, expected, atol=1e-5)

    def test_different_conditioning(self, adaln):
        cond1 = jnp.zeros(COND_DIM)
        cond2 = jnp.ones(COND_DIM)
        params1 = adaln.get_modulation_params(cond1)
        params2 = adaln.get_modulation_params(cond2)
        # At least one param should differ
        any_different = any(
            not jnp.allclose(p1, p2) for p1, p2 in zip(params1, params2)
        )
        assert any_different

    def test_deterministic(self, adaln):
        cond = jnp.ones(COND_DIM)
        p1 = adaln.get_modulation_params(cond)
        p2 = adaln.get_modulation_params(cond)
        for a, b in zip(p1, p2):
            assert jnp.allclose(a, b)

    def test_jit_compatible(self, adaln):
        cond = jnp.ones(COND_DIM)
        alpha, beta, gamma = jax.jit(lambda cc: adaln.get_modulation_params(cc))(cond)
        assert alpha.shape == (HIDDEN_DIM,)
        alpha_eqx, beta_eqx, gamma_eqx = eqx.filter_jit(
            lambda m, cc: m.get_modulation_params(cc)
        )(adaln, cond)
        assert jnp.allclose(alpha, alpha_eqx)
        assert jnp.allclose(beta, beta_eqx)
        assert jnp.allclose(gamma, gamma_eqx)

        h = jnp.ones((3, HIDDEN_DIM))
        out = jax.jit(lambda hh, bb, gg: adaln.modulate(hh, bb, gg))(h, beta, gamma)
        assert out.shape == (3, HIDDEN_DIM)
        out_eqx = eqx.filter_jit(lambda m, hh, bb, gg: m.modulate(hh, bb, gg))(
            adaln, h, beta, gamma
        )
        assert jnp.allclose(out, out_eqx)

    def test_grad_flows(self, adaln):
        def loss(h, cond):
            alpha, beta, gamma = adaln.get_modulation_params(cond)
            return jnp.sum(adaln.modulate(h, beta, gamma))

        h = jnp.ones((3, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        g_h, g_cond = jax.grad(loss, argnums=(0, 1))(h, cond)
        assert g_h.shape == (3, HIDDEN_DIM)
        assert g_cond.shape == (COND_DIM,)
        assert jnp.all(jnp.isfinite(g_h))
        assert jnp.all(jnp.isfinite(g_cond))
