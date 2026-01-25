"""Tests for diffusionlab.models.dit (DiTBlock, DiT, ConditionedDiT)."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from diffusionlab.models.dit import ConditionedDiT, DiT, DiTBlock
from diffusionlab.typing import Prediction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH = 4
SEQ_LEN = 6
INPUT_DIM = 8
OUTPUT_DIM = 8
HIDDEN_DIM = 16
NUM_LAYERS = 2
NUM_HEADS = 2
HEAD_DIM = 8
MLP_DIM = 32
MAX_SEQ_LEN = 10
TIME_EMB_DIM = 32
COND_EMB_DIM = 6
COND_DIM = 16
MAX_PERIOD = 10_000.0


# ---------------------------------------------------------------------------
# DiTBlock
# ---------------------------------------------------------------------------


class TestDiTBlock:
    """Tests for the DiTBlock module."""

    @pytest.fixture
    def block(self):
        return DiTBlock(
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            mlp_dim=MLP_DIM,
            cond_dim=COND_DIM,
            activation=jax.nn.gelu,
            key=jax.random.key(0),
        )

    def test_output_shape(self, block):
        h = jnp.ones((SEQ_LEN, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        out = block(h, cond)
        assert out.shape == (SEQ_LEN, HIDDEN_DIM)

    def test_output_finite(self, block):
        h = jnp.ones((SEQ_LEN, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        out = block(h, cond)
        assert jnp.all(jnp.isfinite(out))

    def test_deterministic(self, block):
        h = jnp.ones((SEQ_LEN, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        out1 = block(h, cond)
        out2 = block(h, cond)
        assert jnp.allclose(out1, out2)

    def test_different_conditioning(self, block):
        h = jnp.ones((SEQ_LEN, HIDDEN_DIM))
        out1 = block(h, jnp.zeros(COND_DIM))
        out2 = block(h, jnp.ones(COND_DIM))
        assert not jnp.allclose(out1, out2)

    def test_different_input(self, block):
        cond = jnp.ones(COND_DIM)
        h1 = jnp.zeros((SEQ_LEN, HIDDEN_DIM))
        h2 = jnp.ones((SEQ_LEN, HIDDEN_DIM))
        out1 = block(h1, cond)
        out2 = block(h2, cond)
        assert not jnp.allclose(out1, out2)

    def test_residual_connection(self, block):
        """Output should differ from input (non-trivial transformation)."""
        h = jnp.ones((SEQ_LEN, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        out = block(h, cond)
        assert not jnp.allclose(h, out)

    def test_jit_compatible(self, block):
        h = jnp.ones((SEQ_LEN, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        out = jax.jit(lambda hh, cc: block(hh, cc))(h, cond)
        assert out.shape == (SEQ_LEN, HIDDEN_DIM)
        assert jnp.all(jnp.isfinite(out))
        out_eqx = eqx.filter_jit(lambda m, hh, cc: m(hh, cc))(block, h, cond)
        assert jnp.allclose(out, out_eqx)

    def test_grad_flows(self, block):
        def loss(h, cond):
            return jnp.sum(block(h, cond))

        h = jnp.ones((SEQ_LEN, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        g_h, g_cond = jax.grad(loss, argnums=(0, 1))(h, cond)
        assert g_h.shape == (SEQ_LEN, HIDDEN_DIM)
        assert g_cond.shape == (COND_DIM,)
        assert jnp.all(jnp.isfinite(g_h))
        assert jnp.all(jnp.isfinite(g_cond))

    def test_seq_len_one(self, block):
        h = jnp.ones((1, HIDDEN_DIM))
        cond = jnp.ones(COND_DIM)
        out = block(h, cond)
        assert out.shape == (1, HIDDEN_DIM)
        assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


def _make_dit(**overrides) -> DiT:
    defaults = dict(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mlp_dim=MLP_DIM,
        max_seq_len=MAX_SEQ_LEN,
        time_emb_dim=TIME_EMB_DIM,
        cond_dim=COND_DIM,
        prediction_kind="eps",
        activation=jax.nn.gelu,
        max_period=MAX_PERIOD,
        key=jax.random.key(42),
    )
    defaults.update(overrides)
    return DiT(**defaults)  # type: ignore


class TestDiT:
    """Tests for the DiT model."""

    @pytest.fixture
    def model(self):
        return _make_dit()

    def test_output_shape(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        pred = model(x, s=None, t=t, cond=None, aux=None)
        assert isinstance(pred, Prediction)
        assert pred.value.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)

    def test_output_finite(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        pred = model(x, s=None, t=t, cond=None, aux=None)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_prediction_kind(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        pred = model(x, s=None, t=t, cond=None, aux=None)
        assert pred.kind == "eps"

    def test_different_prediction_kinds(self):
        for kind in ["eps", "x_0", "v_t"]:
            m = _make_dit(prediction_kind=kind)
            x = jnp.ones((2, SEQ_LEN, INPUT_DIM))
            t = jnp.array([0.1, 0.5])
            pred = m(x, s=None, t=t, cond=None, aux=None)
            assert pred.kind == kind

    def test_deterministic(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        p1 = model(x, s=None, t=t, cond=None, aux=None)
        p2 = model(x, s=None, t=t, cond=None, aux=None)
        assert jnp.allclose(p1.value, p2.value)

    def test_different_times_give_different_output(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t1 = jnp.full(BATCH, 0.1)
        t2 = jnp.full(BATCH, 0.9)
        p1 = model(x, s=None, t=t1, cond=None, aux=None)
        p2 = model(x, s=None, t=t2, cond=None, aux=None)
        assert not jnp.allclose(p1.value, p2.value)

    def test_different_input_gives_different_output(self, model):
        t = jnp.linspace(0.01, 0.99, BATCH)
        x1 = jnp.zeros((BATCH, SEQ_LEN, INPUT_DIM))
        x2 = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        p1 = model(x1, s=None, t=t, cond=None, aux=None)
        p2 = model(x2, s=None, t=t, cond=None, aux=None)
        assert not jnp.allclose(p1.value, p2.value)

    def test_jit_compatible(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        pred = jax.jit(lambda xx, tt: model(xx, s=None, t=tt, cond=None, aux=None))(
            x, t
        )
        assert pred.value.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))
        pred_eqx = eqx.filter_jit(
            lambda m, xx, tt: m(xx, s=None, t=tt, cond=None, aux=None)
        )(model, x, t)
        assert jnp.allclose(pred.value, pred_eqx.value)

    def test_grad_flows_through_model(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)

        @eqx.filter_grad
        def grad_fn(m):
            pred = m(x, s=None, t=t, cond=None, aux=None)
            return jnp.sum(pred.value)

        grads = grad_fn(model)
        leaves = jax.tree.leaves(grads)
        any_nonzero = any(jnp.any(leaf != 0) for leaf in leaves)
        assert any_nonzero

    def test_single_layer(self):
        m = _make_dit(num_layers=1)
        x = jnp.ones((2, SEQ_LEN, INPUT_DIM))
        t = jnp.array([0.1, 0.5])
        pred = m(x, s=None, t=t, cond=None, aux=None)
        assert pred.value.shape == (2, SEQ_LEN, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_many_layers(self):
        m = _make_dit(num_layers=6)
        x = jnp.ones((2, SEQ_LEN, INPUT_DIM))
        t = jnp.array([0.1, 0.5])
        pred = m(x, s=None, t=t, cond=None, aux=None)
        assert pred.value.shape == (2, SEQ_LEN, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_different_input_output_dims(self):
        m = _make_dit(input_dim=4, output_dim=12)
        x = jnp.ones((2, SEQ_LEN, 4))
        t = jnp.array([0.1, 0.5])
        pred = m(x, s=None, t=t, cond=None, aux=None)
        assert pred.value.shape == (2, SEQ_LEN, 12)

    def test_shorter_than_max_seq_len(self, model):
        """Should work with sequences shorter than max_seq_len."""
        short_len = 3
        x = jnp.ones((BATCH, short_len, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        pred = model(x, s=None, t=t, cond=None, aux=None)
        assert pred.value.shape == (BATCH, short_len, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_pos_embed_shape(self, model):
        assert model.pos_embed.shape == (MAX_SEQ_LEN, HIDDEN_DIM)

    def test_batch_size_one(self, model):
        x = jnp.ones((1, SEQ_LEN, INPUT_DIM))
        t = jnp.array([0.5])
        pred = model(x, s=None, t=t, cond=None, aux=None)
        assert pred.value.shape == (1, SEQ_LEN, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_is_eqx_module(self, model):
        assert isinstance(model, eqx.Module)

    def test_is_diffusion_model(self, model):
        from diffusionlab.models.base import DiffusionModel

        assert isinstance(model, DiffusionModel)

    def test_different_keys_give_different_params(self):
        m1 = _make_dit(key=jax.random.key(0))
        m2 = _make_dit(key=jax.random.key(1))
        x = jnp.ones((2, SEQ_LEN, INPUT_DIM))
        t = jnp.array([0.1, 0.5])
        p1 = m1(x, s=None, t=t, cond=None, aux=None)
        p2 = m2(x, s=None, t=t, cond=None, aux=None)
        assert not jnp.allclose(p1.value, p2.value)

    def test_pytree_operations(self, model):
        flat, treedef = jax.tree.flatten(model)
        reconstructed = treedef.unflatten(flat)
        x = jnp.ones((2, SEQ_LEN, INPUT_DIM))
        t = jnp.array([0.1, 0.5])
        p1 = model(x, s=None, t=t, cond=None, aux=None)
        p2 = reconstructed(x, s=None, t=t, cond=None, aux=None)
        assert jnp.allclose(p1.value, p2.value)


# ---------------------------------------------------------------------------
# ConditionedDiT
# ---------------------------------------------------------------------------


def _make_cond_dit(**overrides) -> ConditionedDiT:
    defaults = dict(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        mlp_dim=MLP_DIM,
        max_seq_len=MAX_SEQ_LEN,
        time_emb_dim=TIME_EMB_DIM,
        cond_emb_dim=COND_EMB_DIM,
        cond_dim=COND_DIM,
        prediction_kind="eps",
        activation=jax.nn.gelu,
        max_period=MAX_PERIOD,
        key=jax.random.key(42),
    )
    defaults.update(overrides)
    return ConditionedDiT(**defaults)  # type: ignore


class TestConditionedDiT:
    """Tests for the ConditionedDiT model."""

    @pytest.fixture
    def model(self):
        return _make_cond_dit()

    def test_output_shape_with_cond(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        cond = jnp.ones((BATCH, COND_EMB_DIM))
        pred = model(x, s=None, t=t, cond=cond, aux=None)
        assert isinstance(pred, Prediction)
        assert pred.value.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)

    def test_output_finite_with_cond(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        cond = jnp.ones((BATCH, COND_EMB_DIM))
        pred = model(x, s=None, t=t, cond=cond, aux=None)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_prediction_kind(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        cond = jnp.ones((BATCH, COND_EMB_DIM))
        pred = model(x, s=None, t=t, cond=cond, aux=None)
        assert pred.kind == "eps"

    def test_null_conditioning(self, model):
        """Passing cond=None should use the learned null_cond."""
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        pred = model(x, s=None, t=t, cond=None, aux=None)
        assert pred.value.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_null_cond_differs_from_explicit_cond(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        pred_null = model(x, s=None, t=t, cond=None, aux=None)
        cond = jnp.ones((BATCH, COND_EMB_DIM)) * 5.0
        pred_cond = model(x, s=None, t=t, cond=cond, aux=None)
        assert not jnp.allclose(pred_null.value, pred_cond.value)

    def test_different_conditioning_gives_different_output(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        cond1 = jnp.zeros((BATCH, COND_EMB_DIM))
        cond2 = jnp.ones((BATCH, COND_EMB_DIM))
        p1 = model(x, s=None, t=t, cond=cond1, aux=None)
        p2 = model(x, s=None, t=t, cond=cond2, aux=None)
        assert not jnp.allclose(p1.value, p2.value)

    def test_deterministic(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        cond = jnp.ones((BATCH, COND_EMB_DIM))
        p1 = model(x, s=None, t=t, cond=cond, aux=None)
        p2 = model(x, s=None, t=t, cond=cond, aux=None)
        assert jnp.allclose(p1.value, p2.value)

    def test_different_times_give_different_output(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        cond = jnp.ones((BATCH, COND_EMB_DIM))
        t1 = jnp.full(BATCH, 0.1)
        t2 = jnp.full(BATCH, 0.9)
        p1 = model(x, s=None, t=t1, cond=cond, aux=None)
        p2 = model(x, s=None, t=t2, cond=cond, aux=None)
        assert not jnp.allclose(p1.value, p2.value)

    def test_jit_compatible(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        cond = jnp.ones((BATCH, COND_EMB_DIM))
        pred = jax.jit(lambda xx, tt, cc: model(xx, s=None, t=tt, cond=cc, aux=None))(
            x, t, cond
        )
        assert pred.value.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))
        pred_eqx = eqx.filter_jit(
            lambda m, xx, tt, cc: m(xx, s=None, t=tt, cond=cc, aux=None)
        )(model, x, t, cond)
        assert jnp.allclose(pred.value, pred_eqx.value)

    def test_jit_compatible_null_cond(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        pred = jax.jit(lambda xx, tt: model(xx, s=None, t=tt, cond=None, aux=None))(
            x, t
        )
        assert pred.value.shape == (BATCH, SEQ_LEN, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))
        pred_eqx = eqx.filter_jit(
            lambda m, xx, tt: m(xx, s=None, t=tt, cond=None, aux=None)
        )(model, x, t)
        assert jnp.allclose(pred.value, pred_eqx.value, atol=1e-6)

    def test_grad_flows_through_model(self, model):
        x = jnp.ones((BATCH, SEQ_LEN, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        cond = jnp.ones((BATCH, COND_EMB_DIM))

        @eqx.filter_grad
        def grad_fn(m):
            pred = m(x, s=None, t=t, cond=cond, aux=None)
            return jnp.sum(pred.value)

        grads = grad_fn(model)
        leaves = jax.tree.leaves(grads)
        any_nonzero = any(jnp.any(leaf != 0) for leaf in leaves)
        assert any_nonzero

    def test_null_cond_shape(self, model):
        assert model.null_cond.shape == (COND_EMB_DIM,)

    def test_pos_embed_shape(self, model):
        assert model.pos_embed.shape == (MAX_SEQ_LEN, HIDDEN_DIM)

    def test_shorter_than_max_seq_len(self, model):
        short_len = 3
        x = jnp.ones((BATCH, short_len, INPUT_DIM))
        t = jnp.linspace(0.01, 0.99, BATCH)
        cond = jnp.ones((BATCH, COND_EMB_DIM))
        pred = model(x, s=None, t=t, cond=cond, aux=None)
        assert pred.value.shape == (BATCH, short_len, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_different_input_output_dims(self):
        m = _make_cond_dit(input_dim=4, output_dim=12)
        x = jnp.ones((2, SEQ_LEN, 4))
        t = jnp.array([0.1, 0.5])
        cond = jnp.ones((2, COND_EMB_DIM))
        pred = m(x, s=None, t=t, cond=cond, aux=None)
        assert pred.value.shape == (2, SEQ_LEN, 12)

    def test_batch_size_one(self, model):
        x = jnp.ones((1, SEQ_LEN, INPUT_DIM))
        t = jnp.array([0.5])
        cond = jnp.ones((1, COND_EMB_DIM))
        pred = model(x, s=None, t=t, cond=cond, aux=None)
        assert pred.value.shape == (1, SEQ_LEN, OUTPUT_DIM)
        assert jnp.all(jnp.isfinite(pred.value))

    def test_is_diffusion_model(self, model):
        from diffusionlab.models.base import DiffusionModel

        assert isinstance(model, DiffusionModel)

    def test_different_keys_give_different_params(self):
        m1 = _make_cond_dit(key=jax.random.key(0))
        m2 = _make_cond_dit(key=jax.random.key(1))
        x = jnp.ones((2, SEQ_LEN, INPUT_DIM))
        t = jnp.array([0.1, 0.5])
        cond = jnp.ones((2, COND_EMB_DIM))
        p1 = m1(x, s=None, t=t, cond=cond, aux=None)
        p2 = m2(x, s=None, t=t, cond=cond, aux=None)
        assert not jnp.allclose(p1.value, p2.value)

    def test_pytree_operations(self, model):
        flat, treedef = jax.tree.flatten(model)
        reconstructed = treedef.unflatten(flat)
        x = jnp.ones((2, SEQ_LEN, INPUT_DIM))
        t = jnp.array([0.1, 0.5])
        cond = jnp.ones((2, COND_EMB_DIM))
        p1 = model(x, s=None, t=t, cond=cond, aux=None)
        p2 = reconstructed(x, s=None, t=t, cond=cond, aux=None)
        assert jnp.allclose(p1.value, p2.value)
