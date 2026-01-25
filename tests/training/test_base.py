"""Tests for diffusionlab.training.base (LossOutput, loss functions, default_loss_fn)."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.discrete import MaskLinear, UniformLinear
from diffusionlab.processes.gaussian import FlowMatching, VPCosine
from diffusionlab.processes.simplicial import SimplicialLinear
from diffusionlab.training.base import (
    LossOutput,
    ce_loss,
    default_loss_fn,
    kl_loss,
    mse_loss,
)

BATCH_SIZE = 4
DATA_DIM = 8
SEQ_LEN = 6
VOCAB_SIZE = 5


# ===========================================================================
# LossOutput
# ===========================================================================


class TestLossOutput:
    def test_fields(self):
        out = LossOutput(
            loss=jnp.ones(BATCH_SIZE),
            per_leaf=jnp.ones(BATCH_SIZE),
            metrics={"foo": jnp.zeros(BATCH_SIZE)},
        )
        assert out.loss.shape == (BATCH_SIZE,)
        assert out.per_leaf.shape == (BATCH_SIZE,)

    def test_frozen(self):
        out = LossOutput(
            loss=jnp.ones(BATCH_SIZE),
            per_leaf=jnp.ones(BATCH_SIZE),
            metrics={},
        )
        with pytest.raises(AttributeError):
            out.loss = jnp.zeros(BATCH_SIZE)  # type: ignore[misc]

    def test_pytree_roundtrip(self):
        out = LossOutput(
            loss=jnp.ones(BATCH_SIZE),
            per_leaf=jnp.ones(BATCH_SIZE) * 2,
            metrics={"m": jnp.ones(BATCH_SIZE) * 3},
        )
        leaves, treedef = jax.tree_util.tree_flatten(out)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert jnp.allclose(restored.loss, out.loss)
        assert jnp.allclose(restored.per_leaf, out.per_leaf)
        assert jnp.allclose(restored.metrics["m"], out.metrics["m"])

    def test_tree_map(self):
        out = LossOutput(
            loss=jnp.ones(BATCH_SIZE),
            per_leaf=jnp.ones(BATCH_SIZE),
            metrics={"v": jnp.ones(BATCH_SIZE)},
        )
        doubled = jax.tree_util.tree_map(lambda x: 2 * x, out)
        assert jnp.allclose(doubled.loss, 2 * jnp.ones(BATCH_SIZE))


# ===========================================================================
# Standalone loss functions
# ===========================================================================


class TestMSELoss:
    def test_shape(self):
        pred = jnp.ones((BATCH_SIZE, DATA_DIM))
        target = jnp.zeros((BATCH_SIZE, DATA_DIM))
        out = mse_loss(pred, target)
        assert out.shape == (BATCH_SIZE,)

    def test_zero_when_equal(self):
        x = jnp.ones((BATCH_SIZE, DATA_DIM))
        assert jnp.allclose(mse_loss(x, x), 0.0)

    def test_value(self):
        pred = jnp.ones((BATCH_SIZE, 4))
        target = jnp.zeros((BATCH_SIZE, 4))
        assert jnp.allclose(mse_loss(pred, target), 1.0)

    def test_higher_dim(self):
        pred = jnp.ones((BATCH_SIZE, 3, 4, 4))
        target = jnp.zeros((BATCH_SIZE, 3, 4, 4))
        out = mse_loss(pred, target)
        assert out.shape == (BATCH_SIZE,)
        assert jnp.allclose(out, 1.0)


class TestCELoss:
    def test_shape(self):
        logits = jnp.ones((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        target = jax.nn.one_hot(jnp.zeros((BATCH_SIZE, SEQ_LEN), dtype=int), VOCAB_SIZE)
        out = ce_loss(logits, target)
        assert out.shape == (BATCH_SIZE,)

    def test_lower_for_correct(self):
        key = jax.random.key(0)
        logits = jax.random.normal(key, (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        # Target = argmax of logits
        correct_idx = jnp.argmax(logits, axis=-1)
        correct_target = jax.nn.one_hot(correct_idx, VOCAB_SIZE)
        wrong_idx = (correct_idx + 1) % VOCAB_SIZE
        wrong_target = jax.nn.one_hot(wrong_idx, VOCAB_SIZE)
        assert jnp.all(ce_loss(logits, correct_target) < ce_loss(logits, wrong_target))


class TestKLLoss:
    def test_shape(self):
        logits = jnp.ones((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE))
        target = jnp.ones((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)) / VOCAB_SIZE
        out = kl_loss(logits, target)
        assert out.shape == (BATCH_SIZE,)

    def test_zero_when_equal(self):
        probs = jnp.ones((BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)) / VOCAB_SIZE
        logits = jnp.log(probs)
        out = kl_loss(logits, probs)
        assert jnp.allclose(out, 0.0, atol=1e-5)


# ===========================================================================
# default_loss_fn
# ===========================================================================


class TestDefaultLossFn:
    def test_gaussian(self):
        process = VPCosine((DATA_DIM,))
        assert default_loss_fn(process) is mse_loss

    def test_flow_matching(self):
        process = FlowMatching((DATA_DIM,))
        assert default_loss_fn(process) is mse_loss

    def test_discrete_mask(self):
        process = MaskLinear(VOCAB_SIZE, (SEQ_LEN,))
        assert default_loss_fn(process) is ce_loss

    def test_discrete_uniform(self):
        process = UniformLinear(VOCAB_SIZE, (SEQ_LEN,))
        assert default_loss_fn(process) is ce_loss

    def test_simplicial(self):
        process = SimplicialLinear.mask_process(VOCAB_SIZE, (SEQ_LEN,))
        assert default_loss_fn(process) is kl_loss

    def test_unknown_raises(self):
        with pytest.raises(TypeError):
            default_loss_fn("not a process")  # type: ignore[arg-type]
