"""Tests for DenoisingLoss."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.discrete import MaskLinear, UniformLinear
from diffusionlab.processes.gaussian import FlowMatching, VPCosine
from diffusionlab.processes.simplicial import SimplicialLinear
from diffusionlab.training.base import LossOutput, mse_loss
from diffusionlab.training.denoising import DenoisingLoss
from diffusionlab.training.weighting import uniform_weight
from diffusionlab.typing import Prediction

BATCH_SIZE = 4
DATA_DIM = 8
SEQ_LEN = 6
VOCAB_SIZE = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_predictor(kind, value_fn=None):
    """Predictor returning a fixed-kind prediction."""

    def predict(x, s, t, cond, aux):
        val = value_fn(x) if value_fn is not None else jnp.zeros_like(x)
        return Prediction(value=val, kind=kind)

    return predict


# ===========================================================================
# Tests for DenoisingLoss with Gaussian processes
# ===========================================================================


class TestDenoisingGaussian:
    @pytest.fixture(params=["vpcosine", "flow_matching"])
    def setup(self, request):
        name = request.param
        if name == "vpcosine":
            process = VPCosine((DATA_DIM,))
        else:
            process = FlowMatching((DATA_DIM,))
        return process

    def test_returns_loss_output(self, setup):
        process = setup
        loss = DenoisingLoss(
            process=process, weight_fn=None, loss_fn=None, target_kind=None
        )
        predictor = _dummy_predictor("x_0")
        key = jax.random.key(0)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        x_t, aux = process.forward(key, x_0, jnp.full((BATCH_SIZE,), 0.5))
        t = jnp.full((BATCH_SIZE,), 0.5)
        out = loss(predictor, x_0, x_t, t, aux, None)
        assert isinstance(out, LossOutput)
        assert out.loss.shape == (BATCH_SIZE,)

    def test_zero_loss_for_perfect_prediction(self, setup):
        process = setup
        loss = DenoisingLoss(
            process=process, weight_fn=None, loss_fn=None, target_kind=None
        )
        predictor = _dummy_predictor("x_0", value_fn=lambda x: x)
        key = jax.random.key(1)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        # Predictor returns x_t as x_0 — won't be zero loss, but should be finite
        out = loss(predictor, x_0, x_t, t, aux, None)
        assert jnp.all(jnp.isfinite(out.loss))

    def test_custom_loss_fn(self, setup):
        process = setup
        loss = DenoisingLoss(
            process=process, weight_fn=None, loss_fn=mse_loss, target_kind="x_0"
        )
        predictor = _dummy_predictor("x_0")
        key = jax.random.key(2)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        out = loss(predictor, x_0, x_t, t, aux, None)
        assert out.loss.shape == (BATCH_SIZE,)

    def test_with_weight_fn(self, setup):
        process = setup
        loss = DenoisingLoss(
            process=process, weight_fn=uniform_weight, loss_fn=None, target_kind=None
        )
        predictor = _dummy_predictor("x_0")
        key = jax.random.key(3)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        out = loss(predictor, x_0, x_t, t, aux, None)
        assert "weights" in out.metrics
        assert out.loss.shape == (BATCH_SIZE,)

    def test_eps_prediction_kind(self, setup):
        process = setup

        def eps_predictor(x, s, t, cond, aux):
            return Prediction(value=jnp.zeros_like(x), kind="z")

        loss = DenoisingLoss(
            process=process, weight_fn=None, loss_fn=None, target_kind=None
        )
        key = jax.random.key(4)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        out = loss(eps_predictor, x_0, x_t, t, aux, None)
        assert out.loss.shape == (BATCH_SIZE,)


# ===========================================================================
# Tests for DenoisingLoss with Discrete processes
# ===========================================================================


class TestDenoisingDiscrete:
    @pytest.fixture(params=["mask", "uniform"])
    def setup(self, request):
        if request.param == "mask":
            process = MaskLinear(VOCAB_SIZE, (SEQ_LEN,))
            num_classes = VOCAB_SIZE + 1  # mask token
        else:
            process = UniformLinear(VOCAB_SIZE, (SEQ_LEN,))
            num_classes = VOCAB_SIZE
        return process, num_classes

    def test_returns_loss_output(self, setup):
        process, num_classes = setup
        loss = DenoisingLoss(
            process=process, weight_fn=None, loss_fn=None, target_kind=None
        )

        def predict(x, s, t, cond, aux):
            return Prediction(
                value=jnp.zeros((x.shape[0], *x.shape[1:], num_classes)),
                kind="logits_x0",
            )

        key = jax.random.key(10)
        x_0 = jax.random.randint(key, (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE)
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        out = loss(predict, x_0, x_t, t, aux, None)
        assert isinstance(out, LossOutput)
        assert out.loss.shape == (BATCH_SIZE,)
        assert jnp.all(out.loss >= 0)


# ===========================================================================
# Tests for DenoisingLoss with Simplicial processes
# ===========================================================================


class TestDenoisingSimplicial:
    def test_returns_loss_output(self):
        process = SimplicialLinear.mask_process(VOCAB_SIZE, (SEQ_LEN,))
        num_classes = VOCAB_SIZE + 1
        loss = DenoisingLoss(
            process=process, weight_fn=None, loss_fn=None, target_kind=None
        )

        def predict(x, s, t, cond, aux):
            return Prediction(
                value=jnp.zeros((x.shape[0], SEQ_LEN, num_classes)),
                kind="logits_x0",
            )

        key = jax.random.key(20)
        x_0 = jax.random.randint(key, (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE)
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        out = loss(predict, x_0, x_t, t, aux, None)
        assert isinstance(out, LossOutput)
        assert out.loss.shape == (BATCH_SIZE,)
        assert jnp.all(jnp.isfinite(out.loss))


# ===========================================================================
# Pytree registration
# ===========================================================================


class TestDenoisingLossPytree:
    def test_pytree_roundtrip(self):
        process = VPCosine((DATA_DIM,))
        loss = DenoisingLoss(
            process=process, weight_fn=None, loss_fn=None, target_kind=None
        )
        leaves, treedef = jax.tree_util.tree_flatten(loss)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, DenoisingLoss)
