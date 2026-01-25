"""Tests for DistillationLoss."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.gaussian import FlowMatching, VPCosine
from diffusionlab.training.base import LossOutput
from diffusionlab.training.distillation import DistillationLoss
from diffusionlab.typing import Prediction

BATCH_SIZE = 4
DATA_DIM = 8


def _const_predictor(value, kind="x_0"):
    def predict(x, s, t, cond, aux):
        return Prediction(value=jnp.broadcast_to(value, x.shape), kind=kind)

    return predict


class TestDistillationLoss:
    @pytest.fixture(params=["vpcosine", "flow_matching"])
    def process(self, request):
        if request.param == "vpcosine":
            return VPCosine((DATA_DIM,))
        return FlowMatching((DATA_DIM,))

    def test_returns_loss_output(self, process):
        loss = DistillationLoss(
            process=process, weight_fn=None, loss_fn=None, comparison_kind="x_0"
        )
        student = _const_predictor(jnp.zeros(DATA_DIM))
        teacher = _const_predictor(jnp.ones(DATA_DIM))

        key = jax.random.key(0)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        out = loss(student, x_0, x_t, t, aux, None, teacher=teacher)
        assert isinstance(out, LossOutput)
        assert out.loss.shape == (BATCH_SIZE,)

    def test_zero_loss_when_equal(self, process):
        loss = DistillationLoss(
            process=process, weight_fn=None, loss_fn=None, comparison_kind="x_0"
        )
        shared = _const_predictor(jnp.ones(DATA_DIM))

        key = jax.random.key(1)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        out = loss(shared, x_0, x_t, t, aux, None, teacher=shared)
        assert jnp.allclose(out.loss, 0.0, atol=1e-5)

    def test_requires_teacher(self, process):
        loss = DistillationLoss(
            process=process, weight_fn=None, loss_fn=None, comparison_kind="x_0"
        )
        student = _const_predictor(jnp.zeros(DATA_DIM))
        key = jax.random.key(2)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux = process.forward(key, x_0, t)
        with pytest.raises(AssertionError):
            loss(student, x_0, x_t, t, aux, None)

    def test_pytree_roundtrip(self, process):
        loss = DistillationLoss(
            process=process, weight_fn=None, loss_fn=None, comparison_kind="x_0"
        )
        leaves, treedef = jax.tree_util.tree_flatten(loss)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, DistillationLoss)
        assert restored.comparison_kind == "x_0"
