"""Tests for ConsistencyLoss."""

import jax
import jax.numpy as jnp
import pytest

from diffusionlab.processes.gaussian import FlowMatching, VPCosine
from diffusionlab.training.base import LossOutput
from diffusionlab.training.consistency import ConsistencyLoss
from diffusionlab.typing import Prediction

BATCH_SIZE = 4
DATA_DIM = 8


def _const_predictor(value, kind="x_0"):
    def predict(x, s, t, cond, aux):
        return Prediction(value=jnp.broadcast_to(value, x.shape), kind=kind)

    return predict


class TestConsistencyLoss:
    @pytest.fixture(params=["vpcosine", "flow_matching"])
    def process(self, request):
        if request.param == "vpcosine":
            return VPCosine((DATA_DIM,))
        return FlowMatching((DATA_DIM,))

    def test_returns_loss_output(self, process):
        loss = ConsistencyLoss(process=process, weight_fn=None, loss_fn=None)
        predictor = _const_predictor(jnp.ones(DATA_DIM))

        key = jax.random.key(0)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.7)
        s = jnp.full((BATCH_SIZE,), 0.3)
        results = process.forward_multiple(key, x_0, [t, s])
        x_t, aux_t = results[0]
        x_s, aux_s = results[1]

        out = loss(predictor, x_0, x_t, t, aux_t, None, x_s=x_s, s=s, aux_s=aux_s)
        assert isinstance(out, LossOutput)
        assert out.loss.shape == (BATCH_SIZE,)

    def test_zero_loss_self_consistency(self, process):
        loss = ConsistencyLoss(process=process, weight_fn=None, loss_fn=None)
        # Predictor always returns the same x_0 regardless of input
        predictor = _const_predictor(jnp.ones(DATA_DIM))

        key = jax.random.key(1)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.7)
        s = jnp.full((BATCH_SIZE,), 0.3)
        results = process.forward_multiple(key, x_0, [t, s])
        x_t, aux_t = results[0]
        x_s, aux_s = results[1]

        out = loss(predictor, x_0, x_t, t, aux_t, None, x_s=x_s, s=s, aux_s=aux_s)
        assert jnp.allclose(out.loss, 0.0, atol=1e-5)

    def test_with_target_predictor(self, process):
        loss = ConsistencyLoss(process=process, weight_fn=None, loss_fn=None)
        online = _const_predictor(jnp.zeros(DATA_DIM))
        target = _const_predictor(jnp.ones(DATA_DIM))

        key = jax.random.key(2)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.7)
        s = jnp.full((BATCH_SIZE,), 0.3)
        results = process.forward_multiple(key, x_0, [t, s])
        x_t, aux_t = results[0]
        x_s, aux_s = results[1]

        out = loss(
            online,
            x_0,
            x_t,
            t,
            aux_t,
            None,
            x_s=x_s,
            s=s,
            aux_s=aux_s,
            target_predictor=target,
        )
        assert out.loss.shape == (BATCH_SIZE,)
        assert jnp.all(out.loss > 0)

    def test_requires_s_and_x_s(self, process):
        loss = ConsistencyLoss(process=process, weight_fn=None, loss_fn=None)
        predictor = _const_predictor(jnp.ones(DATA_DIM))
        key = jax.random.key(3)
        x_0 = jnp.ones((BATCH_SIZE, DATA_DIM))
        t = jnp.full((BATCH_SIZE,), 0.5)
        x_t, aux_t = process.forward(key, x_0, t)
        with pytest.raises(AssertionError):
            loss(predictor, x_0, x_t, t, aux_t, None)

    def test_pytree_roundtrip(self, process):
        loss = ConsistencyLoss(process=process, weight_fn=None, loss_fn=None)
        leaves, treedef = jax.tree_util.tree_flatten(loss)
        restored = jax.tree_util.tree_unflatten(treedef, leaves)
        assert isinstance(restored, ConsistencyLoss)
