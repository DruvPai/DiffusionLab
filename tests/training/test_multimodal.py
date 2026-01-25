"""Tests for MultimodalLoss."""

import jax
import jax.numpy as jnp

from diffusionlab.processes.discrete import MaskLinear
from diffusionlab.processes.gaussian import VPCosine
from diffusionlab.training.base import LossOutput, MultimodalLoss
from diffusionlab.training.denoising import DenoisingLoss
from diffusionlab.typing import Prediction

BATCH_SIZE = 4
DATA_DIM = 8
SEQ_LEN = 6
VOCAB_SIZE = 5


class TestMultimodalLoss:
    def _make_setup(self):
        continuous_proc = VPCosine((DATA_DIM,))
        discrete_proc = MaskLinear(VOCAB_SIZE, (SEQ_LEN,))
        num_classes = VOCAB_SIZE + 1

        continuous_loss = DenoisingLoss(
            process=continuous_proc, weight_fn=None, loss_fn=None, target_kind=None
        )
        discrete_loss = DenoisingLoss(
            process=discrete_proc, weight_fn=None, loss_fn=None, target_kind=None
        )

        losses = {"image": continuous_loss, "text": discrete_loss}
        mm_loss = MultimodalLoss(losses=losses)

        def predictor(x, s, t, cond, aux):
            img_pred = Prediction(value=jnp.zeros_like(x["image"]), kind="x_0")
            txt_pred = Prediction(
                value=jnp.zeros((x["text"].shape[0], SEQ_LEN, num_classes)),
                kind="logits_x0",
            )
            return {"image": img_pred, "text": txt_pred}

        return mm_loss, predictor, continuous_proc, discrete_proc, num_classes

    def test_returns_loss_output(self):
        mm_loss, predictor, cont_proc, disc_proc, _ = self._make_setup()
        key = jax.random.key(0)

        x_0_img = jnp.ones((BATCH_SIZE, DATA_DIM))
        x_0_txt = jax.random.randint(key, (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE)
        x_0 = {"image": x_0_img, "text": x_0_txt}

        t = {
            "image": jnp.full((BATCH_SIZE,), 0.5),
            "text": jnp.full((BATCH_SIZE,), 0.5),
        }

        k1, k2 = jax.random.split(key)
        x_t_img, aux_img = cont_proc.forward(k1, x_0_img, t["image"])
        x_t_txt, aux_txt = disc_proc.forward(k2, x_0_txt, t["text"])
        x_t = {"image": x_t_img, "text": x_t_txt}
        aux = {"image": aux_img, "text": aux_txt}

        out = mm_loss(predictor, x_0, x_t, t, aux, None)
        assert isinstance(out, LossOutput)
        assert out.loss.shape == (BATCH_SIZE,)
        assert "image" in out.per_leaf
        assert "text" in out.per_leaf

    def test_with_leaf_weights(self):
        mm_loss_base, predictor, cont_proc, disc_proc, _ = self._make_setup()

        # Rebuild with weights
        losses = {
            "image": mm_loss_base.losses["image"],
            "text": mm_loss_base.losses["text"],
        }
        mm_loss = MultimodalLoss(
            losses=losses, leaf_weights={"image": 2.0, "text": 0.5}
        )

        key = jax.random.key(1)
        x_0_img = jnp.ones((BATCH_SIZE, DATA_DIM))
        x_0_txt = jax.random.randint(key, (BATCH_SIZE, SEQ_LEN), 0, VOCAB_SIZE)
        x_0 = {"image": x_0_img, "text": x_0_txt}

        t = {
            "image": jnp.full((BATCH_SIZE,), 0.5),
            "text": jnp.full((BATCH_SIZE,), 0.5),
        }

        k1, k2 = jax.random.split(key)
        x_t_img, aux_img = cont_proc.forward(k1, x_0_img, t["image"])
        x_t_txt, aux_txt = disc_proc.forward(k2, x_0_txt, t["text"])
        x_t = {"image": x_t_img, "text": x_t_txt}
        aux = {"image": aux_img, "text": aux_txt}

        out = mm_loss(predictor, x_0, x_t, t, aux, None)
        assert out.loss.shape == (BATCH_SIZE,)
