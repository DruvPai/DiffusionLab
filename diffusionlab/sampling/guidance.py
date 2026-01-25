"""Guidance transforms for guided sampling."""

from typing import Callable

import jax
from jax import numpy as jnp

from diffusionlab.sampling.base import SamplerStep, SamplingTransform
from diffusionlab.typing import (
    ConditioningTree,
    DiffusionPredictor,
    Prediction,
    PredictionTree,
    PRNGKey,
    ScalarTree,
    TimeTree,
)
from diffusionlab.utils.tree_ops import bcast_right


class CFG(SamplingTransform):
    """Classifier-free guidance transform.

    Calls ``predict_fn`` a second time with ``uncond_cond`` and linearly
    combines the conditional and unconditional predictions:

        pred_guided = pred_uncond + w * (pred_cond - pred_uncond)

    Attributes:
        guidance_scale_fn: Guidance weight *w*, either a float or a callable
            ``(t) -> Scalar`` for time-dependent guidance.
    """

    guidance_scale_fn: Callable[[TimeTree], ScalarTree]

    def __init__(
        self,
        guidance_scale: float | ScalarTree | Callable[[TimeTree], ScalarTree],
    ):
        super().__init__()
        if isinstance(guidance_scale, float):
            self.guidance_scale_fn: Callable[[TimeTree], ScalarTree] = lambda t: (
                jax.tree_util.tree_map(
                    lambda s: jnp.broadcast_to(jnp.asarray(guidance_scale), s.shape),
                    t,
                )
            )
        elif callable(guidance_scale):
            self.guidance_scale_fn: Callable[[TimeTree], ScalarTree] = guidance_scale
        else:
            self.guidance_scale_fn: Callable[[TimeTree], ScalarTree] = lambda t: (
                guidance_scale
            )

    def transform_prediction(
        self,
        key: PRNGKey,
        predictor: DiffusionPredictor,
        prediction: PredictionTree,
        step: SamplerStep,
        s: TimeTree,
        cond: ConditioningTree | None,
    ) -> PredictionTree:
        pred_cond = prediction
        # Get unconditional prediction
        pred_uncond = predictor(
            step.x,
            s,
            step.t,
            cond=None,
            aux=step.aux,
        )

        guidance_scale = self.guidance_scale_fn(step.t)

        # Linear combination: pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        return jax.tree.map(
            lambda pc, pu, gs: Prediction(
                value=pu.value + bcast_right(gs, pu.value.ndim) * (pc.value - pu.value),
                kind=pc.kind,
            ),
            pred_cond,
            pred_uncond,
            guidance_scale,
            is_leaf=lambda x: isinstance(x, Prediction),
        )
