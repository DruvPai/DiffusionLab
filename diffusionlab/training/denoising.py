"""Standard denoising training loss."""

from dataclasses import dataclass

import jax
from jax import numpy as jnp

from diffusionlab.processes.discrete import DiscreteCorruptionProcess
from diffusionlab.processes.interpolation import InterpolationProcess
from diffusionlab.processes.simplicial import SimplicialCorruptionProcess
from diffusionlab.training.base import (
    Loss,
    LossFn,
    LossOutput,
    default_loss_fn,
)
from diffusionlab.typing import (
    AuxInfoTree,
    ConditioningTree,
    DataTree,
    DiffusionPredictor,
    Prediction,
    TimeTree,
)


@dataclass(frozen=True, kw_only=True)
class DenoisingLoss(Loss):
    """Standard denoising loss for diffusion training.

    Calls the predictor on corrupted data and compares against ground truth
    in a configurable representation space.

    Attributes:
        process: The corruption process.
        weight_fn: Optional per-sample weighting function.
        loss_fn: Element-wise loss function. Auto-selected from process if None.
        target_kind: Representation space for comparison.
            None means auto: use prediction.kind for continuous,
            ``"x_0"`` for discrete/simplicial.
    """

    loss_fn: LossFn | None
    target_kind: str | None

    def __call__(
        self,
        predictor: DiffusionPredictor,
        x_0: DataTree,
        x_t: DataTree,
        t: TimeTree,
        aux: AuxInfoTree,
        cond: ConditioningTree | None,
        **kwargs,
    ) -> LossOutput:
        # 1. Get model prediction
        prediction = predictor(x_t, None, t, cond, aux)

        # 2. Compute loss based on process type
        loss_fn = (
            self.loss_fn if self.loss_fn is not None else default_loss_fn(self.process)
        )

        if isinstance(self.process, InterpolationProcess):
            loss_val = self._continuous_loss(prediction, x_0, x_t, t, loss_fn)
        elif isinstance(
            self.process, (DiscreteCorruptionProcess, SimplicialCorruptionProcess)
        ):
            loss_val = self._categorical_loss(prediction, x_0, loss_fn)
        else:
            raise TypeError(f"Unsupported process type: {type(self.process)}")

        # 3. Apply weight_fn if present
        metrics: dict = {"unweighted_loss": loss_val}
        if self.weight_fn is not None:
            weights = self.weight_fn(aux)
            weighted_loss = loss_val * weights
            metrics["weights"] = weights
        else:
            weighted_loss = loss_val

        return LossOutput(loss=weighted_loss, per_leaf=weighted_loss, metrics=metrics)

    def _continuous_loss(
        self,
        prediction: Prediction,
        x_0: DataTree,
        x_t: DataTree,
        t: TimeTree,
        loss_fn: LossFn,
    ):
        """Compute loss for continuous (interpolation) processes."""
        target_kind = (
            self.target_kind if self.target_kind is not None else prediction.kind
        )
        gt = Prediction(value=x_0, kind="x_0")
        gt_converted = self.process.convert_prediction(gt, x_t, None, t, target_kind)
        pred_converted = self.process.convert_prediction(
            prediction, x_t, None, t, target_kind
        )
        return loss_fn(pred_converted.value, gt_converted.value)

    def _categorical_loss(
        self,
        prediction: Prediction,
        x_0: DataTree,
        loss_fn: LossFn,
    ):
        """Compute loss for discrete or simplicial processes."""
        # Convert prediction to logits_x0 space
        pred_logits = self.process.convert_prediction(
            prediction, x_0, None, jnp.zeros((x_0.shape[0],)), "logits_x0"
        )
        # Build one-hot target from x_0
        num_classes = pred_logits.value.shape[-1]
        one_hot_x0 = jax.nn.one_hot(x_0, num_classes)
        return loss_fn(pred_logits.value, one_hot_x0)


jax.tree_util.register_dataclass(
    DenoisingLoss,
    data_fields=["process"],
    meta_fields=["weight_fn", "loss_fn", "target_kind"],
)
