"""Self-consistency / consistency training loss."""

from dataclasses import dataclass

import jax

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
    TimeTree,
)


@dataclass(frozen=True, kw_only=True)
class ConsistencyLoss(Loss):
    """Consistency training loss.

    Enforces that the predictor's x_0 estimate is consistent across two
    corruption levels (times t and s < t).

    Attributes:
        process: The corruption process.
        weight_fn: Optional per-sample weighting function.
        loss_fn: Element-wise loss function. Auto-selected from process if None.
    """

    loss_fn: LossFn | None

    def __call__(
        self,
        predictor: DiffusionPredictor,
        x_0: DataTree,
        x_t: DataTree,
        t: TimeTree,
        aux: AuxInfoTree,
        cond: ConditioningTree | None,
        x_s: DataTree | None = None,
        s: TimeTree | None = None,
        aux_s: AuxInfoTree | None = None,
        target_predictor: DiffusionPredictor | None = None,
        **kwargs,
    ) -> LossOutput:
        assert x_s is not None, "ConsistencyLoss requires x_s."
        assert s is not None, "ConsistencyLoss requires s."
        assert aux_s is not None, "ConsistencyLoss requires aux_s."

        loss_fn = (
            self.loss_fn if self.loss_fn is not None else default_loss_fn(self.process)
        )

        # Use target_predictor if provided, else self-consistency with predictor
        target_pred_fn = target_predictor if target_predictor is not None else predictor

        # Prediction at time t (online/student)
        pred_t = predictor(x_t, None, t, cond, aux)

        # Prediction at time s (target/EMA), with stop_gradient
        pred_s = target_pred_fn(x_s, None, s, cond, aux_s)
        pred_s = jax.lax.stop_gradient(pred_s)

        # Convert both to x_0 space
        pred_x0_t = self.process.convert_prediction(pred_t, x_t, None, t, "x_0")
        pred_x0_s = self.process.convert_prediction(pred_s, x_s, None, s, "x_0")

        loss_val = loss_fn(pred_x0_t.value, pred_x0_s.value)

        metrics: dict = {"unweighted_loss": loss_val}
        if self.weight_fn is not None:
            weights = self.weight_fn(aux)
            weighted_loss = loss_val * weights
            metrics["weights"] = weights
        else:
            weighted_loss = loss_val

        return LossOutput(loss=weighted_loss, per_leaf=weighted_loss, metrics=metrics)


jax.tree_util.register_dataclass(
    ConsistencyLoss,
    data_fields=["process"],
    meta_fields=["weight_fn", "loss_fn"],
)
