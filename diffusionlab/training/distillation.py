"""Teacher-student distillation loss."""

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
class DistillationLoss(Loss):
    """Teacher-student distillation loss.

    Both student and teacher predictions are converted to a common
    ``comparison_kind`` space before computing the loss.

    Attributes:
        process: The corruption process.
        weight_fn: Optional per-sample weighting function.
        loss_fn: Element-wise loss function. Auto-selected from process if None.
        comparison_kind: Representation space to compare in (e.g. ``"x_0"``, ``"z"``).
    """

    loss_fn: LossFn | None
    comparison_kind: str

    def __call__(
        self,
        predictor: DiffusionPredictor,
        x_0: DataTree,
        x_t: DataTree,
        t: TimeTree,
        aux: AuxInfoTree,
        cond: ConditioningTree | None,
        teacher: DiffusionPredictor | None = None,
        **kwargs,
    ) -> LossOutput:
        assert teacher is not None, "DistillationLoss requires a teacher model."

        loss_fn = (
            self.loss_fn if self.loss_fn is not None else default_loss_fn(self.process)
        )

        # Student prediction
        student_pred = predictor(x_t, None, t, cond, aux)

        # Teacher prediction (stop gradient)
        teacher_pred = teacher(x_t, None, t, cond, aux)
        teacher_pred = jax.lax.stop_gradient(teacher_pred)

        # Convert both to comparison space
        student_conv = self.process.convert_prediction(
            student_pred, x_t, None, t, self.comparison_kind
        )
        teacher_conv = self.process.convert_prediction(
            teacher_pred, x_t, None, t, self.comparison_kind
        )

        loss_val = loss_fn(student_conv.value, teacher_conv.value)

        metrics: dict = {"unweighted_loss": loss_val}
        if self.weight_fn is not None:
            weights = self.weight_fn(aux)
            weighted_loss = loss_val * weights
            metrics["weights"] = weights
        else:
            weighted_loss = loss_val

        return LossOutput(loss=weighted_loss, per_leaf=weighted_loss, metrics=metrics)


jax.tree_util.register_dataclass(
    DistillationLoss,
    data_fields=["process"],
    meta_fields=["weight_fn", "loss_fn", "comparison_kind"],
)
