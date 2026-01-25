"""Loss base class, LossOutput, MultimodalLoss, and standalone loss functions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import jax
from jax import numpy as jnp
from jaxtyping import PyTree

from diffusionlab.processes.base import (
    CorruptionProcess,
    MultimodalCorruptionProcess,
)
from diffusionlab.processes.discrete import DiscreteCorruptionProcess
from diffusionlab.processes.interpolation import InterpolationProcess
from diffusionlab.processes.simplicial import SimplicialCorruptionProcess
from diffusionlab.typing import (
    AuxInfoTree,
    ConditioningTree,
    Data,
    DataTree,
    DiffusionPredictor,
    Prediction,
    Scalar,
    ScalarTree,
    TimeTree,
    TreeShape,
)

# ===========================================================================
# Type aliases
# ===========================================================================

LossFn = Callable[[Data, Data], Scalar]

# ===========================================================================
# LossOutput
# ===========================================================================


@dataclass(frozen=True, kw_only=True)
class LossOutput:
    """Output of a loss computation.

    Attributes:
        loss: Per-sample total loss, shape (batch,).
        per_leaf: Per-modality breakdown, each shape (batch,).
        metrics: Diagnostics dict (e.g. unweighted loss, per-leaf details).
    """

    loss: Scalar
    per_leaf: ScalarTree
    metrics: dict[str, Scalar | ScalarTree]


jax.tree_util.register_dataclass(
    LossOutput,
    data_fields=["loss", "per_leaf", "metrics"],
    meta_fields=[],
)

# ===========================================================================
# Standalone loss functions
# ===========================================================================


def mse_loss(pred: Data, target: Data) -> Scalar:
    """Mean squared error, averaged over non-batch dimensions.

    Args:
        pred: Predicted values, shape (batch, *data_shape).
        target: Target values, shape (batch, *data_shape).

    Returns:
        Per-sample MSE, shape (batch,).
    """
    sq = (pred - target) ** 2
    return jnp.mean(sq, axis=tuple(range(1, sq.ndim)))


def ce_loss(logits: Data, target_probs: Data) -> Scalar:
    """Cross-entropy loss: -sum(target * log_softmax(logits)).

    Averaged over spatial (non-batch, non-class) dimensions.

    Args:
        logits: Unnormalized log-probabilities, shape (batch, *spatial, num_classes).
        target_probs: Target probabilities, shape (batch, *spatial, num_classes).

    Returns:
        Per-sample cross-entropy, shape (batch,).
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    per_position = -jnp.sum(target_probs * log_probs, axis=-1)
    return jnp.mean(per_position, axis=tuple(range(1, per_position.ndim)))


def kl_loss(logits: Data, target_probs: Data) -> Scalar:
    """KL divergence: sum(target * (log(target) - log_softmax(logits))).

    Averaged over spatial (non-batch, non-class) dimensions.

    Args:
        logits: Unnormalized log-probabilities, shape (batch, *spatial, num_classes).
        target_probs: Target probabilities, shape (batch, *spatial, num_classes).

    Returns:
        Per-sample KL divergence, shape (batch,).
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    log_target = jnp.log(jnp.clip(target_probs, 1e-8))
    per_position = jnp.sum(target_probs * (log_target - log_probs), axis=-1)
    return jnp.mean(per_position, axis=tuple(range(1, per_position.ndim)))


def default_loss_fn(process: CorruptionProcess) -> LossFn:
    """Return the default loss function for a given process type.

    Args:
        process: A corruption process instance.

    Returns:
        ``mse_loss`` for interpolation processes,
        ``ce_loss`` for discrete processes,
        ``kl_loss`` for simplicial processes.

    Raises:
        TypeError: If the process type is not recognized.
    """
    if isinstance(process, InterpolationProcess):
        return mse_loss
    if isinstance(process, DiscreteCorruptionProcess):
        return ce_loss
    if isinstance(process, SimplicialCorruptionProcess):
        return kl_loss
    raise TypeError(f"No default loss function for process type {type(process)}")


# ===========================================================================
# Loss ABC
# ===========================================================================


@dataclass(frozen=True, kw_only=True)
class Loss(ABC):
    """Abstract base class for training losses.

    Attributes:
        process: The corruption process.
        weight_fn: Optional per-sample weighting function (aux -> scalar).
    """

    process: CorruptionProcess
    weight_fn: Callable[[AuxInfoTree], ScalarTree] | None

    @abstractmethod
    def __call__(
        self,
        predictor: DiffusionPredictor,
        x_0: DataTree,
        x_t: DataTree,
        t: TimeTree,
        aux: AuxInfoTree,
        cond: ConditioningTree | None,
        **kwargs,
    ) -> LossOutput: ...


jax.tree_util.register_dataclass(
    Loss,
    data_fields=["process"],
    meta_fields=["weight_fn"],
)

LossTree = PyTree[Loss, TreeShape]


# ===========================================================================
# MultimodalLoss
# ===========================================================================


class MultimodalLoss(Loss):
    """Loss for multimodal data, wrapping a pytree of per-modality losses.

    Mirrors ``MultimodalSampler`` pattern.

    Attributes:
        losses: Pytree of ``Loss`` instances (one per modality).
        leaf_weights: Optional per-modality contribution weights.
    """

    losses: LossTree
    leaf_weights: PyTree[float] | None

    def __init__(
        self,
        losses: LossTree,
        leaf_weights: PyTree[float] | None = None,
    ):
        _is_loss = lambda x: isinstance(x, Loss)
        processes = jax.tree_util.tree_map(
            lambda lo: lo.process, losses, is_leaf=_is_loss
        )
        super().__init__(
            process=MultimodalCorruptionProcess(processes),
            weight_fn=None,
        )
        self.losses: LossTree = losses
        self.leaf_weights: PyTree[float] | None = leaf_weights

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
        _is_loss = lambda x: isinstance(x, Loss)

        # Call predictor once on full multimodal data
        prediction = predictor(x_t, None, t, cond, aux)

        # Wrap each per-leaf prediction into a trivial predictor
        def _make_cached_predictor(pred):
            def _cached(x, s, t, cond, aux):
                return pred

            return _cached

        _is_pred = lambda x: isinstance(x, Prediction)
        cached_predictors = jax.tree_util.tree_map(
            _make_cached_predictor, prediction, is_leaf=_is_pred
        )

        # Call each per-leaf loss on its modality's data
        leaf_outputs = jax.tree_util.tree_map(
            lambda lo, cp, x0i, xti, ti, auxi: lo(
                cp, x0i, xti, ti, auxi, cond, **kwargs
            ),
            self.losses,
            cached_predictors,
            x_0,
            x_t,
            t,
            aux,
            is_leaf=_is_loss,
        )

        # Extract per-leaf losses
        _is_loss_output = lambda x: isinstance(x, LossOutput)
        per_leaf = jax.tree_util.tree_map(
            lambda out: out.loss, leaf_outputs, is_leaf=_is_loss_output
        )

        # Apply leaf weights
        if self.leaf_weights is not None:
            weighted = jax.tree_util.tree_map(
                lambda scalar, w: scalar * w, per_leaf, self.leaf_weights
            )
        else:
            weighted = per_leaf

        # Sum across modalities
        flat_losses = jax.tree_util.tree_leaves(weighted)
        total = flat_losses[0]
        for fl in flat_losses[1:]:
            total = total + fl

        # Collect per-leaf metrics
        metrics: dict[str, Scalar | ScalarTree] = {
            "per_leaf_loss": per_leaf,
        }

        return LossOutput(loss=total, per_leaf=per_leaf, metrics=metrics)
