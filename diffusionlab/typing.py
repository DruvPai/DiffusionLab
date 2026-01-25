"""Type definitions and short-hands for common data types."""

from dataclasses import dataclass, field
from typing import Protocol

import jax
from jaxtyping import Array, Bool, Float, Int, Key, PyTree

ContinuousData = Float[Array, "batch *#?data_shape"]
DiscreteData = Int[Array, "batch *#?seq_len"]
ProbabilisticData = Float[Array, "batch *#?seq_len num_classes"]
Data = ContinuousData | DiscreteData | ProbabilisticData
Mask = Bool[Array, "batch *#?seq_len"]
Coefficients = dict[str, Array]
AuxInfo = dict[str, Array]

Time = Float[Array, "batch"]
Scalar = Float[Array, "batch"]

TreeShape = "T"
DataTree = PyTree[Data, TreeShape]
MaskTree = PyTree[Mask, TreeShape]
TimeTree = PyTree[Time, TreeShape]
ScalarTree = PyTree[Scalar, TreeShape]
CoefficientsTree = PyTree[Coefficients, TreeShape]
AuxInfoTree = PyTree[AuxInfo, TreeShape]

Conditioning = Float[Array, "batch *#?cond_shape"]
ConditioningTree = PyTree[Conditioning, TreeShape]

HiddenState = Float[Array, "batch *#?hidden_shape"]
EmbeddingVector = Float[Array, "batch emb_dim"]
SequenceEmbedding = Float[Array, "batch seq_len emb_dim"]

Image = Float[Array, "batch ?height ?width ?channels"]

PRNGKey = Key[Array, ""]


@dataclass(frozen=True, kw_only=True)
class Prediction:
    """A model prediction with its value and kind.

    Attributes:
        value: The predicted data array.
        kind: The type of prediction (e.g. ``'eps'``, ``'x_0'``, ``'v'``). Static (not traced by JAX).
    """

    value: Data
    kind: str = field(metadata=dict(static=True))


jax.tree_util.register_dataclass(
    Prediction,
    data_fields=["value"],
    meta_fields=["kind"],
)

PredictionTree = PyTree[Prediction, TreeShape]


class DiffusionPredictor(Protocol):
    """Protocol for diffusion model predictors."""

    def __call__(
        self,
        x: DataTree,
        s: TimeTree | None,
        t: TimeTree,
        cond: ConditioningTree | None,
        aux: AuxInfoTree | None,
    ) -> PredictionTree: ...
