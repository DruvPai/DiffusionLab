"""Abstract base classes for diffusion model architectures."""

from abc import ABC, abstractmethod

import equinox as eqx
from jaxtyping import PyTree

from diffusionlab.typing import (
    AuxInfoTree,
    ConditioningTree,
    DataTree,
    PredictionTree,
    TimeTree,
)


class DiffusionModel(eqx.Module, ABC):
    """Abstract base class for diffusion model architectures.

    Attributes:
        prediction_kinds: PyTree of the kind of predictions made at each leaf of the data PyTree.
    """

    prediction_kinds: PyTree[str] = eqx.field(static=True)

    @abstractmethod
    def __call__(
        self,
        x: DataTree,
        s: TimeTree | None,
        t: TimeTree,
        cond: ConditioningTree | None,
        aux: AuxInfoTree | None,
    ) -> PredictionTree:
        """Forward pass: predict from noisy data x at time t.

        Args:
            x: Noisy data PyTree of the same shape as `prediction_kinds`.
            s: Optional target time for flow maps (pass ``None`` if unused).
            t: Diffusion time PyTree of the same shape as `prediction_kinds`.
            cond: Optional conditioning information PyTree of the same shape as `prediction_kinds`, each leaf is a PyTree of variable topology.
            aux: Optional auxiliary information PyTree; uses the same shape as `prediction_kinds`, each leaf is an AuxInfo (= dict[str, Array]).

        Returns:
            Prediction(s) wrapped in the same PyTree structure as the data.
        """
        raise NotImplementedError
