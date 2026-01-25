"""
Base class for diffusion processes.
"""

from abc import ABC, abstractmethod

import jax
from jaxtyping import PyTree

from diffusionlab.typing import (
    AuxInfoTree,
    CoefficientsTree,
    DataTree,
    PredictionTree,
    PRNGKey,
    TimeTree,
)
from diffusionlab.utils.tree_ops import tree_map_with_key


class CorruptionProcess(ABC):
    """Base class for data corruption / diffusion forward processes.

    A corruption process defines a forward mapping (x_0 -> x_t) for a
    (potentially tree-structured) data object `x` and time `t`.
    """

    @abstractmethod
    def forward(
        self, key: PRNGKey, x: DataTree, t: TimeTree
    ) -> tuple[DataTree, AuxInfoTree]:
        """
        Corrupt the data x at time t, and return the corrupted data and auxiliary information.

        Args:
            key: A PRNG key.
            x: The data to corrupt.
            t: The time at which to corrupt the data.

        Returns:
            A tuple of the corrupted data and auxiliary information.
        """
        raise NotImplementedError

    @abstractmethod
    def forward_multiple(
        self, key: PRNGKey, x: DataTree, ts: list[TimeTree]
    ) -> list[tuple[DataTree, AuxInfoTree]]:
        """
        Corrupt the data x at each time t, and return the corrupted data and auxiliary information, for each time.

        Args:
            key: A PRNG key.
            x: The data to corrupt.
            ts: The times at which to corrupt the data.

        Returns:
            A list of tuples of the corrupted data and auxiliary information, for each time.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_from_terminal(
        self, key: PRNGKey, batch_size: int
    ) -> tuple[DataTree, AuxInfoTree]:
        """
        Sample from (approximately) the terminal distribution x_1.

        Args:
            key: A PRNG key.
            batch_size: The number of samples to draw.

        Returns:
            A tuple of the sampled data and auxiliary information.
        """
        raise NotImplementedError

    @abstractmethod
    def convert_prediction(
        self,
        prediction: PredictionTree,
        x_t: DataTree,
        s: TimeTree | None,
        t: TimeTree,
        to_kind: PyTree[str],
    ) -> PredictionTree:
        """
        Convert the prediction(s) to the requested kind(s).

        Args:
            prediction: The prediction(s) to convert.
            x_t: The corrupted data at time t.
            s: An optional time which, if provided and the prediction involves an intermediate time 0 < s < t, specifies the time s.
            t: The time at which the prediction is evaluated.
            to_kind: The kind(s) of prediction to convert the predictions) to.

        Returns:
            A prediction(s) with the requested kind(s).
        """
        raise NotImplementedError

    @abstractmethod
    def get_coefficients(self, t: TimeTree) -> CoefficientsTree:
        """
        Get the corruption coefficients for the given time t.

        Args:
            t: The time at which to get the corruption coefficients.

        Returns:
            The corruption coefficients for the given time; returns things like scaling factors (continuous) or transition rates (discrete).
        """
        raise NotImplementedError


class MultimodalCorruptionProcess(CorruptionProcess):
    """A nested corruption process involving multiple corruption processes on independent data."""

    def __init__(self, processes: PyTree[CorruptionProcess, "T"]):
        self.processes: PyTree[CorruptionProcess, "T"] = processes

    def forward(
        self, key: PRNGKey, x: DataTree, t: TimeTree
    ) -> tuple[DataTree, AuxInfoTree]:
        """
        Corrupt the nested data x at time t, and return the corrupted data and auxiliary information.

        Args:
            key: A PRNG key.
            x: The nested data to corrupt; may be a pytree matching the modality structure.
            t: The time at which to corrupt the data; may be a pytree matching the modality structure.

        Returns:
            A tuple of the corrupted nested data and auxiliary information; may be pytrees matching the modality structure.
        """
        out = tree_map_with_key(
            lambda ki, proc, xi, ti: proc.forward(ki, xi, ti),
            key,
            self.processes,
            x,
            t,
        )
        x_t = jax.tree_util.tree_map(lambda proc, pair: pair[0], self.processes, out)
        aux = jax.tree_util.tree_map(lambda proc, pair: pair[1], self.processes, out)
        return x_t, aux

    def forward_multiple(
        self, key: PRNGKey, x: DataTree, ts: list[TimeTree]
    ) -> list[tuple[DataTree, AuxInfoTree]]:
        """
        Corrupt the nested data x at each time t, and return the corrupted data and auxiliary information, for each time.

        Args:
            key: A PRNG key.
            x: The nested data to corrupt; may be a pytree matching the modality structure.
            ts: The times at which to corrupt the data; may be a pytree matching the modality structure.

        Returns:
            A list of tuples of the corrupted nested data and auxiliary information, for each time; may be pytrees matching the modality structure.
        """
        # Reorganize a list-of-time-pytrees into a per-leaf list of times, so
        # each modality leaf process receives a `list[Time]` (or, more generally,
        # a `list[TimeTreeLeaf]`) matching its own expected signature.
        ts_by_leaf = jax.tree_util.tree_map(lambda *tis: list(tis), *ts)
        out = tree_map_with_key(
            lambda ki, proc, xi, tis: proc.forward_multiple(ki, xi, tis),
            key,
            self.processes,
            x,
            ts_by_leaf,
        )

        # `out` is a pytree with the same structure as `x`, where each leaf is
        # a list of (x_t_leaf, aux_leaf) pairs. Transpose it back into a list of
        # full pytrees, one per time.
        results: list[tuple[DataTree, AuxInfoTree]] = []
        for k in range(len(ts)):
            x_tk = jax.tree_util.tree_map(
                lambda proc, leaf_out: leaf_out[k][0], self.processes, out
            )
            aux_k = jax.tree_util.tree_map(
                lambda proc, leaf_out: leaf_out[k][1], self.processes, out
            )
            results.append((x_tk, aux_k))
        return results

    def sample_from_terminal(
        self, key: PRNGKey, batch_size: int
    ) -> tuple[DataTree, AuxInfoTree]:
        """
        Sample from (approximately) the terminal distribution x_1 for each modality.

        Args:
            key: A PRNG key.
            batch_size: The number of samples to draw.

        Returns:
            A tuple of the sampled data and auxiliary information; may be pytrees matching the modality structure.
        """
        out = tree_map_with_key(
            lambda ki, proc: proc.sample_from_terminal(ki, batch_size),
            key,
            self.processes,
        )
        x_1 = jax.tree_util.tree_map(lambda proc, pair: pair[0], self.processes, out)
        aux = jax.tree_util.tree_map(lambda proc, pair: pair[1], self.processes, out)
        return x_1, aux

    def convert_prediction(
        self,
        prediction: PredictionTree,
        x_t: DataTree,
        s: TimeTree | None,
        t: TimeTree,
        to_kind: PyTree[str],
    ) -> PredictionTree:
        """
        Convert the prediction(s) to the requested kind(s).

        Args:
            prediction: The prediction(s) to convert.
            x_t: The corrupted data at time t.
            s: An optional time which, if provided and the prediction involves an intermediate time 0 < s < t, specifies the time s.
            t: The time at which the prediction is evaluated.
            to_kind: The kind(s) of prediction to convert the predictions) to.

        Returns:
            A prediction(s) with the requested kind(s).
        """
        if s is None:
            output = jax.tree_util.tree_map(
                lambda proc, pi, xti, ti, tk: proc.convert_prediction(
                    pi, xti, None, ti, tk
                ),
                self.processes,
                prediction,
                x_t,
                t,
                to_kind,
            )
        else:
            output = jax.tree_util.tree_map(
                lambda proc, pi, xti, si, ti, tk: proc.convert_prediction(
                    pi, xti, si, ti, tk
                ),
                self.processes,
                prediction,
                x_t,
                s,
                t,
                to_kind,
            )
        return output

    def get_coefficients(self, t: TimeTree) -> CoefficientsTree:
        """
        Get the corruption coefficients for the given time t for each modality.

        Args:
            t: The time at which to get the corruption coefficients; may be a pytree matching the modality structure.

        Returns:
            The corruption coefficients for the given time; pytree matching the modality structure.
        """
        return jax.tree_util.tree_map(
            lambda proc, ti: proc.get_coefficients(ti),
            self.processes,
            t,
        )
