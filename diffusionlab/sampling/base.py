"""Sampler base class, multimodal sampler, and sampling orchestrator."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
from jaxtyping import PyTree

from diffusionlab.processes.base import (
    CorruptionProcess,
    MultimodalCorruptionProcess,
)
from diffusionlab.typing import (
    AuxInfoTree,
    ConditioningTree,
    DataTree,
    DiffusionPredictor,
    PredictionTree,
    PRNGKey,
    TimeTree,
    TreeShape,
)
from diffusionlab.utils.tree_ops import tree_map_with_key


@dataclass(frozen=True, kw_only=True)
class SamplerStep:
    """State of the sampler at a single time step.

    Attributes:
        x: The current data (corrupted sample at time t).
        t: The current time.
        aux: Auxiliary schedule information at time t.
    """

    x: DataTree
    t: TimeTree
    aux: AuxInfoTree


jax.tree_util.register_dataclass(
    SamplerStep,
    data_fields=["x", "t", "aux"],
    meta_fields=[],
)


@dataclass(frozen=True, kw_only=True)
class SamplingTransform(ABC):
    """Abstract base class for sampling transforms.

    Transforms intercept the sampling loop at two points:

    1. ``transform_prediction`` — modify the model prediction *before*
       the sampler update (e.g. classifier-free guidance, DPS).
    2. ``transform_step`` — modify the ``SamplerStep`` *after* the
       sampler update.

    Subclasses override one or both hooks.  The default implementations
    are identity functions.
    """

    def transform_prediction(
        self,
        key: PRNGKey,
        predictor: DiffusionPredictor,
        prediction: PredictionTree,
        step: SamplerStep,
        s: TimeTree,
        cond: ConditioningTree | None,
    ) -> PredictionTree:
        """Modify/replace the prediction BEFORE the sampler update.

        Args:
            key: PRNG key.
            predictor: The model predictor, so transforms can call the
                model again (e.g. CFG).
            prediction: The conditional prediction.
            step: Current sampler state (contains x, t, aux).
            s: Target (earlier) time.
            cond: Conditioning data.

        Returns:
            Possibly-modified prediction.
        """
        return prediction

    def transform_step(
        self,
        key: PRNGKey,
        step: SamplerStep,
        prediction: PredictionTree,
        prev_step: SamplerStep,
    ) -> SamplerStep:
        """Modify the ``SamplerStep`` AFTER the sampler update.

        Args:
            key: PRNG key.
            step: The new sampler state produced by ``sampler.update``.
            prediction: The (possibly transformed) prediction used for the update.
            prev_step: The sampler state *before* the update.

        Returns:
            Possibly-modified sampler step.
        """
        return step


@dataclass(frozen=True, kw_only=True)
class Sampler(ABC):
    """Abstract base class for reverse-process samplers.

    Each sampler is tied to a specific `CorruptionProcess` and
    implements the three-step protocol:

    1. `initialize` — wrap terminal noise into a `SamplerStep`.
    2. `update` — given a model prediction, step from time *t* to *s*.
    3. `finalize` — extract clean data from the last prediction.

    Attributes:
        process: The corruption process this sampler reverses.
    """

    process: CorruptionProcess

    @abstractmethod
    def initialize(self, key: PRNGKey, x_T: DataTree, T: TimeTree) -> SamplerStep:
        """Create the initial sampler state from terminal noise.

        Args:
            key: PRNG key.
            x_T: Terminal-distribution sample.
            T: Initial (largest) time.

        Returns:
            A `SamplerStep` at time *T*.
        """
        ...

    @abstractmethod
    def update(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        s: TimeTree,
    ) -> SamplerStep:
        """Advance the sampler from the current time to time *s*.

        Args:
            key: PRNG key (for stochastic samplers).
            prediction: Model prediction at the current time.
            current_step: Current sampler state.
            s: Target (earlier) time.

        Returns:
            A new `SamplerStep` at time *s*.
        """
        ...

    @abstractmethod
    def finalize(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        t_final: TimeTree | None = None,
    ) -> DataTree:
        """Produce clean data from the final prediction.

        Args:
            key: PRNG key.
            prediction: Model prediction at the final time.
            current_step: Final sampler state.
            t_final: The target time of the final prediction step. Needed
                when the prediction kind requires both times for conversion
                (e.g. ``v_st``).

        Returns:
            Estimated clean data.
        """
        ...


jax.tree_util.register_dataclass(
    Sampler,
    data_fields=["process"],
    meta_fields=[],
)
SamplerTree = PyTree[Sampler, TreeShape]


class MultimodalSampler(Sampler):
    """Sampler for multimodal data, wrapping a pytree of per-modality samplers.

    Mirrors `MultimodalCorruptionProcess`.

    Attributes:
        samplers: Pytree of `Sampler` instances (one per modality).
    """

    samplers: SamplerTree

    def __init__(self, samplers: SamplerTree):
        processes = jax.tree_util.tree_map(lambda s: s.process, samplers)
        super().__init__(process=MultimodalCorruptionProcess(processes))
        self.samplers: SamplerTree = samplers

    def initialize(self, key: PRNGKey, x_T: DataTree, T: TimeTree) -> SamplerStep:
        out = tree_map_with_key(
            lambda ki, sampler, xi, ti: sampler.initialize(ki, xi, ti),
            key,
            self.samplers,
            x_T,
            T,
        )
        x = jax.tree_util.tree_map(lambda sampler, step: step.x, self.samplers, out)
        t_out = jax.tree_util.tree_map(lambda sampler, step: step.t, self.samplers, out)
        aux = jax.tree_util.tree_map(lambda sampler, step: step.aux, self.samplers, out)
        return SamplerStep(x=x, t=t_out, aux=aux)

    def update(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        s: TimeTree,
    ) -> SamplerStep:
        per_modal_steps = jax.tree_util.tree_map(
            lambda sampler, xi, ti, auxi: SamplerStep(x=xi, t=ti, aux=auxi),
            self.samplers,
            current_step.x,
            current_step.t,
            current_step.aux,
        )
        out = tree_map_with_key(
            lambda ki, sampler, pred, step, si: sampler.update(ki, pred, step, si),
            key,
            self.samplers,
            prediction,
            per_modal_steps,
            s,
        )
        x = jax.tree_util.tree_map(lambda sampler, step: step.x, self.samplers, out)
        t_out = jax.tree_util.tree_map(lambda sampler, step: step.t, self.samplers, out)
        aux = jax.tree_util.tree_map(lambda sampler, step: step.aux, self.samplers, out)
        return SamplerStep(x=x, t=t_out, aux=aux)

    def finalize(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        t_final: TimeTree | None = None,
    ) -> DataTree:
        per_modal_steps = jax.tree_util.tree_map(
            lambda sampler, xi, ti, auxi: SamplerStep(x=xi, t=ti, aux=auxi),
            self.samplers,
            current_step.x,
            current_step.t,
            current_step.aux,
        )
        if t_final is None:
            return tree_map_with_key(
                lambda ki, sampler, pred, step: sampler.finalize(ki, pred, step),
                key,
                self.samplers,
                prediction,
                per_modal_steps,
            )
        return tree_map_with_key(
            lambda ki, sampler, pred, step, tf: sampler.finalize(ki, pred, step, tf),
            key,
            self.samplers,
            prediction,
            per_modal_steps,
            t_final,
        )


def _sample_impl(
    key: PRNGKey,
    sampler: Sampler,
    predictor: DiffusionPredictor,
    batch_size: int,
    time_schedule: list[TimeTree],
    cond: ConditioningTree | None,
    transforms: list[SamplingTransform],
    *,
    save_trajectory: bool,
) -> tuple[DataTree, list[SamplerStep] | None]:
    """Shared sampling loop used by ``sample`` and ``sample_trajectory``."""
    num_times = len(time_schedule)

    key, k_term, k_init = jax.random.split(key, 3)

    # 1. Sample from the terminal distribution
    x_T, _aux = sampler.process.sample_from_terminal(k_term, batch_size)

    # 2. Initialize
    step = sampler.initialize(k_init, x_T, time_schedule[0])
    trajectory: list[SamplerStep] | None = [step] if save_trajectory else None

    # 3. Intermediate steps
    for i in range(1, num_times - 1):
        key, k_step = jax.random.split(key)
        s = time_schedule[i]
        prediction = predictor(step.x, s, step.t, cond, step.aux)

        # Apply prediction transforms
        for transform in transforms:
            key, k_t = jax.random.split(key)
            prediction = transform.transform_prediction(
                k_t,
                predictor,
                prediction,
                step,
                s,
                cond,
            )

        prev_step = step
        step = sampler.update(k_step, prediction, step, s)

        # Apply step transforms
        for transform in transforms:
            key, k_t = jax.random.split(key)
            step = transform.transform_step(k_t, step, prediction, prev_step)

        if trajectory is not None:
            trajectory.append(step)

    # 4. Final prediction + finalize
    key, k_final = jax.random.split(key)
    prediction = predictor(
        step.x,
        time_schedule[num_times - 1],
        step.t,
        cond,
        step.aux,
    )

    # Apply prediction transforms on final step too
    for transform in transforms:
        key, k_t = jax.random.split(key)
        prediction = transform.transform_prediction(
            k_t,
            predictor,
            prediction,
            step,
            time_schedule[num_times - 1],
            cond,
        )

    final = sampler.finalize(k_final, prediction, step, time_schedule[num_times - 1])
    return final, trajectory


def sample(
    key: PRNGKey,
    sampler: Sampler,
    predictor: DiffusionPredictor,
    batch_size: int,
    time_schedule: list[TimeTree],
    cond: ConditioningTree | None = None,
    transforms: list[SamplingTransform] | None = None,
) -> DataTree:
    """Run the full reverse-process sampling loop.

    Args:
        key: PRNG key.
        sampler: A `Sampler` (or `MultimodalSampler`) object.
        predictor: A `DiffusionPredictor` object.
        batch_size: Number of samples to generate.
        time_schedule: A ``list[TimeTree]`` of length ``num_steps + 1``,
            ordered from ``t=1`` (noisy) to ``t~0`` (clean).  Typically
            produced by `uniform_schedule` or `edm_schedule`.  For
            multimodal samplers each element is a pytree of ``Time``
            arrays matching the sampler structure.
        cond: Optional conditioning data passed through to *predictor*.
        transforms: Optional list of ``SamplingTransform`` instances applied
            sequentially at each step.

    Returns:
        Estimated clean data.
    """
    final, _ = _sample_impl(
        key,
        sampler,
        predictor,
        batch_size,
        time_schedule,
        cond,
        transforms if transforms is not None else [],
        save_trajectory=False,
    )
    return final


def sample_trajectory(
    key: PRNGKey,
    sampler: Sampler,
    predictor: DiffusionPredictor,
    batch_size: int,
    time_schedule: list[TimeTree],
    cond: ConditioningTree | None = None,
    transforms: list[SamplingTransform] | None = None,
) -> tuple[DataTree, list[SamplerStep]]:
    """Run the full reverse-process sampling loop and return the trajectory.

    Identical to `sample` but additionally returns the intermediate sampler
    states, useful for visualising the denoising process or diagnostics.

    .. warning::
        The trajectory stores every intermediate ``SamplerStep`` in memory.
        For large batch sizes or high-dimensional data prefer `sample`.

    Args:
        key: PRNG key.
        sampler: A `Sampler` (or `MultimodalSampler`) object.
        predictor: A `DiffusionPredictor` object.
        batch_size: Number of samples to generate.
        time_schedule: A ``list[TimeTree]`` of length ``num_steps + 1``,
            ordered from ``t=1`` (noisy) to ``t~0`` (clean).  Typically
            produced by `uniform_schedule` or `edm_schedule`.  For
            multimodal samplers each element is a pytree of ``Time``
            arrays matching the sampler structure.
        cond: Optional conditioning data passed through to *predictor*.
        transforms: Optional list of ``SamplingTransform`` instances applied
            sequentially at each step.

    Returns:
        A tuple ``(final_data, trajectory)`` where:

        - *final_data* is the estimated clean data (same as `sample`).
        - *trajectory* is a list of `SamplerStep` objects of length
          ``num_steps``.  The first element is the initial noisy state
          (at ``t=1``), and subsequent elements are the states after each
          sampler update.
    """
    final, trajectory = _sample_impl(
        key,
        sampler,
        predictor,
        batch_size,
        time_schedule,
        cond,
        transforms if transforms is not None else [],
        save_trajectory=True,
    )
    assert trajectory is not None  # guaranteed by save_trajectory=True
    return final, trajectory
