"""Sampler for discrete (categorical) diffusion processes."""

from typing import cast

import jax
from jax import numpy as jnp

from diffusionlab.processes.discrete import DiscreteCorruptionProcess
from diffusionlab.sampling.base import Sampler, SamplerStep
from diffusionlab.typing import DiscreteData, PredictionTree, PRNGKey, Time


class DiscreteAncestralSampler(Sampler):
    """Ancestral sampler for discrete corruption processes.

    At each step, computes the denoising posterior `q(x_s | x_t, x_0_hat)`
    via `convert_prediction` and samples from it categorically.
    Finalization returns the argmax of the predicted clean distribution.

    Attributes:
        process: A `DiscreteCorruptionProcess` object.
    """

    def __init__(self, process: DiscreteCorruptionProcess):
        super().__init__(process=process)
        assert isinstance(process, DiscreteCorruptionProcess)

    def initialize(self, key: PRNGKey, x_T: DiscreteData, T: Time) -> SamplerStep:
        aux = self.process.get_coefficients(T)
        return SamplerStep(x=x_T, t=T, aux=aux)

    def update(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        s: Time,
    ) -> SamplerStep:
        process = cast(DiscreteCorruptionProcess, self.process)
        x_t = current_step.x
        t = current_step.t

        # Posterior probabilities q(x_s | x_t, x_0_hat)
        posterior = process.convert_prediction(prediction, x_t, s, t, "x_s")
        probs = posterior.value  # (batch, *seq_shape, num_classes)

        # Sample from categorical over the last axis
        original_shape = probs.shape[:-1]
        num_classes = probs.shape[-1]
        flat_logits = jnp.log(jnp.maximum(probs.reshape(-1, num_classes), 1e-20))
        flat_samples = jax.random.categorical(key, flat_logits)
        x_s = flat_samples.reshape(original_shape)

        aux = process.get_coefficients(s)
        return SamplerStep(x=x_s, t=s, aux=aux)

    def finalize(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        t_final: Time | None = None,
    ) -> DiscreteData:
        x_0_pred = self.process.convert_prediction(
            prediction, current_step.x, None, current_step.t, "x_0"
        )
        return jnp.argmax(x_0_pred.value, axis=-1)
