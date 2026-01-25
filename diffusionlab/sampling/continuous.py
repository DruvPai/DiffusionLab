"""Samplers for Gaussian diffusion processes."""

from typing import cast

import jax
from jax import numpy as jnp

from diffusionlab.processes.gaussian import GaussianDiffusionProcess
from diffusionlab.processes.interpolation import InterpolationProcess
from diffusionlab.sampling.base import Sampler, SamplerStep
from diffusionlab.typing import ContinuousData, PredictionTree, PRNGKey, Time
from diffusionlab.utils.tree_ops import bcast_right


class DDMSampler(Sampler):
    """Denoising Diffusion Implicit Models sampler (Song et al. 2021).

    Interpolates between deterministic DDIM (`eta=0`) and stochastic
    DDPM (`eta=1`) via the noise injection parameter *eta*.

    Attributes:
        process: A `GaussianDiffusionProcess` object.
        eta: Stochasticity parameter.  0 = deterministic, 1 = DDPM.
    """

    eta: float

    def __init__(self, process: GaussianDiffusionProcess, eta: float = 0.0):
        super().__init__(process=process)
        self.eta: float = eta

    def initialize(self, key: PRNGKey, x_T: ContinuousData, T: Time) -> SamplerStep:
        aux = self.process.get_coefficients(T)
        return SamplerStep(x=x_T, t=T, aux=aux)

    def update(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        s: Time,
    ) -> SamplerStep:
        process = cast(GaussianDiffusionProcess, self.process)
        x_t = current_step.x
        t = current_step.t
        ndim = x_t.ndim

        # Schedule info at current and target times
        coeffs_t = process.get_coefficients(t)
        coeffs_s = process.get_coefficients(s)
        alpha_t = bcast_right(coeffs_t["alpha"], ndim)
        sigma_t = bcast_right(coeffs_t["sigma"], ndim)
        alpha_s = bcast_right(coeffs_s["alpha"], ndim)
        sigma_s = bcast_right(coeffs_s["sigma"], ndim)

        # Convert to x_0 and eps predictions
        x_0_hat = process.convert_prediction(prediction, x_t, s, t, "x_0").value
        eps_hat = process.convert_prediction(prediction, x_t, s, t, "eps").value

        # DDIM noise injection
        #   sigma_tilde^2 = eta^2 * (sigma_s^2 / sigma_t^2)
        #                   * (sigma_t^2 - alpha_t^2 * sigma_s^2 / alpha_s^2)
        sigma_tilde_sq = (
            self.eta**2
            * (sigma_s**2 / jnp.maximum(sigma_t**2, 1e-20))
            * (sigma_t**2 - alpha_t**2 * sigma_s**2 / jnp.maximum(alpha_s**2, 1e-20))
        )
        sigma_tilde_sq = jnp.maximum(sigma_tilde_sq, 0.0)
        sigma_tilde = jnp.sqrt(sigma_tilde_sq)

        # Deterministic component coefficient
        coeff_eps = jnp.sqrt(jnp.maximum(sigma_s**2 - sigma_tilde_sq, 0.0))

        # Step
        z = jax.random.normal(key, x_t.shape)
        x_s = alpha_s * x_0_hat + coeff_eps * eps_hat + sigma_tilde * z

        return SamplerStep(x=x_s, t=s, aux=coeffs_s)

    def finalize(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        t_final: Time | None = None,
    ) -> ContinuousData:
        return self.process.convert_prediction(
            prediction, current_step.x, t_final, current_step.t, "x_0"
        ).value


class EulerSampler(Sampler):
    """Deterministic Euler ODE integrator via velocity parameterisation.

    Converts predictions to a velocity and takes an Euler step:
    ``x_s = x_t + (s - t) * v_hat``.

    The velocity kind is inferred from the prediction: if the prediction
    kind is ``"x_s"`` or ``"v_st"`` (flow-map quantities), the prediction
    is converted to ``"v_st"``; otherwise it is converted to ``"v_t"``.

    Attributes:
        process: An `InterpolationProcess` (or subclass) object.
    """

    def __init__(self, process: InterpolationProcess):
        assert isinstance(process, InterpolationProcess)
        super().__init__(process=process)

    def initialize(self, key: PRNGKey, x_T: ContinuousData, T: Time) -> SamplerStep:
        aux = self.process.get_coefficients(T)
        return SamplerStep(x=x_T, t=T, aux=aux)

    def update(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        s: Time,
    ) -> SamplerStep:
        x_t = current_step.x
        t = current_step.t
        ndim = x_t.ndim

        # Infer velocity kind from prediction
        velocity_kind = "v_st" if prediction.kind in ("x_s", "v_st") else "v_t"

        # Get velocity prediction
        v_hat = self.process.convert_prediction(
            prediction, x_t, s, t, velocity_kind
        ).value

        # Euler step: x_s = x_t + (s - t) * v_hat
        dt = bcast_right(s - t, ndim)
        x_s = x_t + dt * v_hat

        aux = self.process.get_coefficients(s)
        return SamplerStep(x=x_s, t=s, aux=aux)

    def finalize(
        self,
        key: PRNGKey,
        prediction: PredictionTree,
        current_step: SamplerStep,
        t_final: Time | None = None,
    ) -> ContinuousData:
        return self.process.convert_prediction(
            prediction, current_step.x, t_final, current_step.t, "x_0"
        ).value
