"""Gaussian diffusion processes.

Special case of interpolation processes where the source distribution
is a standard Gaussian: x(t) = α(t) x_0 + σ(t) ε, ε ~ N(0, I).

This module provides:
- GaussianDiffusion: abstract base for Gaussian-source interpolation
- VPCosine: variance-preserving with cosine schedule
- VPLinear: variance-preserving with linear beta schedule (DDPM)
- VE: variance-exploding with geometric σ schedule
- EDM: Karras et al. (2022) power-law σ schedule
- FlowMatching: linear interpolation (optimal transport / rectified flow)
"""

import jax
from jax import numpy as jnp

from diffusionlab.processes.interpolation import InterpolationProcess
from diffusionlab.typing import (
    AuxInfo,
    ContinuousData,
    Prediction,
    PRNGKey,
    Scalar,
    Time,
)
from diffusionlab.utils.tree_ops import bcast_right


class GaussianDiffusionProcess(InterpolationProcess):
    """Interpolation process with standard Gaussian source distribution.

    x(t) = α(t) x_0 + σ(t) ε,  ε ~ N(0, I),  t ∈ [0, 1].

    Since the source is Gaussian, ε (eps) replaces the generic z from
    InterpolationProcess, and an additional prediction type "score" is
    available via Tweedie's formula:

        score(x_t, t) ≈ ∇_{x_t} log p_t(x_t) = -ε / σ(t)

    Available prediction kinds:
    - x_0: the original data
    - eps: the Gaussian noise ε
    - v: the velocity field v(t) = α'(t) x_0 + σ'(t) ε
    - score: the score function ∇ log p_t(x_t) ≈ -ε / σ(t)
    - x_s: the estimated noised data at intermediate time s (requires s)
    - v_st: the average velocity between s and t (requires s)

    Attributes:
        data_shape: Shape of a single data sample (excluding batch dim).
    """

    def __init__(self, data_shape: tuple[int, ...]):
        """Initialize the Gaussian diffusion process.

        Args:
            data_shape: Shape of a single data sample (excluding batch dim).
        """
        self.data_shape = data_shape

    def _sample_from_source(self, key: PRNGKey, batch_size: int) -> ContinuousData:
        """Sample ε ~ N(0, I).

        Args:
            key: A PRNG key.
            batch_size: Number of samples to draw.

        Returns:
            Gaussian noise ε of shape (batch_size, *data_shape).
        """
        return jax.random.normal(key, (batch_size, *self.data_shape))

    def _forward_with_random(
        self, x: ContinuousData, r: ContinuousData, t: Time
    ) -> tuple[ContinuousData, AuxInfo]:
        """Compute x(t) = α(t) x_0 + σ(t) ε and return (x_t, aux).

        Overrides parent to label source samples as "eps" instead of "z".

        Args:
            x: Original data x_0, shape (batch, *data_shape).
            r: Gaussian noise ε, shape (batch, *data_shape).
            t: Time, shape (batch,).

        Returns:
            Tuple of (x_t, aux_info) where aux_info uses "eps" key instead of "z".
        """
        x_t, aux = super()._forward_with_random(x, r, t)
        aux["eps"] = aux.pop("z")
        return x_t, aux

    def sample_from_terminal(
        self, key: PRNGKey, batch_size: int
    ) -> tuple[ContinuousData, AuxInfo]:
        """Sample from (approximately) the terminal distribution x_1.

        Overrides parent to label source samples as "eps" instead of "z".

        Args:
            key: A PRNG key.
            batch_size: Number of samples to draw.

        Returns:
            Tuple of (x_1, aux_info) where aux_info uses "eps" key instead of "z".
        """
        x_1, aux = super().sample_from_terminal(key, batch_size)
        aux["eps"] = aux.pop("z")
        return x_1, aux

    def convert_prediction(
        self,
        prediction: Prediction,
        x_t: ContinuousData,
        s: Time | None,
        t: Time,
        to_kind: str,
    ) -> Prediction:
        """Convert a model prediction to the requested kind.

        Extends InterpolationProcess.convert_prediction with:
        - "eps" as a replacement for "z" (the Gaussian noise)
        - "score" via Tweedie's formula: score = -ε / σ(t)

        Conversion algebra for score (beyond standard interpolation):

        From score:
            eps   = -σ(t) · score
            x_0   = (x_t + σ(t)² · score) / α(t)

        To score (from any kind):
            score = -eps / σ(t)  (first convert to eps, then to score)

        Args:
            prediction: The prediction to convert.
            x_t: The corrupted data at time t, shape (batch, *data_shape).
            s: Optional intermediate time for x_s / v_st conversions.
            t: The time t, shape (batch,).
            to_kind: Target prediction kind.

        Returns:
            Prediction of the requested kind.
        """
        # Short-circuit identity
        if prediction.kind == to_kind:
            return prediction

        ndim = x_t.ndim
        coefficients = self.get_coefficients(t)
        sigma = bcast_right(coefficients["sigma"], ndim)  # (batch, 1, ...)

        # --- Normalize input: score/eps -> "z" for parent ---
        if prediction.kind == "score":
            mapped = Prediction(value=-sigma * prediction.value, kind="z")
        elif prediction.kind == "eps":
            mapped = Prediction(value=prediction.value, kind="z")
        else:
            mapped = prediction

        # --- Normalize target kind: eps/score -> "z" for parent ---
        internal_to = "z" if to_kind in ("eps", "score") else to_kind

        # --- Delegate to parent ---
        result = super().convert_prediction(mapped, x_t, s, t, internal_to)

        # --- Remap output: "z" -> eps or score ---
        if to_kind == "eps":
            return Prediction(value=result.value, kind="eps")
        elif to_kind == "score":
            return Prediction(value=-result.value / sigma, kind="score")
        return result


# ===========================================================================
# Concrete Gaussian diffusion schedules
# ===========================================================================


class VPCosine(GaussianDiffusionProcess):
    """Variance-preserving diffusion with cosine schedule.

    α(t) = cos(π t / 2)
    σ(t) = sin(π t / 2)

    Satisfies the VP property α(t)² + σ(t)² = 1.
    Boundaries: α(0) = 1, σ(0) = 0, α(1) = 0, σ(1) = 1.

    Attributes:
        data_shape: Shape of a single data sample (excluding batch dim).
    """

    def alpha(self, t: Time) -> Scalar:
        """α(t) = cos(πt/2).

        Args:
            t: Time, shape (batch,).

        Returns:
            Scaling factor α(t), shape (batch,).
        """
        return jnp.cos(jnp.pi / 2 * t)

    def alpha_prime(self, t: Time) -> Scalar:
        """α'(t) = -(π/2) sin(πt/2).

        Args:
            t: Time, shape (batch,).

        Returns:
            Derivative α'(t), shape (batch,).
        """
        return -jnp.pi / 2 * jnp.sin(jnp.pi / 2 * t)

    def sigma(self, t: Time) -> Scalar:
        """σ(t) = sin(πt/2).

        Args:
            t: Time, shape (batch,).

        Returns:
            Noise level σ(t), shape (batch,).
        """
        return jnp.sin(jnp.pi / 2 * t)

    def sigma_prime(self, t: Time) -> Scalar:
        """σ'(t) = (π/2) cos(πt/2).

        Args:
            t: Time, shape (batch,).

        Returns:
            Derivative σ'(t), shape (batch,).
        """
        return jnp.pi / 2 * jnp.cos(jnp.pi / 2 * t)

    def logsnr(self, t: Time) -> Scalar:
        """log SNR(t) = -2 log tan(πt/2).

        Numerically more stable than the generic 2(log α - log σ).

        Args:
            t: Time, shape (batch,).

        Returns:
            Log signal-to-noise ratio, shape (batch,).
        """
        return -2.0 * jnp.log(jnp.tan(jnp.pi / 2 * t))

    def logsnr_inverse(self, lam: Scalar) -> Time:
        """t = (2/π) arctan(exp(-λ/2)).

        Args:
            lam: Target log-SNR λ, shape (batch,).

        Returns:
            Time t, shape (batch,).
        """
        return 2.0 / jnp.pi * jnp.arctan(jnp.exp(-lam / 2.0))


class VPLinear(GaussianDiffusionProcess):
    """Variance-preserving diffusion with linear beta schedule (DDPM-style).

    β(t)  = β_min + (β_max - β_min) t
    α(t)  = exp(-½ ∫₀ᵗ β(s) ds)
          = exp(-½ (β_min t + (β_max - β_min) t² / 2))
    σ(t)  = √(1 - α(t)²)

    Satisfies the VP property α(t)² + σ(t)² = 1.

    Note: σ'(t) diverges as t → 0. Avoid evaluating at t = 0 exactly.

    Attributes:
        data_shape: Shape of a single data sample (excluding batch dim).
        beta_min: Minimum value of the linear beta schedule.
        beta_max: Maximum value of the linear beta schedule.
    """

    def __init__(
        self,
        data_shape: tuple[int, ...],
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ):
        """Initialize VPLinear diffusion.

        Args:
            data_shape: Shape of a single data sample (excluding batch dim).
            beta_min: Minimum beta value (at t=0).
            beta_max: Maximum beta value (at t=1).
        """
        super().__init__(data_shape)
        self.beta_min: float = beta_min
        self.beta_max: float = beta_max

    def _log_alpha(self, t: Time) -> Scalar:
        """Compute log α(t) = -½ (β_min t + (β_max - β_min) t² / 2).

        Args:
            t: Time, shape (batch,).

        Returns:
            log α(t), shape (batch,).
        """
        return -0.5 * (self.beta_min * t + (self.beta_max - self.beta_min) * t**2 / 2)

    def alpha(self, t: Time) -> Scalar:
        """α(t) = exp(-½ ∫₀ᵗ β(s) ds).

        Args:
            t: Time, shape (batch,).

        Returns:
            Scaling factor α(t), shape (batch,).
        """
        return jnp.exp(self._log_alpha(t))

    def alpha_prime(self, t: Time) -> Scalar:
        """α'(t) = -β(t)/2 · α(t).

        Args:
            t: Time, shape (batch,).

        Returns:
            Derivative α'(t), shape (batch,).
        """
        beta_t = self.beta_min + (self.beta_max - self.beta_min) * t
        return -beta_t / 2 * self.alpha(t)

    def sigma(self, t: Time) -> Scalar:
        """σ(t) = √(1 - α(t)²).

        Args:
            t: Time, shape (batch,).

        Returns:
            Noise level σ(t), shape (batch,).
        """
        return jnp.sqrt(1.0 - self.alpha(t) ** 2)

    def sigma_prime(self, t: Time) -> Scalar:
        """σ'(t) = -α(t) α'(t) / σ(t).

        Warning: diverges as t → 0 since σ(0) = 0.

        Args:
            t: Time, shape (batch,).

        Returns:
            Derivative σ'(t), shape (batch,).
        """
        a = self.alpha(t)
        ap = self.alpha_prime(t)
        s = self.sigma(t)
        return -a * ap / s

    def logsnr(self, t: Time) -> Scalar:
        """log SNR(t) = 2 log α(t) - log(1 - α(t)²).

        More numerically stable than the generic formula for small t.

        Args:
            t: Time, shape (batch,).

        Returns:
            Log signal-to-noise ratio, shape (batch,).
        """
        log_a = self._log_alpha(t)
        a_sq = jnp.exp(2 * log_a)
        return 2 * log_a - jnp.log(1.0 - a_sq)

    def logsnr_inverse(self, lam: Scalar) -> Time:
        """Invert logsnr via the quadratic formula.

        From SNR = α²/(1-α²) = exp(λ), we have α² = sigmoid(λ), giving:
            β_min t + (β_max - β_min) t²/2 = softplus(-λ)

        This quadratic in t is solved analytically.

        Args:
            lam: Target log-SNR λ, shape (batch,).

        Returns:
            Time t, shape (batch,).
        """
        a_coeff = (self.beta_max - self.beta_min) / 2  # quadratic coefficient
        b_coeff = self.beta_min  # linear coefficient
        c_coeff = -jnp.logaddexp(0.0, -lam)  # -softplus(-λ)
        discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
        return (-b_coeff + jnp.sqrt(discriminant)) / (2 * a_coeff)


class VE(GaussianDiffusionProcess):
    """Variance-exploding diffusion with geometric noise schedule.

    α(t) = 1
    σ(t) = σ_min · (σ_max / σ_min)^t

    The noise level grows geometrically from σ_min to σ_max.
    At t=0, the data is approximately clean (σ_min ≈ 0).
    At t=1, the terminal distribution is approximately N(0, σ_max² I).

    Note: α(t) = 1 for all t, so "v_t" predictions carry the same information
    as "eps" (v(t) = σ'(t) ε). All conversions remain valid.

    Attributes:
        data_shape: Shape of a single data sample (excluding batch dim).
        sigma_min: Minimum noise level (at t=0).
        sigma_max: Maximum noise level (at t=1).
    """

    def __init__(
        self,
        data_shape: tuple[int, ...],
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
    ):
        """Initialize VE diffusion.

        Args:
            data_shape: Shape of a single data sample (excluding batch dim).
            sigma_min: Minimum noise level (at t=0). Should be small but > 0.
            sigma_max: Maximum noise level (at t=1).
        """
        super().__init__(data_shape)
        self.sigma_min: float = sigma_min
        self.sigma_max: float = sigma_max

    def alpha(self, t: Time) -> Scalar:
        """α(t) = 1.

        Args:
            t: Time, shape (batch,).

        Returns:
            Constant 1, shape (batch,).
        """
        return jnp.ones_like(t)

    def alpha_prime(self, t: Time) -> Scalar:
        """α'(t) = 0.

        Args:
            t: Time, shape (batch,).

        Returns:
            Constant 0, shape (batch,).
        """
        return jnp.zeros_like(t)

    def sigma(self, t: Time) -> Scalar:
        """σ(t) = σ_min · (σ_max / σ_min)^t.

        Args:
            t: Time, shape (batch,).

        Returns:
            Noise level σ(t), shape (batch,).
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sigma_prime(self, t: Time) -> Scalar:
        """σ'(t) = σ(t) · log(σ_max / σ_min).

        Args:
            t: Time, shape (batch,).

        Returns:
            Derivative σ'(t), shape (batch,).
        """
        log_ratio = jnp.log(self.sigma_max / self.sigma_min)
        return self.sigma(t) * log_ratio

    def logsnr(self, t: Time) -> Scalar:
        """log SNR(t) = -2 (log σ_min + t · log(σ_max / σ_min)).

        Args:
            t: Time, shape (batch,).

        Returns:
            Log signal-to-noise ratio, shape (batch,).
        """
        log_ratio = jnp.log(self.sigma_max / self.sigma_min)
        return -2.0 * (jnp.log(self.sigma_min) + t * log_ratio)

    def logsnr_inverse(self, lam: Scalar) -> Time:
        """t = -(λ/2 + log σ_min) / log(σ_max / σ_min).

        Args:
            lam: Target log-SNR λ, shape (batch,).

        Returns:
            Time t, shape (batch,).
        """
        log_ratio = jnp.log(self.sigma_max / self.sigma_min)
        return -(lam / 2.0 + jnp.log(self.sigma_min)) / log_ratio


class EDM(GaussianDiffusionProcess):
    """EDM (Karras et al. 2022) diffusion schedule.

    Uses the EDM power-law noise schedule with VE-style forward process:

    α(t)  = 1
    σ(t)  = ((1 - t) σ_min^(1/ρ) + t σ_max^(1/ρ))^ρ

    The power-law interpolation (controlled by ρ) concentrates more of
    the schedule's dynamic range at higher noise levels, matching the
    empirically effective training noise distribution from Karras et al.

    Note: σ(0) = σ_min > 0, so the data at t=0 is slightly noisy.
    For generation, σ_min should be small enough to be negligible.

    Attributes:
        data_shape: Shape of a single data sample (excluding batch dim).
        sigma_min: Minimum noise level (at t=0).
        sigma_max: Maximum noise level (at t=1).
        rho: Power-law exponent controlling schedule curvature.
    """

    def __init__(
        self,
        data_shape: tuple[int, ...],
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        """Initialize EDM diffusion.

        Args:
            data_shape: Shape of a single data sample (excluding batch dim).
            sigma_min: Minimum noise level (at t=0).
            sigma_max: Maximum noise level (at t=1).
            rho: Power-law exponent (default 7.0 per Karras et al.).
        """
        super().__init__(data_shape)
        self.sigma_min: float = sigma_min
        self.sigma_max: float = sigma_max
        self.rho: float = rho

    def _u(self, t: Time) -> Scalar:
        """Intermediate value u(t) = (1 - t) σ_min^(1/ρ) + t σ_max^(1/ρ).

        Args:
            t: Time, shape (batch,).

        Returns:
            u(t), shape (batch,).
        """
        s_min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        s_max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        return (1 - t) * s_min_inv_rho + t * s_max_inv_rho

    def alpha(self, t: Time) -> Scalar:
        """α(t) = 1.

        Args:
            t: Time, shape (batch,).

        Returns:
            Constant 1, shape (batch,).
        """
        return jnp.ones_like(t)

    def alpha_prime(self, t: Time) -> Scalar:
        """α'(t) = 0.

        Args:
            t: Time, shape (batch,).

        Returns:
            Constant 0, shape (batch,).
        """
        return jnp.zeros_like(t)

    def sigma(self, t: Time) -> Scalar:
        """σ(t) = u(t)^ρ.

        Args:
            t: Time, shape (batch,).

        Returns:
            Noise level σ(t), shape (batch,).
        """
        return self._u(t) ** self.rho

    def sigma_prime(self, t: Time) -> Scalar:
        """σ'(t) = ρ · u(t)^(ρ-1) · (σ_max^(1/ρ) - σ_min^(1/ρ)).

        Args:
            t: Time, shape (batch,).

        Returns:
            Derivative σ'(t), shape (batch,).
        """
        s_min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        s_max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        u = self._u(t)
        return self.rho * u ** (self.rho - 1) * (s_max_inv_rho - s_min_inv_rho)

    def logsnr(self, t: Time) -> Scalar:
        """log SNR(t) = -2 ρ log u(t).

        Args:
            t: Time, shape (batch,).

        Returns:
            Log signal-to-noise ratio, shape (batch,).
        """
        return -2.0 * self.rho * jnp.log(self._u(t))

    def logsnr_inverse(self, lam: Scalar) -> Time:
        """Invert logsnr analytically.

        From σ = exp(-λ/2) and u = σ^(1/ρ) = exp(-λ/(2ρ)):
            t = (u - σ_min^(1/ρ)) / (σ_max^(1/ρ) - σ_min^(1/ρ))

        Args:
            lam: Target log-SNR λ, shape (batch,).

        Returns:
            Time t, shape (batch,).
        """
        s_min_inv_rho = self.sigma_min ** (1.0 / self.rho)
        s_max_inv_rho = self.sigma_max ** (1.0 / self.rho)
        u = jnp.exp(-lam / (2.0 * self.rho))
        return (u - s_min_inv_rho) / (s_max_inv_rho - s_min_inv_rho)


class FlowMatching(GaussianDiffusionProcess):
    """Linear interpolation with Gaussian source (optimal transport / rectified flow).

    α(t) = 1 - t
    σ(t) = t

    The simplest interpolation schedule, corresponding to optimal transport /
    rectified flow with a Gaussian prior.

    Boundaries: α(0) = 1, σ(0) = 0, α(1) = 0, σ(1) = 1.

    Attributes:
        data_shape: Shape of a single data sample (excluding batch dim).
    """

    def alpha(self, t: Time) -> Scalar:
        """α(t) = 1 - t.

        Args:
            t: Time, shape (batch,).

        Returns:
            Scaling factor α(t), shape (batch,).
        """
        return 1.0 - t

    def alpha_prime(self, t: Time) -> Scalar:
        """α'(t) = -1.

        Args:
            t: Time, shape (batch,).

        Returns:
            Constant -1, shape (batch,).
        """
        return -jnp.ones_like(t)

    def sigma(self, t: Time) -> Scalar:
        """σ(t) = t.

        Args:
            t: Time, shape (batch,).

        Returns:
            Noise level σ(t) = t, shape (batch,).
        """
        return t

    def sigma_prime(self, t: Time) -> Scalar:
        """σ'(t) = 1.

        Args:
            t: Time, shape (batch,).

        Returns:
            Constant 1, shape (batch,).
        """
        return jnp.ones_like(t)

    def logsnr(self, t: Time) -> Scalar:
        """log SNR(t) = 2 (log(1 - t) - log t).

        Args:
            t: Time, shape (batch,).

        Returns:
            Log signal-to-noise ratio, shape (batch,).
        """
        return 2.0 * (jnp.log(1.0 - t) - jnp.log(t))

    def logsnr_inverse(self, lam: Scalar) -> Time:
        """t = sigmoid(-λ/2) = 1 / (1 + exp(λ/2)).

        Args:
            lam: Target log-SNR λ, shape (batch,).

        Returns:
            Time t, shape (batch,).
        """
        return jax.nn.sigmoid(-lam / 2.0)
