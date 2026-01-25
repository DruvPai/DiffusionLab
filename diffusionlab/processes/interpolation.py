from abc import abstractmethod

import jax
from jax import numpy as jnp

from diffusionlab.processes.base import CorruptionProcess
from diffusionlab.typing import (
    AuxInfo,
    Coefficients,
    ContinuousData,
    Prediction,
    PRNGKey,
    Scalar,
    Time,
)
from diffusionlab.utils.tree_ops import bcast_right


class InterpolationProcess(CorruptionProcess):
    """
    Interpolation process between two distributions:

    x(t) = α(t) * x_0 + σ(t) * z, t in [0, 1],

    where z (the endpoint distribution) need not be Gaussian.

    Types of predictions one can make in this model (given x(t) and t):
    - x_0: the original data
    - z: the distribution to transport to law(x_0)
    - v(t): the velocity field at time t, aka x'(t) = α'(t) * x_0 + σ'(t) * z

    Given x(t) and s, t, such that s < t, one can also compute:
    - x(s): the estimated noised data at time s
    - v(s, t): the average velocity field between time s and t, aka (x(t) - x(s)) / (t - s)
    """

    @abstractmethod
    def alpha(self, t: Time) -> Scalar:
        """
        Get the scaling factor α(t) at time t.

        Args:
            t: The time at which to get the scaling factor.

        Returns:
            The scaling factor α(t).
        """
        pass

    @abstractmethod
    def alpha_prime(self, t: Time) -> Scalar:
        """
        Get the scaling factor derivative α'(t) at time t.

        Args:
            t: The time at which to get the scaling factor derivative.

        Returns:
            The scaling factor derivative α'(t).
        """
        pass

    @abstractmethod
    def sigma(self, t: Time) -> Scalar:
        """
        Get the noise standard deviation σ(t) at time t.

        Args:
            t: The time at which to get the noise standard deviation.

        Returns:
            The noise standard deviation σ(t).
        """
        pass

    @abstractmethod
    def sigma_prime(self, t: Time) -> Scalar:
        """
        Get the noise standard deviation derivative σ'(t) at time t.

        Args:
            t: The time at which to get the noise standard deviation derivative.

        Returns:
            The noise standard deviation derivative σ'(t).
        """
        pass

    def logsnr(self, t: Time) -> Scalar:
        """
        Get the log of the signal-to-noise ratio at time t.

        Args:
            t: The time at which to get the log of the signal-to-noise ratio.

        Returns:
            The log of the signal-to-noise ratio log SNR(t) = 2 * (log α(t) - log σ(t)).
        """
        return 2.0 * (jnp.log(self.alpha(t)) - jnp.log(self.sigma(t)))

    @abstractmethod
    def logsnr_inverse(self, lam: Scalar) -> Time:
        """
        Get the time t at which the signal-to-noise ratio is lam.

        Args:
            lam: The target signal-to-noise ratio.

        Returns:
            The time t at which the signal-to-noise ratio is lam.
        """
        pass

    def forward(
        self, key: PRNGKey, x: ContinuousData, t: Time
    ) -> tuple[ContinuousData, AuxInfo]:
        """
        Corrupt the data x at time t using the interpolation x(t) = α(t) * x_0 + σ(t) * z, and return the corrupted data and auxiliary information.

        Args:
            key: A PRNG key.
            x: The original data x_0.
            t: The time t at which to corrupt the data.

        Returns:
            A tuple of the corrupted data x(t) and auxiliary information.
        """
        batch_size = x.shape[0]
        z = self._sample_from_source(key, batch_size)
        return self._forward_with_random(x, z, t)

    def _forward_with_random(
        self, x: ContinuousData, r: ContinuousData, t: Time
    ) -> tuple[ContinuousData, AuxInfo]:
        z = r
        coefficients = self.get_coefficients(t)
        alpha = coefficients["alpha"]  # (batch,)
        sigma = coefficients["sigma"]  # (batch,)
        x_t = bcast_right(alpha, x.ndim) * x + bcast_right(sigma, x.ndim) * z
        return x_t, {"x": x, "t": t, "z": z} | coefficients

    def forward_multiple(
        self, key: PRNGKey, x: ContinuousData, ts: list[Time]
    ) -> list[tuple[ContinuousData, AuxInfo]]:
        """
        Corrupt the data x at each time t, and return the corrupted data and auxiliary information, for each time.

        Args:
            key: A PRNG key.
            x: The original data x_0.
            ts: The times t at which to corrupt the data.

        Returns:
            A list of tuples of the corrupted data x(t) and auxiliary information, for each time.
        """
        batch_size = x.shape[0]
        z = self._sample_from_source(key, batch_size)
        stacked_ts = jnp.stack(ts)  # (num_times, batch)
        stacked_x_t, stacked_aux = jax.vmap(
            lambda t: self._forward_with_random(x, z, t)
        )(stacked_ts)
        stacked = (stacked_x_t, stacked_aux)
        return [jax.tree_util.tree_map(lambda v: v[i], stacked) for i in range(len(ts))]

    def sample_from_terminal(
        self, key: PRNGKey, batch_size: int
    ) -> tuple[ContinuousData, AuxInfo]:
        """
        Sample from (approximately) the terminal distribution x_1.

        Args:
            key: A PRNG key.
            batch_size: The number of samples to draw.

        Returns:
            A tuple of the sampled data x_1 and auxiliary information.
        """
        z = self._sample_from_source(key, batch_size)
        t = jnp.ones((batch_size,))
        coefficients = self.get_coefficients(t)
        sigma = coefficients["sigma"]
        x_1 = bcast_right(sigma, z.ndim) * z
        return x_1, {
            "t": t,
            "z": z,
        } | coefficients

    def convert_prediction(
        self,
        prediction: Prediction,
        x_t: ContinuousData,
        s: Time | None,
        t: Time,
        to_kind: str,
    ) -> Prediction:
        """
        Convert a model prediction to the requested kind.

        Given the model
        x(t) = α(t) * x_0 + σ(t) * z,
        the available types of predictions:
        - x_0: the original data
        - z: the distribution to transport to law(x_0)
        - v(t): the velocity field at time t, aka x'(t) = α'(t) * x_0 + σ'(t) * z

        If intermediate time s is provided, one can also compute:
        - x(s): the estimated noised data at time s
        - v(s, t): the average velocity field between time s and t, aka (x(t) - x(s)) / (t - s)

        Conversion algebra (shorthand: α = α(t), σ = σ(t), etc.):

        From x_0:
            z     = (x_t - α x_0) / σ
            v     = α' x_0 + σ' (x_t - α x_0) / σ
            x_s   = α_s x_0 + σ_s (x_t - α x_0) / σ
            v_st  = (x_t - x_s) / (t - s)

        From z:
            x_0   = (x_t - σ z) / α
            v     = α' (x_t - σ z) / α + σ' z
            x_s   = α_s (x_t - σ z) / α + σ_s z
            v_st  = (x_t - x_s) / (t - s)

        From v (solving v = α' x_0 + σ' z jointly with x_t = α x_0 + σ z):
            x_0   = (σ v - σ' x_t) / (σ α' - σ' α)
            z     = (α v - α' x_t) / (α σ' - α' σ)
            x_s   = α_s x_0 + σ_s z          (using x_0, z above)
            v_st  = (x_t - x_s) / (t - s)

        From x_s (solving x_s = α_s x_0 + σ_s z jointly with x_t = α x_0 + σ z):
            x_0   = (σ x_s - σ_s x_t) / (σ α_s - σ_s α)
            z     = (α x_s - α_s x_t) / (α σ_s - α_s σ)
            v     = α' x_0 + σ' z             (using x_0, z above)
            v_st  = (x_t - x_s) / (t - s)

        From v_st (first recover x_s = x_t - (t - s) v_st, then as from x_s):
            x_s   = x_t - (t - s) v_st
            x_0   = (σ x_s - σ_s x_t) / (σ α_s - σ_s α)
            z     = (α x_s - α_s x_t) / (α σ_s - α_s σ)
            v     = α' x_0 + σ' z             (using x_0, z above)

        Args:
            prediction: The prediction to convert.
            x_t: The corrupted data at time t.
            s: An optional time which, if provided and the prediction involves an intermediate time 0 < s < t, specifies the time s.
            t: The time t at which the prediction is evaluated.
            to_kind: The kind of prediction to convert the prediction to.

        Returns:
            A prediction of the requested kind.
        """
        ndim = x_t.ndim
        val = prediction.value

        coefficients = self.get_coefficients(t)
        alpha = bcast_right(coefficients["alpha"], ndim)  # (batch, 1, ...)
        sigma = bcast_right(coefficients["sigma"], ndim)  # (batch, 1, ...)
        alpha_prime = bcast_right(coefficients["alpha_prime"], ndim)  # (batch, 1, ...)
        sigma_prime = bcast_right(coefficients["sigma_prime"], ndim)  # (batch, 1, ...)

        alpha_s = jnp.ones_like(alpha)
        sigma_s = jnp.zeros_like(sigma)
        dt = jnp.zeros_like(alpha)
        if s is not None:
            coefficients_s = self.get_coefficients(s)
            alpha_s = bcast_right(coefficients_s["alpha"], ndim)  # (batch, 1, ...)
            sigma_s = bcast_right(coefficients_s["sigma"], ndim)  # (batch, 1, ...)
            dt = bcast_right(t - s, ndim)

        from_kind = prediction.kind
        match (from_kind, to_kind):
            case (
                ("x_0", "x_0")
                | ("z", "z")
                | ("v_t", "v_t")
                | ("x_s", "x_s")
                | ("v_st", "v_st")
            ):
                new_prediction = prediction

            # --- From x_0 ---
            case ("x_0", "z"):
                # z = (x_t - α(t) x_0) / σ(t)
                new_prediction = Prediction(
                    value=(x_t - alpha * val) / sigma,
                    kind="z",
                )
            case ("x_0", "v_t"):
                # v(t) = α'(t) x_0 + σ'(t) z
                #      = α'(t) x_0 + σ'(t) (x_t - α(t) x_0) / σ(t)
                new_prediction = Prediction(
                    value=(
                        alpha_prime * val + sigma_prime * (x_t - alpha * val) / sigma
                    ),
                    kind="v_t",
                )
            case ("x_0", "x_s"):
                assert s is not None
                # x(s) = α(s) x_0 + σ(s) (x_t - α(t) x_0) / σ(t)
                new_prediction = Prediction(
                    value=(alpha_s * val + sigma_s * (x_t - alpha * val) / sigma),
                    kind="x_s",
                )
            case ("x_0", "v_st"):
                assert s is not None
                assert dt is not None
                # v(s,t) = (x(t) - x(s)) / (t - s)
                x_s = alpha_s * val + sigma_s * (x_t - alpha * val) / sigma
                new_prediction = Prediction(
                    value=(x_t - x_s) / dt,
                    kind="v_st",
                )

            # --- From z ---
            case ("z", "x_0"):
                # x_0 = (x_t - σ(t) z) / α(t)
                new_prediction = Prediction(
                    value=(x_t - sigma * val) / alpha,
                    kind="x_0",
                )
            case ("z", "v_t"):
                # v(t) = α'(t) (x_t - σ(t) z) / α(t) + σ'(t) z
                new_prediction = Prediction(
                    value=(
                        alpha_prime * (x_t - sigma * val) / alpha + sigma_prime * val
                    ),
                    kind="v_t",
                )
            case ("z", "x_s"):
                assert s is not None
                # x(s) = α(s) (x_t - σ(t) z) / α(t) + σ(s) z
                new_prediction = Prediction(
                    value=(alpha_s * (x_t - sigma * val) / alpha + sigma_s * val),
                    kind="x_s",
                )
            case ("z", "v_st"):
                assert s is not None
                assert dt is not None
                # v(s,t) = (x(t) - x(s)) / (t - s)
                x_s = alpha_s * (x_t - sigma * val) / alpha + sigma_s * val
                new_prediction = Prediction(
                    value=(x_t - x_s) / dt,
                    kind="v_st",
                )

            # --- From v ---
            case ("v_t", "x_0"):
                # x_0 = (σ(t) v - σ'(t) x_t) / (σ(t) α'(t) - σ'(t) α(t))
                new_prediction = Prediction(
                    value=(
                        (sigma * val - sigma_prime * x_t)
                        / (sigma * alpha_prime - sigma_prime * alpha)
                    ),
                    kind="x_0",
                )
            case ("v_t", "z"):
                # z = (α(t) v - α'(t) x_t) / (α(t) σ'(t) - α'(t) σ(t))
                new_prediction = Prediction(
                    value=(
                        (alpha * val - alpha_prime * x_t)
                        / (alpha * sigma_prime - alpha_prime * sigma)
                    ),
                    kind="z",
                )
            case ("v_t", "x_s"):
                assert s is not None
                x_0 = (sigma * val - sigma_prime * x_t) / (
                    sigma * alpha_prime - sigma_prime * alpha
                )
                z = (x_t - alpha * x_0) / sigma
                new_prediction = Prediction(
                    value=alpha_s * x_0 + sigma_s * z,
                    kind="x_s",
                )
            case ("v_t", "v_st"):
                assert s is not None
                assert dt is not None
                x_0 = (sigma * val - sigma_prime * x_t) / (
                    sigma * alpha_prime - sigma_prime * alpha
                )
                z = (x_t - alpha * x_0) / sigma
                x_s = alpha_s * x_0 + sigma_s * z
                new_prediction = Prediction(
                    value=(x_t - x_s) / dt,
                    kind="v_st",
                )

            # --- From x_s ---
            case ("x_s", "x_0"):
                assert s is not None
                # x_0 = (σ(t) x(s) - σ(s) x(t)) / (σ(t) α(s) - σ(s) α(t))
                new_prediction = Prediction(
                    value=(
                        (sigma * val - sigma_s * x_t)
                        / (sigma * alpha_s - sigma_s * alpha)
                    ),
                    kind="x_0",
                )
            case ("x_s", "z"):
                assert s is not None
                # z = (α(t) x(s) - α(s) x(t)) / (α(t) σ(s) - α(s) σ(t))
                new_prediction = Prediction(
                    value=(
                        (alpha * val - alpha_s * x_t)
                        / (alpha * sigma_s - alpha_s * sigma)
                    ),
                    kind="z",
                )
            case ("x_s", "v_t"):
                assert s is not None
                x_0 = (sigma * val - sigma_s * x_t) / (
                    sigma * alpha_s - sigma_s * alpha
                )
                z = (x_t - alpha * x_0) / sigma
                new_prediction = Prediction(
                    value=alpha_prime * x_0 + sigma_prime * z,
                    kind="v_t",
                )
            case ("x_s", "v_st"):
                assert s is not None
                assert dt is not None
                # v(s,t) = (x(t) - x(s)) / (t - s)
                new_prediction = Prediction(
                    value=(x_t - val) / dt,
                    kind="v_st",
                )

            # --- From v_st ---
            case ("v_st", "x_0"):
                assert s is not None
                assert dt is not None
                # x(s) = x(t) - (t - s) v(s,t)
                x_s = x_t - dt * val
                new_prediction = Prediction(
                    value=(
                        (sigma * x_s - sigma_s * x_t)
                        / (sigma * alpha_s - sigma_s * alpha)
                    ),
                    kind="x_0",
                )
            case ("v_st", "z"):
                assert s is not None
                assert dt is not None
                x_s = x_t - dt * val
                new_prediction = Prediction(
                    value=(
                        (alpha * x_s - alpha_s * x_t)
                        / (alpha * sigma_s - alpha_s * sigma)
                    ),
                    kind="z",
                )
            case ("v_st", "v_t"):
                assert s is not None
                assert dt is not None
                x_s = x_t - dt * val
                x_0 = (sigma * x_s - sigma_s * x_t) / (
                    sigma * alpha_s - sigma_s * alpha
                )
                z = (x_t - alpha * x_0) / sigma
                new_prediction = Prediction(
                    value=alpha_prime * x_0 + sigma_prime * z,
                    kind="v_t",
                )
            case ("v_st", "x_s"):
                assert s is not None
                assert dt is not None
                # x(s) = x(t) - (t - s) v(s,t)
                new_prediction = Prediction(
                    value=x_t - dt * val,
                    kind="x_s",
                )

            case _:
                raise NotImplementedError(
                    f"Conversion from {from_kind} to {to_kind} is not implemented."
                )
        return new_prediction

    @abstractmethod
    def _sample_from_source(self, key: PRNGKey, batch_size: int) -> ContinuousData:
        """
        Sample from the source distribution z.

        Args:
            key: A PRNG key.
            batch_size: The number of samples to draw.

        Returns:
            The sampled data z.
        """
        pass

    def get_coefficients(self, t: Time) -> Coefficients:
        """
        Get the corruption coefficients for the given time t.

        Args:
            t: The time at which to get the corruption coefficients.

        Returns:
            The corruption coefficients for the given time; contains the following keys:
            - alpha: the scaling factor α(t)
            - alpha_prime: the scaling factor derivative α'(t)
            - sigma: the noise standard deviation σ(t)
            - sigma_prime: the noise standard deviation derivative σ'(t)
            - logsnr: the log of the signal-to-noise ratio log SNR(t) = 2 * (log α(t) - log σ(t))
        """
        alpha = self.alpha(t)
        alpha_prime = self.alpha_prime(t)
        sigma = self.sigma(t)
        sigma_prime = self.sigma_prime(t)
        logsnr = self.logsnr(t)
        return {
            "alpha": alpha,
            "alpha_prime": alpha_prime,
            "sigma": sigma,
            "sigma_prime": sigma_prime,
            "logsnr": logsnr,
        }
