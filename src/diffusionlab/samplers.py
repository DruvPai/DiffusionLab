from typing import Callable, Tuple

from dataclasses import dataclass, field
import jax
from jax import Array, numpy as jnp

from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import (
    VectorFieldType,
    convert_vector_field_type,
)


@dataclass
class Sampler:
    """
    Base class for sampling from diffusion models using various vector field types.

    A Sampler combines a diffusion process, a vector field prediction function, and a scheduler
    to generate samples from a trained diffusion model using the reverse process (denoising/sampling).

    The sampler supports different vector field types (``VectorFieldType.SCORE``, ``VectorFieldType.X0``, ``VectorFieldType.EPS``, ``VectorFieldType.V``) and can perform both stochastic and deterministic sampling based on the subclass implementation and the `use_stochastic_sampler`` flag.

    Attributes:
        diffusion_process (``DiffusionProcess``): The diffusion process defining the forward dynamics.
        vector_field (``Callable[[Array[*data_dims], Array[]], Array[*data_dims]]``): The function predicting the vector field.
            Takes the current state ``x_t`` and time ``t`` as input.
        vector_field_type (``VectorFieldType``): The type of the vector field predicted by ``vector_field``.
        use_stochastic_sampler (``bool``): Whether to use a stochastic or deterministic reverse process.
        sample_step (``Callable[[int, Array, Array, Array], Array]``): The specific function used to perform one sampling step.
            Takes step index ``idx``, current state ``x_t``, noise array ``zs``, and time schedule ``ts`` as input.
            Set during initialization based on the sampler type and ``use_stochastic_sampler``.
    """

    diffusion_process: DiffusionProcess
    vector_field: Callable[[Array, Array], Array]
    vector_field_type: VectorFieldType
    use_stochastic_sampler: bool
    sample_step: Callable[[int, Array, Array, Array], Array] = field(init=False)

    def __post_init__(self):
        self.sample_step = self.get_sample_step_function()

    def sample(self, x_init: Array, zs: Array, ts: Array) -> Array:
        """
        Sample from the model using the reverse diffusion process.

        This method generates a final sample by iteratively applying the ``sample_step`` function,
        starting from an initial state ``x_init`` and using the provided noise ``zs`` and time schedule ``ts``.

        Args:
            x_init (``Array[*data_dims]``): The initial noisy tensor from which to initialize sampling (typically sampled from the prior distribution at ``ts[0]``).
            zs (``Array[num_steps, *data_dims]``): The noise tensors used at each step for stochastic sampling. Unused for deterministic samplers.
            ts (``Array[num_steps+1]``): The time schedule for sampling. A sorted decreasing array of times from ``t_max`` to ``t_min``.

        Returns:
            ``Array[*data_dims]``: The generated sample at the final time ``ts[-1]``.
        """

        def scan_fn(x, idx):
            next_x = self.sample_step(idx, x, zs, ts)
            return next_x, None

        final_x, _ = jax.lax.scan(scan_fn, x_init, jnp.arange(zs.shape[0]))

        return final_x

    def sample_trajectory(self, x_init: Array, zs: Array, ts: Array) -> Array:
        """
        Sample a trajectory from the model using the reverse diffusion process.

        This method generates the entire trajectory of intermediate samples by iteratively
        applying the ``sample_step`` function.

        Args:
            x_init (``Array[*data_dims]``): The initial noisy tensor from which to start sampling (at time ``ts[0]``).
            zs (``Array[num_steps, *data_dims]``): The noise tensors used at each step for stochastic sampling. Unused for deterministic samplers.
            ts (``Array[num_steps+1]``): The time schedule for sampling. A sorted decreasing array of times from ``t_max`` to ``t_min``.

        Returns:
            ``Array[num_steps+1, *data_dims]``: The complete generated trajectory including the initial state ``x_init``.
        """

        def scan_fn(x, idx):
            next_x = self.sample_step(idx, x, zs, ts)
            return next_x, next_x

        _, xs = jax.lax.scan(scan_fn, x_init, jnp.arange(zs.shape[0]))

        xs = jnp.concatenate([x_init[None, ...], xs], axis=0)
        return xs

    def get_sample_step_function(self) -> Callable[[int, Array, Array, Array], Array]:
        """
        Abstract method to get the appropriate sampling step function.

        Subclasses must implement this method to return the specific function used
        for performing one step of the reverse process, based on the sampler's
        implementation details (e.g., integrator type) and the ``use_stochastic_sampler`` flag.

        Returns:
            ``Callable[[int, Array, Array, Array], Array]``: The sampling step function, which has signature:

            ``(idx: int, x_t: Array[*data_dims], zs: Array[num_steps, *data_dims], ts: Array[num_steps+1]) -> Array[*data_dims]``
        """
        raise NotImplementedError


@dataclass
class EulerMaruyamaSampler(Sampler):
    """
    Class for sampling from diffusion models using the first-order Euler-Maruyama integrator
    for the reverse process SDE/ODE.

    This sampler implements the step function based on the Euler-Maruyama discretization
    of the reverse SDE (if ``use_stochastic_sampler`` is True) or the corresponding
    probability flow ODE (if ``use_stochastic_sampler`` is False). It supports all
    vector field types (``VectorFieldType.SCORE``, ``VectorFieldType.X0``, ``VectorFieldType.EPS``, ``VectorFieldType.V``).


    Attributes:
        diffusion_process (``DiffusionProcess``): The diffusion process defining the forward dynamics.
        vector_field (``Callable[[Array[*data_dims], Array[]], Array[*data_dims]]``): The function predicting the vector field.
            Takes the current state ``x_t`` and time ``t`` as input.
        vector_field_type (``VectorFieldType``): The type of the vector field predicted by ``vector_field``.
        use_stochastic_sampler (``bool``): Whether to use a stochastic or deterministic reverse process.
        sample_step (``Callable[[int, Array, Array, Array], Array]``): The specific function used to perform one sampling step.
            Takes step index ``idx``, current state ``x_t``, noise array ``zs``, and time schedule ``ts`` as input.
            Set during initialization based on the sampler type and ``use_stochastic_sampler``.
    """

    def get_sample_step_function(self) -> Callable[[int, Array, Array, Array], Array]:
        """
        Get the appropriate Euler-Maruyama sampling step function based on the
        vector field type and stochasticity.

        Returns:
            Callable[[int, Array, Array, Array], Array]: The specific Euler-Maruyama step function to use.

                Signature: ``(idx: int, x_t: Array[*data_dims], zs: Array[num_steps, *data_dims], ts: Array[num_steps+1]) -> Array[*data_dims]``
        """
        match (self.vector_field_type, self.use_stochastic_sampler):
            case (VectorFieldType.SCORE, False):
                return self._sample_step_score_deterministic
            case (VectorFieldType.SCORE, True):
                return self._sample_step_score_stochastic
            case (VectorFieldType.X0, False):
                return self._sample_step_x0_deterministic
            case (VectorFieldType.X0, True):
                return self._sample_step_x0_stochastic
            case (VectorFieldType.EPS, False):
                return self._sample_step_eps_deterministic
            case (VectorFieldType.EPS, True):
                return self._sample_step_eps_stochastic
            case (VectorFieldType.V, False):
                return self._sample_step_v_deterministic
            case (VectorFieldType.V, True):
                return self._sample_step_v_stochastic
            case _:
                raise ValueError(
                    f"Unsupported vector field type: {self.vector_field_type} and stochasticity: {self.use_stochastic_sampler}"
                )

    def _get_step_quantities(
        self,
        idx: int,
        zs: Array,
        ts: Array,
    ) -> Tuple[
        Array, Array, Array, Array, Array, Array, Array, Array, Array, Array, Array
    ]:
        """
        Calculate common quantities used in Euler-Maruyama sampling steps based on the diffusion process.

        Args:
            idx (``int``): Current step index (corresponds to time ``ts[idx]``).
            zs (``Array[num_steps, *data_dims]``): Noise tensors for stochastic sampling. Only ``zs[idx]`` is used if needed.
            ts (``Array[num_steps+1]``): Time schedule for sampling. Used to get ``ts[idx]`` and ``ts[idx+1]``.

        Returns:
            ``Tuple[Array[], Array[], Array[], Array[*data_dims], Array[], Array[], Array[], Array[], Array[], Array[], Array[]]``: A tuple containing

                - t (``Array[]``): Current time ``ts[idx]``.
                - t1 (``Array[]``): Next time ``ts[idx+1]``.
                - dt (``Array[]``): Time difference ``(t1 - t)``, should be negative.
                - dwt (``Array[*data_dims]``): Scaled noise increment ``sqrt(-dt) * zs[idx]`` for the stochastic step.
                - alpha_t (``Array[]``): ``α`` at current time ``t``.
                - sigma_t (``Array[]``): ``σ`` at current time ``t``.
                - alpha_prime_t (``Array[]``): Derivative of ``α`` at current time ``t``.
                - sigma_prime_t (``Array[]``): Derivative of ``σ`` at current time ``t``.
                - alpha_ratio_t (``Array[]``): ``alpha_prime_t / alpha_t``.
                - sigma_ratio_t (``Array[]``): ``sigma_prime_t / sigma_t``.
                - diff_ratio_t (``Array[]``): ``sigma_ratio_t - alpha_ratio_t``.
        """
        t = ts[idx]
        t1 = ts[idx + 1]
        dt = t1 - t
        dw_t = zs[idx] * jnp.sqrt(-dt)  # dt is negative

        alpha_t = self.diffusion_process.alpha(t)
        sigma_t = self.diffusion_process.sigma(t)
        alpha_prime_t = self.diffusion_process.alpha_prime(t)
        sigma_prime_t = self.diffusion_process.sigma_prime(t)
        alpha_ratio_t = alpha_prime_t / alpha_t
        sigma_ratio_t = sigma_prime_t / sigma_t
        diff_ratio_t = sigma_ratio_t - alpha_ratio_t

        return (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        )

    def _sample_step_score_deterministic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one deterministic Euler step using the score vector field (i.e., ``VectorFieldType.SCORE``).
        Corresponds to the probability flow ODE associated with the score SDE.

        Args:
            idx (``int``): Current step index.
            x_t (``Array[*data_dims]``): Current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors (unused).
            ts (``Array[num_steps+1]``): Time schedule.

        Returns:
            ``Array[*data_dims]``: Next state tensor at time ``ts[idx+1]``.
        """
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(idx, zs, ts)
        score_x_t = self.vector_field(x_t, t)
        drift_t = alpha_ratio_t * x_t - (sigma_t**2) * diff_ratio_t * score_x_t
        x_t1 = x_t + drift_t * dt
        return x_t1

    def _sample_step_score_stochastic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one stochastic Euler-Maruyama step using the score vector field (i.e., ``VectorFieldType.SCORE``).
        Corresponds to discretizing the reverse SDE derived using the score field.

        Args:
            idx (``int``): Current step index.
            x_t (``Array[*data_dims]``): Current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors. Uses ``zs[idx]``.
            ts (``Array[num_steps+1]``): Time schedule.

        Returns:
            ``Array[*data_dims]``: Next state tensor at time ``ts[idx+1]``.
        """
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(idx, zs, ts)
        score_x_t = self.vector_field(x_t, t)
        drift_t = alpha_ratio_t * x_t - 2 * (sigma_t**2) * diff_ratio_t * score_x_t
        diffusion_t = jnp.sqrt(2 * diff_ratio_t) * sigma_t
        x_t1 = x_t + drift_t * dt + diffusion_t * dw_t
        return x_t1

    def _sample_step_x0_deterministic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one deterministic Euler step using the ``x_0`` vector field (i.e., ``VectorFieldType.X0``).
        Corresponds to the probability flow ODE associated with the ``x_0`` SDE.

        Args:
            idx (``int``): Current step index.
            x_t (``Array[*data_dims]``): Current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors (unused).
            ts (``Array[num_steps+1]``): Time schedule.

        Returns:
            ``Array[*data_dims]``: Next state tensor at time ``ts[idx+1]``.
        """
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(idx, zs, ts)
        x0_x_t = self.vector_field(x_t, t)
        drift_t = sigma_ratio_t * x_t - alpha_t * diff_ratio_t * x0_x_t
        x_t1 = x_t + drift_t * dt
        return x_t1

    def _sample_step_x0_stochastic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one stochastic Euler-Maruyama step using the ``x_0`` vector field (i.e., ``VectorFieldType.X0``).
        Corresponds to discretizing the reverse SDE derived using the ``x_0`` field.

        Args:
            idx (``int``): Current step index.
            x_t (``Array[*data_dims]``): Current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors. Uses ``zs[idx]``.
            ts (``Array[num_steps+1]``): Time schedule.

        Returns:
            ``Array[*data_dims]``: Next state tensor at time ``ts[idx+1]``.
        """
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(idx, zs, ts)
        x0_x_t = self.vector_field(x_t, t)
        drift_t = (
            alpha_ratio_t + 2 * diff_ratio_t
        ) * x_t - 2 * alpha_t * diff_ratio_t * x0_x_t
        diffusion_t = jnp.sqrt(2 * diff_ratio_t) * sigma_t
        x_t1 = x_t + drift_t * dt + diffusion_t * dw_t
        return x_t1

    def _sample_step_eps_deterministic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one deterministic Euler step using the ε vector field (i.e., ``VectorFieldType.EPS``).
        Corresponds to the probability flow ODE associated with the ε SDE.

        Args:
            idx (``int``): Current step index.
            x_t (``Array[*data_dims]``): Current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors (unused).
            ts (``Array[num_steps+1]``): Time schedule.

        Returns:
            ``Array[*data_dims]``: Next state tensor at time ``ts[idx+1]``.
        """
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(idx, zs, ts)
        eps_x_t = self.vector_field(x_t, t)
        drift_t = alpha_ratio_t * x_t + sigma_t * diff_ratio_t * eps_x_t
        x_t1 = x_t + drift_t * dt
        return x_t1

    def _sample_step_eps_stochastic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one stochastic Euler-Maruyama step using the ε vector field (i.e., ``VectorFieldType.EPS``).
        Corresponds to discretizing the reverse SDE derived using the ε field.

        Args:
            idx (int): Current step index.
            x_t (``Array[*data_dims]``): Current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors. Uses ``zs[idx]``.
            ts (``Array[num_steps+1]``): Time schedule.

        Returns:
            ``Array[*data_dims]``: Next state tensor at time ``ts[idx+1]``.
        """
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(idx, zs, ts)
        eps_x_t = self.vector_field(x_t, t)
        drift_t = alpha_ratio_t * x_t + 2 * sigma_t * diff_ratio_t * eps_x_t
        diffusion_t = jnp.sqrt(2 * diff_ratio_t) * sigma_t
        x_t1 = x_t + drift_t * dt + diffusion_t * dw_t
        return x_t1

    def _sample_step_v_deterministic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one deterministic Euler step using the velocity vector field (i.e., ``VectorFieldType.V``).
        Corresponds to the probability flow ODE associated with the velocity SDE.

        Args:
            idx (``int``): Current step index.
            x_t (``Array[*data_dims]``): Current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors (unused).
            ts (``Array[num_steps+1]``): Time schedule.

        Returns:
            ``Array[*data_dims]``: Next state tensor at time ``ts[idx+1]``.
        """
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(idx, zs, ts)
        v_x_t = self.vector_field(x_t, t)
        drift_t = v_x_t
        x_t1 = x_t + drift_t * dt
        return x_t1

    def _sample_step_v_stochastic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one stochastic Euler-Maruyama step using the velocity vector field (i.e., ``VectorFieldType.V``).
        Corresponds to discretizing the reverse SDE derived using the velocity field.

        Args:
            idx (``int``): Current step index.
            x_t (``Array[*data_dims]``): Current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors. Uses ``zs[idx]``.
            ts (``Array[num_steps+1]``): Time schedule.

        Returns:
            ``Array[*data_dims]``: Next state tensor at time ``ts[idx+1]``.
        """
        (
            t,
            t1,
            dt,
            dw_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(idx, zs, ts)
        v_x_t = self.vector_field(x_t, t)
        drift_t = -alpha_ratio_t * x_t + 2 * v_x_t
        diffusion_t = jnp.sqrt(2 * diff_ratio_t) * sigma_t
        x_t1 = x_t + drift_t * dt + diffusion_t * dw_t
        return x_t1


@dataclass
class DDMSampler(Sampler):
    """
    Class for sampling from diffusion models using the Denoising Diffusion Probabilistic Models (DDPM)
    or Denoising Diffusion Implicit Models (DDIM) sampling strategy.

    This sampler first converts any given vector field type (``VectorFieldType.SCORE``, ``VectorFieldType.X0``, ``VectorFieldType.EPS``, ``VectorFieldType.V``) provided by ``vector_field`` into an equivalent x0 prediction using the ``convert_vector_field_type`` utility.
    Then, it applies the DDPM (if ``use_stochastic_sampler`` is ``True``) or DDIM (if ``use_stochastic_sampler`` is ``False``) update rule based on this x0 prediction.

    Attributes:
        diffusion_process (``DiffusionProcess``): The diffusion process defining the forward dynamics.
        vector_field (``Callable[[Array[*data_dims], Array[]], Array[*data_dims]]``): The function predicting the vector field.
        vector_field_type (``VectorFieldType``): The type of the vector field predicted by ``vector_field``.
        use_stochastic_sampler (``bool``): If ``True``, uses DDPM (stochastic); otherwise, uses DDIM (deterministic).
        sample_step (``Callable[[int, Array, Array, Array], Array]``): The DDPM or DDIM step function.
    """

    def get_sample_step_function(self) -> Callable[[int, Array, Array, Array], Array]:
        """
        Get the appropriate DDPM/DDIM sampling step function based on stochasticity.

        Returns:
            ``Callable[[int, Array, Array, Array], Array]``: The DDPM (stochastic) or DDIM (deterministic) step function, which has signature:

                ``(idx: int, x: Array[*data_dims], zs: Array[num_steps, *data_dims], ts: Array[num_steps+1]) -> Array[*data_dims]``
        """
        if self.use_stochastic_sampler:
            return self._sample_step_stochastic
        else:
            return self._sample_step_deterministic

    def _get_x0_prediction(self, x_t: Array, t: Array) -> Array:
        """
        Predict the initial state x_0 from the current noisy state x_t at time t.

        This uses the provided ``vector_field`` function and its ``vector_field_type``
        to compute the prediction, converting it to an X0 prediction if necessary.

        Args:
            x_t (``Array[*data_dims]``): The current state tensor.
            t (``Array[]``): The current time.

        Returns:
            ``Array[*data_dims]``: The predicted initial state x_0.
        """
        alpha_t = self.diffusion_process.alpha(t)
        sigma_t = self.diffusion_process.sigma(t)
        alpha_prime_t = self.diffusion_process.alpha_prime(t)
        sigma_prime_t = self.diffusion_process.sigma_prime(t)
        f_x_t = self.vector_field(x_t, t)
        x0_x_t = convert_vector_field_type(
            x_t,
            f_x_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            self.vector_field_type,
            VectorFieldType.X0,
        )
        return x0_x_t

    def _sample_step_deterministic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one deterministic DDIM sampling step.

        This involves predicting x0 from the current state ``(x_t, t)`` and then applying
        the DDIM update rule to get the state at the next timestep ``t1``.

        Args:
            idx (``int``): The current step index (corresponds to time ``ts[idx]``).
            x_t (``Array[*data_dims]``): The current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors (unused in DDIM).
            ts (``Array[num_steps+1]``): The time schedule for sampling.

        Returns:
            ``Array[*data_dims]``: The next state tensor at time ``ts[idx+1]`` after applying the DDIM update.
        """
        t = ts[idx]
        x0_x_t = self._get_x0_prediction(x_t, t)

        t1 = ts[idx + 1]
        alpha_t = self.diffusion_process.alpha(t)
        sigma_t = self.diffusion_process.sigma(t)
        alpha_t1 = self.diffusion_process.alpha(t1)
        sigma_t1 = self.diffusion_process.sigma(t1)

        r01 = sigma_t1 / sigma_t
        r11 = (alpha_t / alpha_t1) * r01

        mean = r01 * x_t + alpha_t1 * (1 - r11) * x0_x_t
        x_t1 = mean
        return x_t1

    def _sample_step_stochastic(
        self, idx: int, x_t: Array, zs: Array, ts: Array
    ) -> Array:
        """
        Perform one stochastic DDPM sampling step.

        This involves predicting x0 from the current state (x, t), and then applying
        the DDPM update rule, which corresponds to sampling from the conditional
        distribution p(x_{t-1}|x_t, x_0), adding noise scaled by sigma_t.

        Args:
            idx (``int``): The current step index (corresponds to time ``ts[idx]``).
            x_t (``Array[*data_dims]``): The current state tensor at time ``ts[idx]``.
            zs (``Array[num_steps, *data_dims]``): Noise tensors. Uses ``zs[idx]``.
            ts (``Array[num_steps+1]``): The time schedule for sampling.

        Returns:
            ``Array[*data_dims]``: The next state tensor at time ``ts[idx+1]`` after applying the DDPM update.
        """
        t = ts[idx]
        x0_x_t = self._get_x0_prediction(x_t, t)
        z_t = zs[idx]

        t1 = ts[idx + 1]
        alpha_t = self.diffusion_process.alpha(t)
        sigma_t = self.diffusion_process.sigma(t)
        alpha_t1 = self.diffusion_process.alpha(t1)
        sigma_t1 = self.diffusion_process.sigma(t1)

        r11 = (alpha_t / alpha_t1) * (sigma_t1 / sigma_t)
        r12 = r11 * (sigma_t1 / sigma_t)
        r22 = (alpha_t / alpha_t1) * r12

        mean = r12 * x_t + alpha_t1 * (1 - r22) * x0_x_t
        std = sigma_t1 * (1 - (r11**2)) ** (1 / 2)
        x_t1 = mean + std * z_t
        return x_t1
