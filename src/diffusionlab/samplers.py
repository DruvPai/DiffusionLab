from typing import Any, Callable, Dict, Tuple

import torch

from diffusionlab.utils import pad_shape_back, scalar_derivative
from diffusionlab.vector_fields import VectorField, VectorFieldType


class Sampler:
    """
    An encapsulation of a diffusion sampler, which has three conceptual components:
    - A schedule, which is a list of times t_1, ..., t_L.
    - A way to transform data according to the forward process at times given by the schedule, for the sake of learning the score or equivalent vector fields.
    - A way to transform data according to the reverse process at times given by the schedule, for the sake of sampling from the model.

    This class currently only implements a linear diffusion:

    x_t = alpha_t * x_0 + sigma_t * eps

    where eps ~ N(0, I), and alpha_0 = 1, sigma_0 = 0, alpha_t non-increasing, sigma_t non-decreasing. Nonetheless, a more general diffusion sampler can use essentially the same API (just substitute alpha and sigma with the drift and diffusion).
    """

    def __init__(
        self,
        is_stochastic: bool,
        alpha: Callable[[torch.Tensor], torch.Tensor],
        sigma: Callable[[torch.Tensor], torch.Tensor],
        schedule_params: Dict[str, Any],
    ):
        """
        Initializes a diffusion sampler.

        Arguments:
            is_stochastic: Whether the sampler's reverse process is stochastic.
            alpha: The scale parameter, presented as a function of time.
            sigma: The noise level parameter, presented as a function of time.
            schedule_params: A dictionary of parameters for the schedule.
        """
        self.is_stochastic: bool = is_stochastic
        self.alpha: Callable[[torch.Tensor], torch.Tensor] = alpha
        self.sigma: Callable[[torch.Tensor], torch.Tensor] = sigma
        self.alpha_prime: Callable[[torch.Tensor], torch.Tensor] = scalar_derivative(
            alpha
        )
        self.sigma_prime: Callable[[torch.Tensor], torch.Tensor] = scalar_derivative(
            sigma
        )
        self.schedule: torch.Tensor = self.get_schedule(schedule_params)

    def get_schedule(self, schedule_params: Dict[str, Any]) -> torch.Tensor:
        """
        Computes the schedule, which is a list of times t_1, ..., t_L.

        Arguments:
            schedule_params: A dictionary of parameters for the schedule.
                t_min: The minimum time.
                t_max: The maximum time.
                L: The number of steps in the schedule.

        Returns:
            A tensor of shape (L,) representing the schedule.
        """
        t_min = schedule_params["t_min"]
        t_max = schedule_params["t_max"]
        L = schedule_params["L"]
        return torch.linspace(t_min, t_max, L).flip(0)

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Adds noise to the data according to the forward process, i.e.,

        x_t = alpha_t * x_0 + sigma_t * eps

        where eps ~ N(0, I).

        Arguments:
            x: The data to add noise to, of shape (N, *D).
            t: The time at which to add noise, of shape (N,).
            eps: The noise to add, of shape (N, *D).

        Returns:
            The noisy data, of shape (N, *D).
        """
        alpha_x = pad_shape_back(self.alpha(t), x.shape) * x
        sigma_eps = pad_shape_back(self.sigma(t), x.shape) * eps
        return alpha_x + sigma_eps

    def sample(
        self, vector_field: VectorField, x0: torch.Tensor, zs: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples from the model using the reverse process.

        Arguments:
            vector_field: The vector field (either score, x0, eps, or v) to use for sampling.
            x0: The initialization (usually standard normal), of shape (N, *D).
            zs: The noise, of shape (L-1, N, *D).

        Returns:
            The sampled data, of shape (N, *D).
        """
        sample_step_function = self.get_sample_step_function(
            vector_field.vector_field_type
        )
        x = x0
        for i in range(self.schedule.shape[0] - 1):
            x = sample_step_function(vector_field, x, zs, i)
        return x

    def sample_trajectory(
        self, vector_field: VectorField, x0: torch.Tensor, zs: torch.Tensor
    ) -> torch.Tensor:
        """
        Samples a trajectory from the model using the reverse process.

        Arguments:
            vector_field: The vector field (either score, x0, eps, or v) to use for sampling.
            x0: The initialization (usually standard normal), of shape (N, *D).
            zs: The noise, of shape (L-1, N, *D).

        Returns:
            The sampled data, of shape (L, N, *D).
        """
        sample_step_function = self.get_sample_step_function(
            vector_field.vector_field_type
        )
        xs = [x0]
        x = x0
        for i in range(self.schedule.shape[0] - 1):
            x = sample_step_function(vector_field, x, zs, i)
            xs.append(x)
        return torch.stack(xs)

    def get_sample_step_function(
        self, vector_field_type: VectorFieldType
    ) -> Callable[[VectorField, torch.Tensor, torch.Tensor, int], torch.Tensor]:
        f = None
        if self.is_stochastic:
            if vector_field_type == VectorFieldType.SCORE:
                f = self.sample_step_stochastic_score
            elif vector_field_type == VectorFieldType.X0:
                f = self.sample_step_stochastic_x0
            elif vector_field_type == VectorFieldType.EPS:
                f = self.sample_step_stochastic_eps
            elif vector_field_type == VectorFieldType.V:
                f = self.sample_step_stochastic_v
        else:
            if vector_field_type == VectorFieldType.SCORE:
                f = self.sample_step_deterministic_score
            elif vector_field_type == VectorFieldType.X0:
                f = self.sample_step_deterministic_x0
            elif vector_field_type == VectorFieldType.EPS:
                f = self.sample_step_deterministic_eps
            elif vector_field_type == VectorFieldType.V:
                f = self.sample_step_deterministic_v
        return f

    def _get_step_quantities(
        self, zs: torch.Tensor, idx: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        x_shape = zs.shape[1:]
        t = pad_shape_back(self.schedule[idx], x_shape)
        t1 = pad_shape_back(self.schedule[idx + 1], x_shape)
        dt = t1 - t
        dwt = zs[idx] * torch.sqrt(-dt)

        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)
        alpha_prime_t = self.alpha_prime(t)
        sigma_prime_t = self.sigma_prime(t)
        alpha_ratio_t = alpha_prime_t / alpha_t
        sigma_ratio_t = sigma_prime_t / sigma_t
        diff_ratio_t = sigma_ratio_t - alpha_ratio_t
        return (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        )

    def _fix_t_shape(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: shape (N, *D): batch_dim, image_dims...
        # t: shape (1, 1, ..., 1): batch_dim, image_dims...
        t = t.view((1,)).expand(x.shape[0])
        return t

    def sample_step_stochastic_score(
        self, score: VectorField, x: torch.Tensor, zs: torch.Tensor, idx: int
    ) -> torch.Tensor:
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(zs, idx)
        drift_t = alpha_ratio_t * x - 2 * sigma_t**2 * diff_ratio_t * score(
            x, self._fix_t_shape(x, t)
        )
        diffusion_t = torch.sqrt(2 * diff_ratio_t) * sigma_t
        return x + drift_t * dt + diffusion_t * dwt

    def sample_step_deterministic_score(
        self, score: VectorField, x: torch.Tensor, zs: torch.Tensor, idx: int
    ) -> torch.Tensor:
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(zs, idx)
        drift_t = alpha_ratio_t * x - (sigma_t**2) * diff_ratio_t * score(
            x, self._fix_t_shape(x, t)
        )
        return x + drift_t * dt

    def sample_step_stochastic_x0(
        self, x0: VectorField, x: torch.Tensor, zs: torch.Tensor, idx: int
    ) -> torch.Tensor:
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(zs, idx)
        drift_t = (
            alpha_ratio_t + 2 * diff_ratio_t
        ) * x - 2 * alpha_t * diff_ratio_t * x0(x, self._fix_t_shape(x, t))
        diffusion_t = torch.sqrt(2 * diff_ratio_t) * sigma_t
        return x + drift_t * dt + diffusion_t * dwt

    def sample_step_deterministic_x0(
        self, x0: VectorField, x: torch.Tensor, zs: torch.Tensor, idx: int
    ) -> torch.Tensor:
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(zs, idx)
        drift_t = sigma_ratio_t * x - alpha_t * diff_ratio_t * x0(
            x, self._fix_t_shape(x, t)
        )
        return x + drift_t * dt

    def sample_step_stochastic_eps(
        self, eps: VectorField, x: torch.Tensor, zs: torch.Tensor, idx: int
    ) -> torch.Tensor:
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(zs, idx)
        drift_t = alpha_ratio_t * x + 2 * sigma_t * diff_ratio_t * eps(
            x, self._fix_t_shape(x, t)
        )
        diffusion_t = torch.sqrt(2 * diff_ratio_t) * sigma_t
        return x + drift_t * dt + diffusion_t * dwt

    def sample_step_deterministic_eps(
        self, eps: VectorField, x: torch.Tensor, zs: torch.Tensor, idx: int
    ) -> torch.Tensor:
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(zs, idx)
        drift_t = alpha_ratio_t * x + sigma_t * diff_ratio_t * eps(
            x, self._fix_t_shape(x, t)
        )
        return x + drift_t * dt

    def sample_step_stochastic_v(
        self, v: VectorField, x: torch.Tensor, zs: torch.Tensor, idx: int
    ) -> torch.Tensor:
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(zs, idx)
        drift_t = -alpha_ratio_t * x + v(x, self._fix_t_shape(x, t))
        diffusion_t = torch.sqrt(2 * diff_ratio_t) * sigma_t
        return x + drift_t * dt + diffusion_t * dwt

    def sample_step_deterministic_v(
        self, v: VectorField, x: torch.Tensor, zs: torch.Tensor, idx: int
    ) -> torch.Tensor:
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = self._get_step_quantities(zs, idx)
        drift_t = v(x, self._fix_t_shape(x, t))
        return x + drift_t * dt


class VESampler(Sampler):
    """
    A sampler corresponding to the variance-exploding (VE) diffusion process:

    x_t = x_0 + t * eps

    where eps ~ N(0, I).
    """

    def __init__(self, is_stochastic: bool, t_min: float, t_max: float, L: int):
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        schedule_params = {
            "t_min": t_min,
            "t_max": t_max,
            "L": L,
        }
        super().__init__(is_stochastic, alpha, sigma, schedule_params)


class VPSampler(Sampler):
    """
    A sampler corresponding to the variance-preserving (VP) diffusion process:

    x_t = sqrt(1 - t^2) * x_0 + t * eps

    where eps ~ N(0, I).
    """

    def __init__(self, is_stochastic: bool, t_min: float, t_max: float, L: int):
        assert 0 <= t_min <= t_max <= 1, "t_min and t_max must be in the range [0, 1]"
        alpha = lambda t: torch.sqrt(1 - t**2)
        sigma = lambda t: t
        schedule_params = {
            "t_min": t_min,
            "t_max": t_max,
            "L": L,
        }
        super().__init__(is_stochastic, alpha, sigma, schedule_params)


class FMSampler(Sampler):
    """
    A sampler corresponding to the flow-matching (FM) process:

    x_t = (1 - t) * x_0 + t * eps

    where eps ~ N(0, I).
    """

    def __init__(self, is_stochastic: bool, t_min: float, t_max: float, L: int):
        assert 0 <= t_min <= t_max <= 1, "t_min and t_max must be in the range [0, 1]"
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        schedule_params = {
            "t_min": t_min,
            "t_max": t_max,
            "L": L,
        }
        super().__init__(is_stochastic, alpha, sigma, schedule_params)
