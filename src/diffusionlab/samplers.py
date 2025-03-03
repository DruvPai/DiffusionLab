from typing import Callable, Tuple

import torch

from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.utils import pad_shape_back
from diffusionlab.vector_fields import VectorField, VectorFieldType


class Sampler:
    """
    Class for sampling from diffusion models using various vector field types.

    A Sampler combines a diffusion process and a scheduler to generate samples from
    a trained diffusion model. It handles both the forward process (adding noise) and
    the reverse process (denoising/sampling).

    The sampler supports different vector field types (SCORE, X0, EPS, V) and can perform
    both stochastic and deterministic sampling.

    Attributes:
        diffusion (DiffusionProcess): The diffusion process defining the forward and reverse dynamics
        is_stochastic (bool): Whether the reverse process is stochastic or deterministic
    """

    def __init__(
        self,
        diffusion_process: DiffusionProcess,
        is_stochastic: bool,
    ):
        """
        Initialize a sampler with a diffusion process and sampling strategy.

        Args:
            diffusion (DiffusionProcess): The diffusion process to use for sampling
            is_stochastic (bool): Whether the reverse process should be stochastic
        """
        self.diffusion_process: DiffusionProcess = diffusion_process
        self.is_stochastic: bool = is_stochastic

    def sample(
        self,
        vector_field: VectorField,
        x_t: torch.Tensor,
        zs: torch.Tensor,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from the model using the reverse diffusion process.

        This method generates a sample by iteratively applying the appropriate sampling step
        function based on the vector field type.

        Args:
            vector_field (VectorField): The vector field model to use for sampling
            x_t (torch.Tensor): The initial noisy tensor to start sampling from, of shape (N, *D)
                where N is the batch size and D represents the data dimensions
            zs (torch.Tensor): The noise tensors for stochastic sampling, of shape (L-1, N, *D)
                where L is the number of time steps
            ts (torch.Tensor): The time schedule for sampling, of shape (L,)
                where L is the number of time steps

        Returns:
            torch.Tensor: The generated sample, of shape (N, *D)
        """
        sample_step_function = self.get_sample_step_function(
            vector_field.vector_field_type
        )
        x = x_t
        for i in range(ts.shape[0] - 1):
            x = sample_step_function(vector_field, x, zs, i, ts)
        return x

    def sample_trajectory(
        self,
        vector_field: VectorField,
        x_t: torch.Tensor,
        zs: torch.Tensor,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample a trajectory from the model using the reverse diffusion process.

        This method is similar to sample() but returns the entire trajectory of
        intermediate samples rather than just the final sample.

        Args:
            vector_field (VectorField): The vector field model to use for sampling
            x_t (torch.Tensor): The initial noisy tensor to start sampling from, of shape (N, *D)
                where N is the batch size and D represents the data dimensions
            zs (torch.Tensor): The noise tensors for stochastic sampling, of shape (L-1, N, *D)
                where L is the number of time steps
            ts (torch.Tensor): The time schedule for sampling, of shape (L,)
                where L is the number of time steps

        Returns:
            torch.Tensor: The generated trajectory, of shape (L, N, *D)
                where L is the number of time steps
        """
        sample_step_function = self.get_sample_step_function(
            vector_field.vector_field_type
        )
        xs = [x_t]
        x = x_t
        for i in range(ts.shape[0] - 1):
            x = sample_step_function(vector_field, x, zs, i, ts)
            xs.append(x)
        return torch.stack(xs)

    def get_sample_step_function(
        self, vector_field_type: VectorFieldType
    ) -> Callable[
        [VectorField, torch.Tensor, torch.Tensor, int, torch.Tensor], torch.Tensor
    ]:
        """
        Get the appropriate sampling step function based on the vector field type.

        This method selects the correct sampling function based on the vector field type
        and whether sampling is stochastic or deterministic.

        Args:
            vector_field_type (VectorFieldType): The type of vector field being used
                                                (SCORE, X0, EPS, or V)

        Returns:
            Callable: A function that performs one step of the sampling process with signature:
                     (vector_field, x, zs, idx, ts) -> next_x
                     where:
                     - vector_field is the model
                     - x is the current state tensor of shape (N, *D)
                       where N is the batch size and D represents the data dimensions
                     - zs is the noise tensors of shape (L-1, N, *D)
                       where L is the number of time steps
                     - idx is the current step index
                     - ts is the time steps tensor of shape (L,)
                       where L is the number of time steps
                     - next_x is the next state tensor of shape (N, *D)
        """
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

    def _fix_t_shape(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Reshape the time tensor to be compatible with the batch dimension of x.

        Args:
            x (torch.Tensor): The data tensor of shape (N, *D)
                where N is the batch size and D represents the data dimensions
            t (torch.Tensor): The time tensor to reshape, of shape (1, 1, ..., 1)
                or any shape that can be broadcast to match the batch size

        Returns:
            torch.Tensor: The reshaped time tensor of shape (N,)
                where N is the batch size of x
        """
        t = t.view((1,)).expand(x.shape[0])
        return t

    def sample_step_stochastic_score(
        self,
        score: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a stochastic sampling step using the score vector field.

        This method implements one step of the stochastic reverse process using the score function.

        Args:
            score (VectorField): The score vector field model
            x (torch.Tensor): The current state tensor, of shape (N, *D)
                where N is the batch size and D represents the data dimensions
            zs (torch.Tensor): The noise tensors for stochastic sampling, of shape (L-1, N, *D)
                where L is the number of time steps
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)
                where L is the number of time steps

        Returns:
            torch.Tensor: The next state tensor, of shape (N, *D)
        """
        raise NotImplementedError

    def sample_step_deterministic_score(
        self,
        score: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of deterministic sampling using the score vector field.

        Args:
            score (VectorField): The score vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors (unused in deterministic sampling), of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
        raise NotImplementedError

    def sample_step_stochastic_x0(
        self,
        x0: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of stochastic sampling using the x0 vector field.

        Args:
            x0 (VectorField): The x0 vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors, of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
        raise NotImplementedError

    def sample_step_deterministic_x0(
        self,
        x0: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of deterministic sampling using the x0 vector field.

        Args:
            x0 (VectorField): The x0 vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors (unused in deterministic sampling), of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
        raise NotImplementedError

    def sample_step_stochastic_eps(
        self,
        eps: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of stochastic sampling using the eps vector field.

        Args:
            eps (VectorField): The eps vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors, of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
        raise NotImplementedError

    def sample_step_deterministic_eps(
        self,
        eps: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of deterministic sampling using the eps vector field.

        Args:
            eps (VectorField): The eps vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors (unused in deterministic sampling), of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
        raise NotImplementedError

    def sample_step_stochastic_v(
        self,
        v: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of stochastic sampling using the v vector field.

        Args:
            v (VectorField): The velocity vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors, of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
        raise NotImplementedError

    def sample_step_deterministic_v(
        self,
        v: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of deterministic sampling using the v vector field.

        Args:
            v (VectorField): The velocity vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors (unused in deterministic sampling), of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
        raise NotImplementedError


class EulerMaruyamaSampler(Sampler):
    def _get_step_quantities(
        self, zs: torch.Tensor, idx: int, ts: torch.Tensor
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
        """
        Calculate various quantities needed for a sampling step.

        This helper method computes time-dependent quantities used in the sampling
        step functions.

        Args:
            zs (torch.Tensor): The noise tensors for stochastic sampling, of shape (L-1, N, *D)
                where L is the number of time steps, N is the batch size, and D represents the data dimensions
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)
                where L is the number of time steps

        Returns:
            Tuple: A tuple containing various time-dependent quantities:
                  - t (torch.Tensor): Current time, of shape (1*), where 1* is a tuple with the same number of dimensions as (N, D*)
                  - t1 (torch.Tensor): Next time, of shape (1*)
                  - alpha_t (torch.Tensor): Alpha at current time, of shape (1*)
                  - sigma_t (torch.Tensor): Sigma at current time, of shape (1*)
                  - alpha_prime_t (torch.Tensor): Derivative of alpha at current time, of shape (1*)
                  - sigma_prime_t (torch.Tensor): Derivative of sigma at current time, of shape (1*)
                  - dt (torch.Tensor): Time difference, of shape (1*)
                  - dwt (torch.Tensor): Scaled noise, of shape (N, *D)
                  - alpha_ratio_t (torch.Tensor): alpha_prime_t / alpha_t, of shape (1*)
                  - sigma_ratio_t (torch.Tensor): sigma_prime_t / sigma_t, of shape (1*)
                  - diff_ratio_t (torch.Tensor): sigma_ratio_t - alpha_ratio_t, of shape (1*)
        """
        x_shape = zs.shape[1:]
        t = pad_shape_back(ts[idx], x_shape)
        t1 = pad_shape_back(ts[idx + 1], x_shape)
        dt = t1 - t
        dwt = zs[idx] * torch.sqrt(-dt)

        alpha_t = pad_shape_back(self.diffusion_process.alpha(ts[idx]), x_shape)
        sigma_t = pad_shape_back(self.diffusion_process.sigma(ts[idx]), x_shape)
        alpha_prime_t = pad_shape_back(
            self.diffusion_process.alpha_prime(ts[idx]), x_shape
        )
        sigma_prime_t = pad_shape_back(
            self.diffusion_process.sigma_prime(ts[idx]), x_shape
        )
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

    def sample_step_deterministic_score(
        self,
        score: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of deterministic sampling using the score vector field.

        Args:
            score (VectorField): The score vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors (unused in deterministic sampling), of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
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
        ) = self._get_step_quantities(zs, idx, ts)
        drift_t = alpha_ratio_t * x - (sigma_t**2) * diff_ratio_t * score(
            x, self._fix_t_shape(x, t)
        )
        return x + drift_t * dt

    def sample_step_stochastic_x0(
        self,
        x0: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of stochastic sampling using the x0 vector field.

        Args:
            x0 (VectorField): The x0 vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors, of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
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
        ) = self._get_step_quantities(zs, idx, ts)
        drift_t = (
            alpha_ratio_t + 2 * diff_ratio_t
        ) * x - 2 * alpha_t * diff_ratio_t * x0(x, self._fix_t_shape(x, t))
        diffusion_t = torch.sqrt(2 * diff_ratio_t) * sigma_t
        return x + drift_t * dt + diffusion_t * dwt

    def sample_step_deterministic_x0(
        self,
        x0: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of deterministic sampling using the x0 vector field.

        Args:
            x0 (VectorField): The x0 vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors (unused in deterministic sampling), of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
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
        ) = self._get_step_quantities(zs, idx, ts)
        drift_t = sigma_ratio_t * x - alpha_t * diff_ratio_t * x0(
            x, self._fix_t_shape(x, t)
        )
        return x + drift_t * dt

    def sample_step_stochastic_eps(
        self,
        eps: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of stochastic sampling using the eps vector field.

        Args:
            eps (VectorField): The eps vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors, of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
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
        ) = self._get_step_quantities(zs, idx, ts)
        drift_t = alpha_ratio_t * x + 2 * sigma_t * diff_ratio_t * eps(
            x, self._fix_t_shape(x, t)
        )
        diffusion_t = torch.sqrt(2 * diff_ratio_t) * sigma_t
        return x + drift_t * dt + diffusion_t * dwt

    def sample_step_deterministic_eps(
        self,
        eps: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of deterministic sampling using the eps vector field.

        Args:
            eps (VectorField): The eps vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors (unused in deterministic sampling), of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
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
        ) = self._get_step_quantities(zs, idx, ts)
        drift_t = alpha_ratio_t * x + sigma_t * diff_ratio_t * eps(
            x, self._fix_t_shape(x, t)
        )
        return x + drift_t * dt

    def sample_step_stochastic_v(
        self,
        v: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of stochastic sampling using the v vector field.

        Args:
            v (VectorField): The velocity vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors, of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
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
        ) = self._get_step_quantities(zs, idx, ts)
        drift_t = -alpha_ratio_t * x + v(x, self._fix_t_shape(x, t))
        diffusion_t = torch.sqrt(2 * diff_ratio_t) * sigma_t
        return x + drift_t * dt + diffusion_t * dwt

    def sample_step_deterministic_v(
        self,
        v: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform one step of deterministic sampling using the v vector field.

        Args:
            v (VectorField): The velocity vector field model
            x (torch.Tensor): The current state, of shape (N, *D)
            zs (torch.Tensor): The noise tensors (unused in deterministic sampling), of shape (L-1, N, *D)
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)

        Returns:
            torch.Tensor: The next state after one sampling step, of shape (N, *D)
        """
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
        ) = self._get_step_quantities(zs, idx, ts)
        drift_t = v(x, self._fix_t_shape(x, t))
        return x + drift_t * dt

    def sample_step_stochastic_score(
        self,
        score: VectorField,
        x: torch.Tensor,
        zs: torch.Tensor,
        idx: int,
        ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform a stochastic sampling step using the score vector field.

        This implements the stochastic reverse SDE for score-based models using the
        Euler-Maruyama discretization method.

        Args:
            score (VectorField): The score vector field model
            x (torch.Tensor): The current state tensor, of shape (N, *D)
                where N is the batch size and D represents the data dimensions
            zs (torch.Tensor): The noise tensors for stochastic sampling, of shape (L-1, N, *D)
                where L is the number of time steps
            idx (int): The current step index
            ts (torch.Tensor): The time steps tensor, of shape (L,)
                where L is the number of time steps

        Returns:
            torch.Tensor: The next state tensor, of shape (N, *D)
        """
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
        ) = self._get_step_quantities(zs, idx, ts)

        # Compute score at current state
        score_x_t = score(x, ts[idx])

        # Compute drift and diffusion terms
        drift = alpha_prime_t * x / alpha_t - sigma_t * sigma_prime_t * score_x_t
        diffusion = sigma_prime_t * dwt

        # Update state using Euler-Maruyama method
        x_next = x + drift * dt + diffusion

        return x_next
