from typing import Any, Dict, Tuple, Callable

import torch

from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType, convert_vector_field_type


class Distribution:
    """
    Base class for all distributions.

    This class should be subclassed by other distributions when you want to use ground truth
    scores, denoisers, noise predictors, or velocity estimators.

    Each distribution implementation provides methods to compute various vector fields
    related to the diffusion process, such as denoising (x0), noise prediction (eps),
    velocity estimation (v), and score estimation.
    """

    @classmethod
    def validate_hparams(cls, dist_hparams: Dict[str, Any]) -> None:
        """
        Validate the hyperparameters for the distribution.

        Arguments:
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            None

        Throws:
            AssertionError: If the parameters are invalid, the assertion fails at exactly the point of failure.
        """
        assert len(dist_hparams) == 0

    @classmethod
    def get_vector_field_method(
        cls, vector_field_type: VectorFieldType
    ) -> Callable[
        [
            torch.Tensor,
            torch.Tensor,
            DiffusionProcess,
            Dict[str, torch.Tensor],
            Dict[str, Any],
        ],
        torch.Tensor,
    ]:
        """
        Returns the appropriate method to compute the specified vector field type.

        Arguments:
            vector_field_type: The type of vector field to compute.

        Returns:
            A method that computes the specified vector field, with signature:
            (x_t, t, diffusion_process, batched_dist_params, dist_hparams) -> tensor

        Raises:
            ValueError: If the vector field type is not recognized.
        """
        if vector_field_type == VectorFieldType.X0:
            return cls.x0
        elif vector_field_type == VectorFieldType.EPS:
            return cls.eps
        elif vector_field_type == VectorFieldType.V:
            return cls.v
        elif vector_field_type == VectorFieldType.SCORE:
            return cls.score
        else:
            raise ValueError(f"Unrecognized vector field type: {vector_field_type}")

    @classmethod
    def validate_params(
        cls, possibly_batched_dist_params: Dict[str, torch.Tensor]
    ) -> None:
        """
        Validate the parameters for the distribution.

        Arguments:
            possibly_batched_dist_params: A dictionary of parameters for the distribution.
                Each value is a PyTorch tensor, possibly having a batch dimension.

        Returns:
            None

        Throws:
            AssertionError: If the parameters are invalid, the assertion fails at exactly the point of failure.
        """
        assert len(possibly_batched_dist_params) == 0

    @classmethod
    def x0(
        cls,
        x_t: torch.Tensor,
        t: torch.Tensor,
        diffusion_process: DiffusionProcess,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the denoiser E[x_0 | x_t] at a given time t and input x_t, under the data model

        x_t = alpha(t) * x_0 + sigma(t) * eps

        where x_0 is drawn from the data distribution, and eps is drawn independently from N(0, I).

        Arguments:
            x_t: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            diffusion_process: The diffusion process whose forward and reverse dynamics determine
                the time-evolution of the vector fields corresponding to the distribution.
            batched_dist_params: A dictionary of batched parameters for the distribution.
                Each parameter is of shape (N, *P) where P is the shape of the parameter.
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            The prediction of x_0, of shape (N, *D).

        Note:
            The batched_dist_params dictionary contains BATCHED tensors, i.e., the first dimension is the batch dimension.
        """
        raise NotImplementedError

    @classmethod
    def eps(
        cls,
        x_t: torch.Tensor,
        t: torch.Tensor,
        diffusion_process: DiffusionProcess,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the noise predictor E[eps | x_t] at a given time t and input x_t, under the data model

        x_t = alpha(t) * x_0 + sigma(t) * eps

        where x_0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            x_t: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            diffusion_process: The diffusion process whose forward and reverse dynamics determine
                the time-evolution of the vector fields corresponding to the distribution.
            batched_dist_params: A dictionary of batched parameters for the distribution.
                Each parameter is of shape (N, *P) where P is the shape of the parameter.
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            The prediction of eps, of shape (N, *D).

        Note:
            The batched_dist_params dictionary contains BATCHED tensors, i.e., the first dimension is the batch dimension.
        """
        x0_hat = cls.x0(x_t, t, diffusion_process, batched_dist_params, dist_hparams)
        eps_hat = convert_vector_field_type(
            x_t,
            x0_hat,
            diffusion_process.alpha(t),
            diffusion_process.sigma(t),
            diffusion_process.alpha_prime(t),
            diffusion_process.sigma_prime(t),
            in_type=VectorFieldType.X0,
            out_type=VectorFieldType.EPS,
        )
        return eps_hat

    @classmethod
    def v(
        cls,
        x_t: torch.Tensor,
        t: torch.Tensor,
        diffusion_process: DiffusionProcess,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the velocity estimator E[d/dt x_t | x_t] at a given time t and input x_t, under the data model

        x_t = alpha(t) * x_0 + sigma(t) * eps

        where x_0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            x_t: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            diffusion_process: The diffusion process whose forward and reverse dynamics determine
                the time-evolution of the vector fields corresponding to the distribution.
            batched_dist_params: A dictionary of batched parameters for the distribution.
                Each parameter is of shape (N, *P) where P is the shape of the parameter.
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            The prediction of d/dt x_t, of shape (N, *D).

        Note:
            The batched_dist_params dictionary contains BATCHED tensors, i.e., the first dimension is the batch dimension.
        """
        x0_hat = cls.x0(x_t, t, diffusion_process, batched_dist_params, dist_hparams)
        v_hat = convert_vector_field_type(
            x_t,
            x0_hat,
            diffusion_process.alpha(t),
            diffusion_process.sigma(t),
            diffusion_process.alpha_prime(t),
            diffusion_process.sigma_prime(t),
            in_type=VectorFieldType.X0,
            out_type=VectorFieldType.V,
        )
        return v_hat

    @classmethod
    def score(
        cls,
        x_t: torch.Tensor,
        t: torch.Tensor,
        diffusion_process: DiffusionProcess,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the score estimator grad_x log p(x_t, t) at a given time t and input x_t, under the data model

        x_t = alpha(t) * x_0 + sigma(t) * eps

        where x_0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            x_t: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            diffusion_process: The diffusion process whose forward and reverse dynamics determine
                the time-evolution of the vector fields corresponding to the distribution.
            batched_dist_params: A dictionary of batched parameters for the distribution.
                Each parameter is of shape (N, *P) where P is the shape of the parameter.
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            The prediction of grad_x log p(x_t, t), of shape (N, *D).

        Note:
            The batched_dist_params dictionary contains BATCHED tensors, i.e., the first dimension is the batch dimension.
        """
        x0_hat = cls.x0(x_t, t, diffusion_process, batched_dist_params, dist_hparams)
        score_hat = convert_vector_field_type(
            x_t,
            x0_hat,
            diffusion_process.alpha(t),
            diffusion_process.sigma(t),
            diffusion_process.alpha_prime(t),
            diffusion_process.sigma_prime(t),
            in_type=VectorFieldType.X0,
            out_type=VectorFieldType.SCORE,
        )
        return score_hat

    @classmethod
    def sample(
        cls,
        N: int,
        dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Any]:
        """
        Draws N i.i.d. samples from the data distribution.

        Arguments:
            N: The number of samples to draw.
            dist_params: A dictionary of parameters for the distribution.
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            A tuple (samples, metadata), where samples is a tensor of shape (N, *D) and metadata is any additional information.
            For example, if the distribution has labels, the metadata is a tensor of shape (N, ) containing the labels.
            Note that the samples are always placed on the CPU.
        """
        raise NotImplementedError

    @staticmethod
    def batch_dist_params(
        N: int, dist_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Add a batch dimension to the distribution parameters.

        Arguments:
            N: The number of samples in the batch.
            dist_params: A dictionary of parameters for the distribution.

        Returns:
            A dictionary of parameters for the distribution, with a batch dimension added.
        """
        return {k: v.unsqueeze(0).expand(N, *v.shape) for k, v in dist_params.items()}
