from typing import Any, Dict, Tuple

import torch

from diffusionlab.samplers import Sampler
from diffusionlab.vector_fields import VectorFieldType, convert_vector_field_type


class Distribution:
    """
    Base class for all distributions.
    It should be subclassed by other distributions for you want to use the ground truth
    scores (resp. denoisers, noise predictors, velocity estimators).
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
    def validate_params(
        cls, possibly_batched_dist_params: Dict[str, torch.Tensor]
    ) -> None:
        """
        Validate the parameters for the distribution.

        Arguments:
            dist_params: A dictionary of parameters for the distribution. Each value is a PyTorch tensor, possibly having a batch dimension.

        Returns:
            None

        Throws:
            AssertionError: If the parameters are invalid, the assertion fails at exactly the point of failure.
        """
        assert len(possibly_batched_dist_params) == 0

    @classmethod
    def x0(
        cls,
        xt: torch.Tensor,
        t: torch.Tensor,
        sampler: Sampler,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the denoiser E[x0 | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).

        Arguments:
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            batched_dist_params: A dictionary of batched parameters for the distribution. Each parameter is of shape (N, *P) where P is the shape of the parameter.
            dist_hparams: A dictionary of hyperparameters for the distribution.
        Returns:
            The prediction of x0, of shape (N, *D).
        """
        raise NotImplementedError

    @classmethod
    def eps(
        cls,
        xt: torch.Tensor,
        t: torch.Tensor,
        sampler: Sampler,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the noise predictor E[eps | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            batched_dist_params: A dictionary of batched parameters for the distribution. Each parameter is of shape (N, *P) where P is the shape of the parameter.
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            The prediction of eps, of shape (N, *D).
        """
        x0_hat = cls.x0(xt, t, sampler, batched_dist_params, dist_hparams)
        eps_hat = convert_vector_field_type(
            xt,
            x0_hat,
            sampler.alpha(t),
            sampler.sigma(t),
            sampler.alpha_prime(t),
            sampler.sigma_prime(t),
            in_type=VectorFieldType.X0,
            out_type=VectorFieldType.EPS,
        )
        return eps_hat

    @classmethod
    def v(
        cls,
        xt: torch.Tensor,
        t: torch.Tensor,
        sampler: Sampler,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the velocity estimator E[d/dt xt | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            batched_dist_params: A dictionary of batched parameters for the distribution. Each parameter is of shape (N, *P) where P is the shape of the parameter.
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            The prediction of d/dt xt, of shape (N, *D).
        """
        x0_hat = cls.x0(xt, t, sampler, batched_dist_params, dist_hparams)
        v_hat = convert_vector_field_type(
            xt,
            x0_hat,
            sampler.alpha(t),
            sampler.sigma(t),
            sampler.alpha_prime(t),
            sampler.sigma_prime(t),
            in_type=VectorFieldType.X0,
            out_type=VectorFieldType.V,
        )
        return v_hat

    @classmethod
    def score(
        cls,
        xt: torch.Tensor,
        t: torch.Tensor,
        sampler: Sampler,
        batched_dist_params: Dict[str, torch.Tensor],
        dist_hparams: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Computes the score estimator grad_x log p(xt, t) at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            batched_dist_params: A dictionary of batched parameters for the distribution. Each parameter is of shape (N, *P) where P is the shape of the parameter.
            dist_hparams: A dictionary of hyperparameters for the distribution.

        Returns:
            The prediction of grad_x log p(xt, t), of shape (N, *D).
        """
        x0_hat = cls.x0(xt, t, sampler, batched_dist_params, dist_hparams)
        score_hat = convert_vector_field_type(
            xt,
            x0_hat,
            sampler.alpha(t),
            sampler.sigma(t),
            sampler.alpha_prime(t),
            sampler.sigma_prime(t),
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
            Note that the samples are placed on the device corresponding to the sampler.
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
