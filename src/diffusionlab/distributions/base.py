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

    def __init__(self, sampler: Sampler, dist_params: Dict[str, Any]):
        """
        Initialize the distribution.

        Arguments:
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            dist_params: A dictionary of parameters for the distribution.
        """
        super().__init__()
        self.__class__.validate_params(dist_params)
        self.sampler: Sampler = sampler
        self.dist_params: Dict[str, Any] = dist_params

    @classmethod
    def validate_params(cls, dist_params: Dict[str, Any]) -> None:
        """
        Validate the parameters for the distribution. Run at initialization, and can be run elsewhere where desired.

        Arguments:
            dist_params: A dictionary of parameters for the distribution.

        Returns:
            None

        Throws:
            AssertionError: If the parameters are invalid, the assertion fails at exactly the point of failure.
        """
        raise NotImplementedError

    @classmethod
    def stateless_x0(
        cls,
        sampler: Sampler,
        dist_params: Dict[str, Any],
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the denoiser E[x0 | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).

        This is stateless for efficiency's sake: in high-performance training it is actually pretty costly to initialize a new class every iteration so that we can obtain a denoiser of desired functional form.

        Arguments:
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            dist_params: A dictionary of parameters for the distribution.
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).

        Returns:
            The prediction of x0, of shape (N, *D).
        """
        raise NotImplementedError

    @classmethod
    def stateless_eps(
        cls,
        sampler: Sampler,
        dist_params: Dict[str, Any],
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the noise predictor E[eps | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            dist_params: A dictionary of parameters for the distribution.
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).

        Returns:
            The prediction of eps, of shape (N, *D).
        """
        x0_hat = cls.stateless_x0(sampler, dist_params, x, t)
        eps_hat = convert_vector_field_type(
            x,
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
    def stateless_v(
        cls,
        sampler: Sampler,
        dist_params: Dict[str, Any],
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the velocity estimator E[d/dt xt | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            dist_params: A dictionary of parameters for the distribution.
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).

        Returns:
            The prediction of d/dt xt, of shape (N, *D).
        """
        x0_hat = cls.stateless_x0(sampler, dist_params, xt, t)
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
    def stateless_score(
        cls,
        sampler: Sampler,
        dist_params: Dict[str, Any],
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the score estimator grad_x log p(xt, t) at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).
        This is stateless for the same reason as the denoiser method.

        Arguments:
            sampler: The sampler whose forward and reverse dynamics determine the time-evolution of the vector fields corresponding to the distribution.
            dist_params: A dictionary of parameters for the distribution.
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).

        Returns:
            The prediction of grad_x log p(xt, t), of shape (N, *D).
        """
        x0_hat = cls.stateless_x0(sampler, dist_params, x, t)
        score_hat = convert_vector_field_type(
            x,
            x0_hat,
            sampler.alpha(t),
            sampler.sigma(t),
            sampler.alpha_prime(t),
            sampler.sigma_prime(t),
            in_type=VectorFieldType.X0,
            out_type=VectorFieldType.SCORE,
        )
        return score_hat

    def x0(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the denoiser E[x0 | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).

        Arguments:
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).

        Returns:
            The prediction of x0, of shape (N, *D).
        """
        return self.stateless_x0(self.sampler, self.dist_params, xt, t)

    def eps(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise predictor E[eps | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).

        Arguments:
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data, and having law equal to xt.
            t: The time tensor, of shape (N, ).

        Returns:
            The prediction of eps, of shape (N, *D).
        """
        return self.stateless_eps(self.sampler, self.dist_params, xt, t)

    def v(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the velocity estimator E[d/dt xt | xt] at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).

        Arguments:
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).

        Returns:
            The prediction of d/dt xt, of shape (N, *D).
        """
        return self.stateless_v(self.sampler, self.dist_params, xt, t)

    def score(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the score estimator grad_x log p(xt, t) at a given time t and input xt, under the data model

        xt = alpha(t) * x0 + sigma(t) * eps

        where x0 is drawn from the data distribution, and eps is drawn independently from N(0, I).

        Arguments:
            xt: The input tensor, of shape (N, *D), where *D is the shape of each data.
            t: The time tensor, of shape (N, ).

        Returns:
            The prediction of grad_x log p(xt, t), of shape (N, *D).
        """
        return self.stateless_score(self.sampler, self.dist_params, xt, t)

    def sample(self, N: int) -> Tuple[torch.Tensor, Any]:
        """
        Draws N i.i.d. samples from the data distribution.

        Arguments:
            N: The number of samples to draw.

        Returns:
            A tuple (samples, metadata), where samples is a tensor of shape (N, *D) and metadata is any additional information.
            For example, if the distribution has labels, the metadata is a tensor of shape (N, ) containing the labels.
            Note that the samples are placed on the device corresponding to the sampler.
        """
        raise NotImplementedError
