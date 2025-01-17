from typing import Any, Dict, Literal, Tuple
import torch
from diffusionlab.samplers import Sampler
from diffusionlab.vector_fields import VectorFieldType, convert_vector_field_type


class Distribution:
    def __init__(self, sampler: Sampler, dist_params: Dict[str, Any]):
        super().__init__()
        self.__class__.validate_params(dist_params)
        self.sampler: Sampler = sampler
        self.dist_params: Dict[str, Any] = dist_params

    @classmethod
    def validate_params(cls, dist_params: Dict[str, Any]) -> None:
        raise NotImplementedError
    
    @classmethod
    def stateless_x0(cls, sampler: Sampler, dist_params: Dict[str, Any], x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @classmethod
    def stateless_eps(cls, sampler: Sampler, dist_params: Dict[str, Any], x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0_hat = cls.stateless_x0(sampler, dist_params, x, t)
        eps_hat = convert_vector_field_type(
            x, x0_hat, 
            sampler.alpha(t), sampler.sigma(t), 
            sampler.alpha_prime(t), sampler.sigma_prime(t), 
            in_type=VectorFieldType.X0, out_type=VectorFieldType.EPS
        )
        return eps_hat
    
    @classmethod
    def stateless_v(cls, sampler: Sampler, dist_params: Dict[str, Any], x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0_hat = cls.stateless_x0(sampler, dist_params, x, t)
        v_hat = convert_vector_field_type(
            x, x0_hat, 
            sampler.alpha(t), sampler.sigma(t), 
            sampler.alpha_prime(t), sampler.sigma_prime(t), 
            in_type=VectorFieldType.X0, out_type=VectorFieldType.V
        )
        return v_hat
    
    @classmethod
    def stateless_score(cls, sampler: Sampler, dist_params: Dict[str, Any], x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0_hat = cls.stateless_x0(sampler, dist_params, x, t)
        score_hat = convert_vector_field_type(
            x, x0_hat, 
            sampler.alpha(t), sampler.sigma(t), 
            sampler.alpha_prime(t), sampler.sigma_prime(t), 
            in_type=VectorFieldType.X0, out_type=VectorFieldType.SCORE
        )
        return score_hat

    def x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.stateless_x0(self.sampler, self.dist_params, x, t)
    
    def eps(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.stateless_eps(self.sampler, self.dist_params, x, t)
    
    def v(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.stateless_v(self.sampler, self.dist_params, x, t)
    
    def score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.stateless_score(self.sampler, self.dist_params, x, t)
    
    def sample(self, N: int) -> Tuple[torch.Tensor, Any]:
        raise NotImplementedError
