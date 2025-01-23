import enum
from typing import Callable

import torch

from diffusionlab.utils import pad_shape_back


class VectorFieldType(enum.Enum):
    SCORE = enum.auto()
    X0 = enum.auto()
    EPS = enum.auto()
    V = enum.auto()


class VectorField:
    """
    A wrapper around a function (x, t) -> f(x, t) which provides some extra data,
    namely the type of vector field the function f represents.
    """

    def __init__(
        self,
        f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        vector_field_type: VectorFieldType,
    ):
        self.f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = f
        self.vector_field_type: VectorFieldType = vector_field_type

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.f(x, t)


def convert_vector_field_type(
    x: torch.Tensor,
    fx: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    alpha_prime: torch.Tensor,
    sigma_prime: torch.Tensor,
    in_type: VectorFieldType,
    out_type: VectorFieldType,
) -> torch.Tensor:
    """
    Converts the output of a vector field from one type to another.

    Arguments:
        x: A tensor of shape (N, *D), where D is the shape of the data (say (C, H, W) for images or (D, ) for vectors or (N, D) for token sequences).
        fx: The output of the vector field f, of shape (N, *D).
        alpha: A tensor of shape (N,) representing the scale parameter.
        sigma: A tensor of shape (N,) representing the noise level parameter.
        alpha_prime: A tensor of shape (N,) representing the scale derivative parameter.
        sigma_prime: A tensor of shape (N,) representing the noise level derivative parameter.
        in_type: The type of the input vector field (e.g. Score, X0, Eps, V).
        out_type: The type of the output vector field.

    Returns:
        The converted output of the vector field, of shape (N, *D).
    """
    """
    Derivation:
    ----------------------------
    Define certain quantities:
    alpha_r = alpha' / alpha
    sigma_r = sigma' / sigma
    diff_r = sigma_r - alpha_r
    and note that diff_r >= 0 since alpha' < 0 and all other terms are > 0. 
    Under the data model 
    (1) x := alpha * x0 + sigma * eps
    it holds that 
    (2) x = alpha * E[x0 | x] + sigma * E[eps | x]
    Therefore 
    (3) E[x0 | x] = (x - sigma * E[eps | x]) / alpha
    (4) E[eps | x] = (x - alpha * E[x0 | x]) / sigma
    Furthermore, from (1) it holds that
    (5) v := x' = alpha' * x0 + sigma' * eps,
    or in particular
    (6) E[v | x] = alpha' * E[x0 | x] + sigma' * E[eps | x]
    Using (3), (4), (6) it holds 
    (7) E[v | x] = alpha_r * (x - sigma * E[eps | x]) + sigma' * E[eps | x] 
    => E[v | x] = alpha'/alpha * x + (sigma' - sigma * alpha'/alpha) * E[eps | x]
    => E[v | x] = alpha'/alpha * x + sigma * (sigma'/sigma - alpha'/alpha) * E[eps | x]
    => E[v | x] = alpha_r * x + sigma * diff_r * E[eps | x]
    (8) E[eps | x] = (E[v | x] - alpha_r * x) / (sigma * diff_r)
    and, similarly,
    (9) E[v | x] = alpha' * E[x0 | x] + sigma'/sigma * (x - alpha * E[x0 | x]) 
    => E[v | x] = sigma'/sigma * x + (alpha' - alpha * sigma'/sigma) * E[x0 | x]
    => E[v | x] = sigma'/sigma * x + alpha * (alpha'/alpha - sigma'/sigma) * E[x0 | x]
    => E[v | x] = sigma_r * x - alpha * diff_r * E[x0 | x]
    (10) E[x0 | x] = (sigma_r * x - E[v | x]) / (alpha * diff_r)
    To connect the score function to the other types, we use Tweedie's formula:
    (11) alpha * E[x0 | x] = x + sigma^2 * score(x, alpha, sigma).
    Therefore, from (11):
    (12) E[x0 | x] = (x + sigma^2 * score(x, alpha, sigma)) / alpha
    From (12):
    (13) score(x, alpha, sigma) = (alpha * E[x0 | x] - x) / sigma^2
    From (11) and (4):
    (14) E[eps | x] = -sigma * score(x, alpha, sigma)
    From (14):
    (15) score(x, alpha, sigma) = -E[eps | x] / sigma
    From (14) and (7):
    (16) E[v | x] = alpha_r * x - sigma^2 * diff_r * score(x, alpha, sigma)
    From (16):
    (17) score(x, alpha, sigma) = (alpha_r * x - E[v | x]) / (sigma^2 * diff_r)
    """
    alpha = pad_shape_back(alpha, x.shape)
    alpha_prime = pad_shape_back(alpha_prime, x.shape)
    sigma = pad_shape_back(sigma, x.shape)
    sigma_prime = pad_shape_back(sigma_prime, x.shape)
    alpha_ratio = alpha_prime / alpha
    sigma_ratio = sigma_prime / sigma
    ratio_diff = sigma_ratio - alpha_ratio
    converted_fx = fx

    if in_type == VectorFieldType.SCORE:
        if out_type == VectorFieldType.X0:
            converted_fx = (x + sigma**2 * fx) / alpha  # From equation (12)
        elif out_type == VectorFieldType.EPS:
            converted_fx = -sigma * fx  # From equation (14)
        elif out_type == VectorFieldType.V:
            converted_fx = (
                alpha_ratio * x - sigma**2 * ratio_diff * fx
            )  # From equation (16)

    elif in_type == VectorFieldType.X0:
        if out_type == VectorFieldType.SCORE:
            converted_fx = (alpha * fx - x) / sigma**2  # From equation (13)
        elif out_type == VectorFieldType.EPS:
            converted_fx = (x - alpha * fx) / sigma  # From equation (4)
        elif out_type == VectorFieldType.V:
            converted_fx = (
                sigma_ratio * x - alpha * ratio_diff * fx
            )  # From equation (9)

    elif in_type == VectorFieldType.EPS:
        if out_type == VectorFieldType.SCORE:
            converted_fx = -fx / sigma  # From equation (15)
        elif out_type == VectorFieldType.X0:
            converted_fx = (x - sigma * fx) / alpha  # From equation (3)
        elif out_type == VectorFieldType.V:
            converted_fx = (
                alpha_ratio * x + sigma * ratio_diff * fx
            )  # From equation (7)

    elif in_type == VectorFieldType.V:
        if out_type == VectorFieldType.SCORE:
            converted_fx = (alpha_ratio * x - fx) / (
                sigma**2 * ratio_diff
            )  # From equation (17)
        elif out_type == VectorFieldType.X0:
            converted_fx = (sigma_ratio * x - fx) / (
                alpha * ratio_diff
            )  # From equation (10)
        elif out_type == VectorFieldType.EPS:
            converted_fx = (fx - alpha_ratio * x) / (
                sigma * ratio_diff
            )  # From equation (8)

    return converted_fx
