import torch
from torch.func import jacrev


from math import prod
from typing import Callable, cast


def scalar_derivative(
    f: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Computes the scalar derivative of a function f: R -> R.
    Returns a function f_prime: R -> R that computes the derivative of f at a given point,
    and is broadcastable with the same broadcast rules as f.

    Arguments:
        f: A function whose input is a scalar (0-dimensional Pytorch tensor) and whose output is a scalar, that can be broadcasted to a tensor of any shape.

    Returns:
        f_prime: A function that computes the derivative of f at a given point, and is broadcastable with the same broadcast rules as f.
    """
    df = jacrev(f)

    def f_prime(x: torch.Tensor) -> torch.Tensor:
        dfx = cast(torch.Tensor, df(x))
        if dfx.ndim > 1:
            x_size = prod(x.shape)
            dfx = dfx.reshape(x_size, x_size)
            dfx = dfx.diagonal(dim1=0, dim2=1)
            dfx = dfx.reshape(x.shape)
        return dfx

    return f_prime


def pad_shape_front(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Pads the front of a tensor with singleton dimensions until it can broadcast with target_shape.

    Arguments:
        x: A tensor of any shape, say (P, Q, R, S).
        target_shape: A shape to which x can broadcast, say (M, N, O, P, Q, R, S).

    Returns:
        x_padded: The tensor x reshaped to be broadcastable with target_shape, say (1, 1, 1, P, Q, R, S).

    Note:
        This function does not use any additional memory, returning a different view of the same underlying data.
    """
    ndim_target = len(target_shape)
    ndim_x = x.ndim
    expand_dims = (1,) * max(ndim_target - ndim_x, 0)
    return x.view(*expand_dims, *x.shape)


def pad_shape_back(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """
    Pads the back of a tensor with singleton dimensions until it can broadcast with target_shape.

    Arguments:
        x: A tensor of any shape, say (P, Q, R, S).
        target_shape: A shape to which x can broadcast, say (P, Q, R, S, T, U, V).

    Returns:
        x_padded: The tensor x reshaped to be broadcastable with target_shape, say (P, Q, R, S, 1, 1, 1).

    Note:
        This function does not use any additional memory, returning a different view of the same underlying data.
    """
    ndim_target = len(target_shape)
    ndim_x = x.ndim
    expand_dims = (1,) * max(ndim_target - ndim_x, 0)
    return x.view(*x.shape, *expand_dims)


def vector_lstsq(A: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.linalg.lstsq(A, y[..., None]).solution[..., 0]


def logdet_pd(A: torch.Tensor) -> torch.Tensor:
    L = torch.linalg.cholesky(A)
    eigvals = torch.diagonal(L, dim1=-2, dim2=-1)
    return 2 * torch.sum(torch.log(eigvals), dim=-1)


def sqrt_psd(A: torch.Tensor) -> torch.Tensor:
    L, Q = torch.linalg.eigh(A)
    return Q @ torch.diag_embed(torch.sqrt(L)) @ Q.transpose(-2, -1)
