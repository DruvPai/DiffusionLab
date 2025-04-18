import enum

from jax import Array


class VectorFieldType(enum.Enum):
    """
    Enum representing the type of a vector field.
    A vector field is a function that takes in ``x_t`` (``Array[*data_dims]``) and ``t`` (``Array[]``)
    and returns a vector of the same shape as ``x_t`` (``Array[*data_dims]``).

    DiffusionLab supports the following vector field types:

    - ``VectorFieldType.SCORE``: The score of the distribution.
    - ``VectorFieldType.X0``: The denoised state.
    - ``VectorFieldType.EPS``: The noise.
    - ``VectorFieldType.V``: The velocity field.
    """

    SCORE = enum.auto()
    X0 = enum.auto()
    EPS = enum.auto()
    V = enum.auto()


def convert_vector_field_type(
    x: Array,
    f_x: Array,
    alpha: Array,
    sigma: Array,
    alpha_prime: Array,
    sigma_prime: Array,
    in_type: VectorFieldType,
    out_type: VectorFieldType,
) -> Array:
    """
    Converts the output of a vector field from one type to another.

    Arguments:
        x (``Array[*data_dims]``): The input tensor.
        f_x (``Array[*data_dims]``): The output of the vector field f evaluated at x.
        alpha (``Array[]``): A scalar representing the scale parameter.
        sigma (``Array[]``): A scalar representing the noise level parameter.
        alpha_prime (``Array[]``): A scalar representing the scale derivative parameter.
        sigma_prime (``Array[]``): A scalar representing the noise level derivative parameter.
        in_type (``VectorFieldType``): The type of the input vector field (e.g. ``VectorFieldType.SCORE``, ``VectorFieldType.X0``, ``VectorFieldType.EPS``, ``VectorFieldType.V``).
        out_type (``VectorFieldType``): The type of the output vector field.

    Returns:
        ``Array[*data_dims]``: The converted output of the vector field
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
    alpha_ratio = alpha_prime / alpha
    sigma_ratio = sigma_prime / sigma
    ratio_diff = sigma_ratio - alpha_ratio
    converted_fx = f_x

    if in_type == VectorFieldType.SCORE:
        if out_type == VectorFieldType.X0:
            converted_fx = (x + sigma**2 * f_x) / alpha  # From equation (12)
        elif out_type == VectorFieldType.EPS:
            converted_fx = -sigma * f_x  # From equation (14)
        elif out_type == VectorFieldType.V:
            converted_fx = (
                alpha_ratio * x - sigma**2 * ratio_diff * f_x
            )  # From equation (16)

    elif in_type == VectorFieldType.X0:
        if out_type == VectorFieldType.SCORE:
            converted_fx = (alpha * f_x - x) / sigma**2  # From equation (13)
        elif out_type == VectorFieldType.EPS:
            converted_fx = (x - alpha * f_x) / sigma  # From equation (4)
        elif out_type == VectorFieldType.V:
            converted_fx = (
                sigma_ratio * x - alpha * ratio_diff * f_x
            )  # From equation (9)

    elif in_type == VectorFieldType.EPS:
        if out_type == VectorFieldType.SCORE:
            converted_fx = -f_x / sigma  # From equation (15)
        elif out_type == VectorFieldType.X0:
            converted_fx = (x - sigma * f_x) / alpha  # From equation (3)
        elif out_type == VectorFieldType.V:
            converted_fx = (
                alpha_ratio * x + sigma * ratio_diff * f_x
            )  # From equation (7)

    elif in_type == VectorFieldType.V:
        if out_type == VectorFieldType.SCORE:
            converted_fx = (alpha_ratio * x - f_x) / (
                sigma**2 * ratio_diff
            )  # From equation (17)
        elif out_type == VectorFieldType.X0:
            converted_fx = (sigma_ratio * x - f_x) / (
                alpha * ratio_diff
            )  # From equation (10)
        elif out_type == VectorFieldType.EPS:
            converted_fx = (f_x - alpha_ratio * x) / (
                sigma * ratio_diff
            )  # From equation (8)

    return converted_fx
