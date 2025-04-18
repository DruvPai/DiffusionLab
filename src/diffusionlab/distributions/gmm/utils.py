from jax import Array, numpy as jnp
from typing import cast, Callable, Tuple
from diffusionlab.dynamics import DiffusionProcess
from diffusionlab.vector_fields import VectorFieldType, convert_vector_field_type


def _logdeth(cov: Array) -> Array:
    """
    Computes the log determinant of a positive semi-definite (PSD) matrix.

    Uses ``eigh`` for numerical stability with symmetric matrices like covariance matrices.

    Args:
        cov (``Array[dim, dim]``): The input PSD matrix (e.g., a covariance matrix).

    Returns:
        ``Array[]``: The log determinant of the matrix (scalar).
    """
    eigvals = jnp.linalg.eigvalsh(cov)
    return jnp.sum(jnp.log(eigvals))


def _lstsq(A: Array, y: Array) -> Array:
    """
    Solves the linear system Ax = y using least squares.

    Handles potential conditioning issues by setting rcond based on machine epsilon.
    Equivalent to computing A^+ y where A^+ is the Moore-Penrose pseudoinverse.

    Args:
        A (``Array[out_dim, in_dim]``): The coefficient matrix.
        y (``Array[out_dim]``): The dependent variable values.

    Returns:
        ``Array[in_dim]``: The least-squares solution ``x``.
    """
    eps = cast(float, jnp.finfo(A.dtype).eps)
    x = jnp.linalg.lstsq(A, y, rcond=eps)[0]
    return x


def create_gmm_vector_field_fns(
    x0_fn: Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array],
) -> Tuple[
    Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array],
    Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array],
    Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array],
]:
    """
    Factory to create eps, score, and v functions from a given x0 function.

    Args:
        x0_fn: The specific x0 calculation function (e.g., ``gmm_x0``, ``iso_gmm_x0``).
               It must accept ``(x_t, t, diffusion_process, means, specific_cov, priors)``.

    Returns:
        ``Tuple[Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array], Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array], Callable[[Array, Array, DiffusionProcess, Array, Array, Array], Array]]``:
        A tuple containing the generated ``(eps_fn, score_fn, v_fn)``.
        These functions will have the same signature as ``x0_fn``, accepting
        ``(x_t, t, diffusion_process, means, specific_cov, priors)``.
    """

    def common_wrapper(
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
        means: Array,
        specific_cov: Array,
        priors: Array,
        target_type: VectorFieldType,
    ) -> Array:
        """Internal helper to perform the conversion."""
        x0_x_t = x0_fn(x_t, t, diffusion_process, means, specific_cov, priors)
        alpha_t = diffusion_process.alpha(t)
        sigma_t = diffusion_process.sigma(t)
        alpha_prime_t = diffusion_process.alpha_prime(t)
        sigma_prime_t = diffusion_process.sigma_prime(t)
        return convert_vector_field_type(
            x_t,
            x0_x_t,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            VectorFieldType.X0,
            target_type,
        )

    def eps_fn(
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
        means: Array,
        specific_cov: Array,
        priors: Array,
    ) -> Array:
        """Computes the noise prediction field ε based on the provided x0 function."""
        return common_wrapper(
            x_t, t, diffusion_process, means, specific_cov, priors, VectorFieldType.EPS
        )

    def score_fn(
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
        means: Array,
        specific_cov: Array,
        priors: Array,
    ) -> Array:
        """Computes the score field based on the provided x0 function."""
        return common_wrapper(
            x_t,
            t,
            diffusion_process,
            means,
            specific_cov,
            priors,
            VectorFieldType.SCORE,
        )

    def v_fn(
        x_t: Array,
        t: Array,
        diffusion_process: DiffusionProcess,
        means: Array,
        specific_cov: Array,
        priors: Array,
    ) -> Array:
        """Computes the velocity field v based on the provided x0 function."""
        return common_wrapper(
            x_t, t, diffusion_process, means, specific_cov, priors, VectorFieldType.V
        )

    # Add base docstrings - specific details might be lost compared to original funcs
    base_doc = f"Computes the {{}} field based on {x0_fn.__name__} by converting the x0 prediction.\n\n    Args:\n        x_t (Array[data_dim]): The noisy state tensor at time `t`.\n        t (Array[]): The time step (scalar).\n        diffusion_process (DiffusionProcess): Provides diffusion coefficients and derivatives.\n        means (Array[num_components, data_dim]): GMM component means.\n        specific_cov: GMM component specific covariance representation (covs, factors, variances, or variance).\n        priors (Array[num_components]): GMM component mixture weights.\n\n    Returns:\n        Array[data_dim]: The corresponding vector field evaluated at `x_t` and `t`."

    eps_fn.__doc__ = base_doc.format("noise prediction ε")
    score_fn.__doc__ = base_doc.format("score")
    v_fn.__doc__ = base_doc.format("velocity v")

    return eps_fn, score_fn, v_fn
