"""Shared components for diffusion model architectures.

Provides reusable building blocks: sinusoidal time embeddings,
FiLM conditioning layers, and adaptive layer normalization (AdaLN-Zero).
"""

from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from diffusionlab.typing import PRNGKey
from diffusionlab.utils.tree_ops import bcast_left


class SinusoidalEmbedding(eqx.Module):
    """Sinusoidal positional/time embedding (DDPM-style).

    Maps scalar values to dense vectors using fixed sinusoidal frequencies.

    Attributes:
        emb_dim: Dimensionality of the output embedding. Must be even.
        max_period: Maximum period for the sinusoidal frequencies.
    """

    emb_dim: int = eqx.field(static=True)
    max_period: float = eqx.field(static=True)

    def __call__(
        self, t: Float[Array, " batch"]
    ) -> Float[Array, "batch {self.emb_dim}"]:
        """Embed scalar values into dense sinusoidal vectors.

        Args:
            t: Scalar values to embed, shape ``(batch,)``.

        Returns:
            Embedding vectors, shape ``(batch, emb_dim)``.
        """
        half = self.emb_dim // 2
        freqs = jnp.exp(-jnp.log(self.max_period) * jnp.arange(half) / half)  # (half,)
        args = t[:, None] * freqs[None, :]  # (batch, half)
        return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class FiLM(eqx.Module):
    """Feature-wise Linear Modulation (FiLM) layer.

    Applies affine conditioning after layer normalization:
    ``(1 + gamma) * LayerNorm(Linear(h)) + beta``
    where ``[beta, gamma] = film_proj(cond)``.

    Attributes:
        proj: Conditioning projection, ``(cond_dim,) -> (2 * hidden_dim,)``.
        norm: RMS normalization over hidden_dim.
    """

    proj: eqx.nn.MLP = eqx.field(static=False)
    norm: eqx.nn.LayerNorm = eqx.field(static=False)

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        key: PRNGKey,
    ):
        """Initialize a FiLM layer.

        Args:
            hidden_dim: Dimensionality of the hidden representation.
            cond_dim: Dimensionality of the conditioning vector.
            key: PRNG key for parameter initialization.
        """
        self.proj = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=2 * hidden_dim,
            width_size=2 * hidden_dim,
            depth=2,
            key=key,
            activation=jax.nn.silu,
        )
        self.norm = eqx.nn.LayerNorm(hidden_dim, use_weight=False, use_bias=False)

    def __call__(
        self,
        h: Float[Array, " hidden_dim"],
        cond: Float[Array, " cond_dim"],
    ) -> Float[Array, " hidden_dim"]:
        """Apply FiLM-conditioned transformation.

        Args:
            h: Input hidden state, shape ``(hidden_dim,)``.
            cond: Conditioning vector, shape ``(cond_dim,)``.

        Returns:
            Modulated hidden state, shape ``(hidden_dim,)``.
        """
        modulation = self.proj(cond)  # (2 * hidden_dim,)
        beta, gamma = jnp.split(modulation, 2, axis=-1)  # each (hidden_dim,)
        return beta + gamma * self.norm(h)


class AdaLNZero(eqx.Module):
    """Adaptive Layer Normalization with zero-initialization (AdaLN-Zero).

    Produces modulation vectors (gamma, beta, alpha) from a conditioning vector.
    from a conditioning vector. The projection is zero-initialized so the model
    starts near identity.

    Attributes:
        proj: Zero-initialized MLP projection, ``(cond_dim,) -> (3 * hidden_dim,)``.
        norm: RMS normalization for the hidden state.
    """

    proj: eqx.nn.MLP = eqx.field(static=False)
    norm: eqx.nn.LayerNorm = eqx.field(static=False)

    def __init__(self, hidden_dim: int, cond_dim: int, key: PRNGKey):
        """Initialize AdaLN-Zero.

        Args:
            hidden_dim: Dimensionality of the hidden representation.
            cond_dim: Dimensionality of the conditioning vector.
            key: PRNG key for parameter initialization.
        """
        self.proj = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=3 * hidden_dim,
            width_size=3 * hidden_dim,
            depth=2,
            key=key,
            activation=jax.nn.silu,
        )
        self.norm = eqx.nn.LayerNorm(hidden_dim, use_weight=False, use_bias=False)

    def get_modulation_params(
        self, cond: Float[Array, " cond_dim"]
    ) -> tuple[
        Float[Array, " hidden_dim"],
        Float[Array, " hidden_dim"],
        Float[Array, " hidden_dim"],
    ]:
        """Compute three modulation vectors from a conditioning vector.

        Args:
            cond: Conditioning vector, shape ``(cond_dim,)``.

        Returns:
            Tuple of (alpha, beta, gamma),
            each shape ``(hidden_dim,)``.
        """
        params = self.proj(cond)  # (3 * hidden_dim,)
        return cast(
            tuple[Array, Array, Array],
            tuple(jnp.split(params, 3, axis=-1)),
        )

    def modulate(
        self,
        h: Float[Array, "*seq hidden_dim"],
        beta: Float[Array, " hidden_dim"],
        gamma: Float[Array, " hidden_dim"],
    ) -> Float[Array, "*seq hidden_dim"]:
        """Apply adaptive layer normalization.

        Computes ``beta + gamma * norm(h)``.

        Args:
            h: Input hidden state, shape ``(*seq, hidden_dim)``.
            gamma: Scale modulation, shape ``(hidden_dim,)``.
            beta: Shift modulation, shape ``(hidden_dim,)``.

        Returns:
            Modulated hidden state, shape ``(*seq, hidden_dim)``.
        """
        gamma = bcast_left(gamma, h.ndim)
        beta = bcast_left(beta, h.ndim)
        return (1 + gamma) * jax.vmap(self.norm)(h) + beta
