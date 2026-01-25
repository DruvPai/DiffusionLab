"""Diffusion Transformer (DiT) architectures for sequence data."""

from typing import Callable

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from diffusionlab.models.base import DiffusionModel
from diffusionlab.models.components import AdaLNZero, SinusoidalEmbedding
from diffusionlab.typing import (
    AuxInfoTree,
    ConditioningTree,
    DataTree,
    Prediction,
    PredictionTree,
    TimeTree,
)
from diffusionlab.utils.tree_ops import bcast_left


class DiTBlock(eqx.Module):
    """Single transformer block with AdaLN-Zero conditioning.

    Consists of a multi-head self-attention sub-block and a two-layer MLP
    sub-block, both modulated by adaptive layer normalization with
    zero-initialized gating.

    Attributes:
        ln1: Layer normalization before attention.
        ln2: Layer normalization before MLP.
        attn: Multi-head self-attention layer.
        mlp_linear1: First MLP linear layer,
            ``(hidden_dim,) -> (mlp_dim,)``.
        mlp_linear2: Second MLP linear layer,
            ``(mlp_dim,) -> (hidden_dim,)``.
        activation: Elementwise activation function,
            ``(mlp_dim,) -> (mlp_dim,)``.
    """

    ln1: AdaLNZero
    ln2: AdaLNZero
    attn: eqx.nn.MultiheadAttention
    mlp: eqx.nn.MLP

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        cond_dim: int,
        activation: Callable,
        key: jax.Array,
    ):
        """Initialize a DiT block.

        Args:
            hidden_dim: Dimensionality of the hidden representation.
            num_heads: Number of attention heads.
            head_dim: Dimensionality of each attention head.
            mlp_dim: Dimensionality of the MLP intermediate layer.
            cond_dim: Dimensionality of the conditioning vector.
            activation: Elementwise activation function,
                ``(mlp_dim,) -> (mlp_dim,)``.
            key: PRNG key for parameter initialization.
        """
        key_ln1, key_ln2, key_attn, key_mlp = jax.random.split(key, 4)
        self.ln1 = AdaLNZero(hidden_dim, cond_dim, key=key_ln1)
        self.ln2 = AdaLNZero(hidden_dim, cond_dim, key=key_ln2)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_dim,
            qk_size=head_dim,
            vo_size=head_dim,
            key=key_attn,
        )
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim,
            width_size=mlp_dim,
            depth=2,
            key=key_mlp,
            activation=activation,
        )

    def __call__(
        self,
        h: Float[Array, "seq_len hidden_dim"],
        cond: Float[Array, " cond_dim"],
    ) -> Float[Array, "seq_len hidden_dim"]:
        """Apply the DiT block.

        Args:
            h: Input sequence, shape ``(seq_len, hidden_dim)``.
            cond: Conditioning vector, shape ``(cond_dim,)``.

        Returns:
            Output sequence, shape ``(seq_len, hidden_dim)``.
        """
        alpha1, beta1, gamma1 = self.ln1.get_modulation_params(cond)
        h_norm = self.ln1.modulate(h, beta1, gamma1)
        h = h + bcast_left(alpha1, h.ndim) * self.attn(h_norm, h_norm, h_norm)

        alpha2, beta2, gamma2 = self.ln2.get_modulation_params(cond)
        h_norm = self.ln2.modulate(h, beta2, gamma2)
        h = h + bcast_left(alpha2, h.ndim) * jax.vmap(self.mlp)(h_norm)

        return h


class DiT(DiffusionModel):
    """Diffusion Transformer for sequence data.

    Processes data of shape ``(batch, seq_len, input_dim)`` using a stack of
    DiT blocks with AdaLN-Zero conditioning from sinusoidal time embeddings.
    Learnable positional embeddings are added to the input tokens.

    Attributes:
        time_embed: Sinusoidal time embedding module.
        time_proj: Projects time embedding to conditioning dimension,
            ``(time_emb_dim,) -> (cond_dim,)``.
        input_proj: Projects input tokens to hidden dimension,
            ``(input_dim,) -> (hidden_dim,)``.
        pos_embed: Learnable positional embeddings,
            shape ``(max_seq_len, hidden_dim)``.
        blocks: Stack of DiT transformer blocks.
        output_norm: Final layer normalization.
        output_proj: Output projection, ``(hidden_dim,) -> (output_dim,)``.
    """

    time_embed: SinusoidalEmbedding
    time_proj: eqx.nn.MLP
    input_proj: eqx.nn.Linear
    pos_embed: Float[Array, "max_seq_len hidden_dim"]
    blocks: list[DiTBlock]
    output_norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        max_seq_len: int,
        time_emb_dim: int,
        cond_dim: int,
        prediction_kind: str,
        activation: Callable,
        max_period: float,
        key: jax.Array,
    ):
        """Initialize a DiT model.

        Args:
            input_dim: Dimensionality of input tokens.
            output_dim: Dimensionality of output tokens.
            hidden_dim: Dimensionality of the hidden representation.
            num_layers: Number of DiT transformer blocks.
            num_heads: Number of attention heads per block.
            head_dim: Dimension of each attention head.
            mlp_dim: Dimensionality of the MLP intermediate layer.
            max_seq_len: Maximum sequence length for positional embeddings.
            time_emb_dim: Dimensionality of the sinusoidal time embedding.
            cond_dim: Dimensionality of the internal conditioning vector.
            prediction_kind: Type of prediction (e.g. ``'eps'``, ``'x_0'``, ``'v'``).
            activation: Elementwise activation function,
                ``(mlp_dim,) -> (mlp_dim,)``.
            max_period: Maximum period for sinusoidal time embedding.
            key: PRNG key for parameter initialization.
        """
        key_tp, key_ip, key_pe, key_op, *block_keys = jax.random.split(
            key, 4 + num_layers
        )

        self.prediction_kinds = prediction_kind
        self.time_embed = SinusoidalEmbedding(
            emb_dim=time_emb_dim, max_period=max_period
        )
        self.time_proj = eqx.nn.MLP(
            in_size=time_emb_dim,
            out_size=cond_dim,
            width_size=hidden_dim,
            depth=2,
            key=key_tp,
            activation=jax.nn.silu,
        )
        self.input_proj = eqx.nn.Linear(input_dim, hidden_dim, key=key_ip)
        self.pos_embed = jax.random.normal(key_pe, (max_seq_len, hidden_dim)) * 0.02
        self.blocks = [
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                activation=activation,
                key=k,
            )
            for k in block_keys
        ]
        self.output_norm = eqx.nn.LayerNorm(hidden_dim)
        self.output_proj = eqx.nn.Linear(hidden_dim, output_dim, key=key_op)

    def _forward_single(
        self,
        x: Float[Array, "seq_len input_dim"],
        t: Float[Array, ""],
    ) -> Prediction:
        """Forward pass for a single sample.

        Args:
            x: Input token sequence, shape ``(seq_len, input_dim)``.
            t: Diffusion time, scalar.

        Returns:
            Prediction for this sample.
        """
        seq_len = x.shape[0]

        # Time conditioning
        t_emb = self.time_embed(t[None])[0]  # (time_emb_dim)
        cond = self.time_proj(t_emb)  # (cond_dim,)

        # Input projection + positional embedding
        h = jax.vmap(self.input_proj)(x)  # (seq_len, hidden_dim)
        h = h + self.pos_embed[:seq_len]

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cond)

        # Output projection
        h = jax.vmap(self.output_norm)(h)  # (seq_len, hidden_dim)
        out = jax.vmap(self.output_proj)(h)  # (seq_len, output_dim)
        return Prediction(value=out, kind=self.prediction_kinds)

    def __call__(
        self,
        x: DataTree,
        s: TimeTree | None,
        t: TimeTree,
        cond: ConditioningTree | None,
        aux: AuxInfoTree | None,
    ) -> PredictionTree:
        """Forward pass over a batch of samples.

        Args:
            x: Noisy token sequences, shape ``(batch, seq_len, input_dim)``.
            s: Unused (pass ``None``).
            t: Diffusion times, shape ``(batch,)``.
            cond: Unused (pass ``None``).
            aux: Unused (pass ``None``).

        Returns:
            Predictions, shape ``(batch, seq_len, output_dim)``.
        """
        return jax.vmap(self._forward_single)(x, t)


class ConditionedDiT(DiffusionModel):
    """Diffusion Transformer with external conditioning for sequence data.

    Same architecture as DiT but accepts an additional external conditioning
    signal that is additively combined with the time embedding, following the
    DiT paper.

    Attributes:
        time_embed: Sinusoidal time embedding module.
        time_proj: Projects time embedding to conditioning dimension,
            ``(time_emb_dim,) -> (cond_dim,)``.
        cond_proj: Projects external conditioning to conditioning dimension,
            ``(external_cond_dim,) -> (cond_dim,)``.
        input_proj: Projects input tokens to hidden dimension,
            ``(input_dim,) -> (hidden_dim,)``.
        pos_embed: Learnable positional embeddings,
            shape ``(max_seq_len, hidden_dim)``.
        blocks: Stack of DiT transformer blocks.
        output_norm: Final layer normalization.
        output_proj: Zero-initialized output projection,
            ``(hidden_dim,) -> (output_dim,)``.
    """

    time_embed: SinusoidalEmbedding = eqx.field(static=False)
    time_proj: eqx.nn.MLP = eqx.field(static=False)
    cond_proj: eqx.nn.MLP = eqx.field(static=False)
    input_proj: eqx.nn.Linear = eqx.field(static=False)
    pos_embed: Float[Array, "max_seq_len hidden_dim"] = eqx.field(static=False)
    blocks: list[DiTBlock] = eqx.field(static=False)
    output_norm: eqx.nn.LayerNorm = eqx.field(static=False)
    output_proj: eqx.nn.Linear = eqx.field(static=False)
    null_cond: Float[Array, " cond_emb_dim"] = eqx.field(static=False)

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        max_seq_len: int,
        time_emb_dim: int,
        cond_emb_dim: int,
        cond_dim: int,
        prediction_kind: str,
        activation: Callable,
        max_period: float,
        key: jax.Array,
    ):
        """Initialize a Conditioned DiT model.

        Args:
            input_dim: Dimensionality of input tokens.
            output_dim: Dimensionality of output tokens.
            hidden_dim: Dimensionality of the hidden representation.
            num_layers: Number of DiT transformer blocks.
            num_heads: Number of attention heads per block.
            head_dim: Dimension of each attention head.
            mlp_dim: Dimensionality of the MLP intermediate layer.
            max_seq_len: Maximum sequence length for positional embeddings.
            time_emb_dim: Dimensionality of the sinusoidal time embedding.
            cond_emb_dim: Dimensionality of the external conditioning input.
            cond_dim: Dimensionality of the internal conditioning vector.
            prediction_kind: Type of prediction (e.g. ``'eps'``, ``'x_0'``, ``'v'``).
            activation: Elementwise activation function,
                ``(mlp_dim,) -> (mlp_dim,)``.
            max_period: Maximum period for sinusoidal time embedding.
            key: PRNG key for parameter initialization.
        """
        key_tp, key_cp, key_ip, key_pe, key_op, key_nc, *block_keys = jax.random.split(
            key, 6 + num_layers
        )

        self.prediction_kinds = prediction_kind
        self.time_embed = SinusoidalEmbedding(
            emb_dim=time_emb_dim, max_period=max_period
        )
        self.time_proj = eqx.nn.MLP(
            in_size=time_emb_dim,
            out_size=cond_dim,
            width_size=hidden_dim,
            depth=2,
            key=key_tp,
            activation=jax.nn.silu,
        )
        self.cond_proj = eqx.nn.MLP(
            in_size=cond_emb_dim,
            out_size=cond_dim,
            width_size=hidden_dim,
            depth=2,
            key=key_cp,
            activation=jax.nn.silu,
        )
        self.input_proj = eqx.nn.Linear(input_dim, hidden_dim, key=key_ip)
        self.pos_embed = jax.random.normal(key_pe, (max_seq_len, hidden_dim)) * 0.02
        self.blocks = [
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_dim=mlp_dim,
                cond_dim=cond_dim,
                activation=activation,
                key=k,
            )
            for k in block_keys
        ]
        self.output_norm = eqx.nn.LayerNorm(hidden_dim)
        self.output_proj = eqx.nn.Linear(hidden_dim, output_dim, key=key_op)
        self.null_cond = jax.random.normal(key_nc, (cond_emb_dim,))

    def _forward_single(
        self,
        x: Float[Array, "seq_len input_dim"],
        t: Float[Array, ""],
        cond: Float[Array, " cond_emb_dim"],
    ) -> Prediction:
        """Forward pass for a single sample.

        Args:
            x: Input token sequence, shape ``(seq_len, input_dim)``.
            t: Diffusion time, scalar.
            cond: External conditioning vector, shape ``(cond_emb_dim,)``.

        Returns:
            Prediction for this sample.
        """
        seq_len = x.shape[0]

        # Time + external conditioning (additive combination)
        t_emb = self.time_embed(t[None])[0]  # (time_emb_dim, )
        cond_vec = self.time_proj(t_emb) + self.cond_proj(cond)  # (cond_dim, )

        # Input projection + positional embedding
        h = jax.vmap(self.input_proj)(x)  # (seq_len, hidden_dim)
        h = h + self.pos_embed[:seq_len]

        # Transformer blocks
        for block in self.blocks:
            h = block(h, cond_vec)

        # Output projection
        h = jax.vmap(self.output_norm)(h)  # (seq_len, hidden_dim)
        out = jax.vmap(self.output_proj)(h)  # (seq_len, output_dim)
        return Prediction(value=out, kind=self.prediction_kinds)

    def __call__(
        self,
        x: DataTree,
        s: TimeTree | None,
        t: TimeTree,
        cond: ConditioningTree | None,
        aux: AuxInfoTree | None,
    ) -> PredictionTree:
        """Forward pass over a batch of samples.

        Args:
            x: Noisy token sequences, shape ``(batch, seq_len, input_dim)``.
            s: Unused (pass ``None``).
            t: Diffusion times, shape ``(batch,)``.
            cond: External conditioning, shape ``(batch, external_cond_dim)``.
            aux: Unused (pass ``None``).

        Returns:
            Predictions, shape ``(batch, seq_len, output_dim)``.
        """

        if cond is None:
            batch_size, _, _ = x.shape
            cond_emb_dim = self.cond_proj.in_size
            cond = jnp.broadcast_to(self.null_cond, (batch_size, cond_emb_dim))
        return jax.vmap(self._forward_single)(x, t, cond)
