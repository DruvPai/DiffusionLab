"""FiLM-conditioned MLP for diffusion models on flat vector data."""

from typing import Callable

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from diffusionlab.models.base import DiffusionModel
from diffusionlab.models.components import FiLM, SinusoidalEmbedding
from diffusionlab.typing import (
    AuxInfo,
    Conditioning,
    ContinuousData,
    Prediction,
    PRNGKey,
    Time,
)


class MLPBlock(eqx.Module):
    """MLP block with FiLM conditioning.

    Attributes:
        film: FiLM module.
        mlp: MLP module, ``(hidden_dim,) -> (hidden_dim,)``.
    """

    film: FiLM = eqx.field(static=False)
    mlp: eqx.nn.MLP = eqx.field(static=False)

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        activation: Callable,
        key: PRNGKey,
    ):
        """Initialize an MLP block.

        Args:
            hidden_dim: Dimensionality of the hidden representation.
            cond_dim: Dimensionality of the conditioning vector.
            activation: Elementwise activation function,
                ``(hidden_dim,) -> (hidden_dim,)``.
            key: PRNG key for parameter initialization.
        """
        key_film, key_mlp = jax.random.split(key, 2)
        self.film = FiLM(hidden_dim=hidden_dim, cond_dim=cond_dim, key=key_film)
        self.mlp = eqx.nn.MLP(
            in_size=hidden_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=2,
            key=key_mlp,
            activation=activation,
        )

    def __call__(
        self, h: Float[Array, " hidden_dim"], cond: Float[Array, " cond_dim"]
    ) -> Float[Array, " hidden_dim"]:
        """Forward pass for a single sample.

        Args:
            h: Input data, shape ``(hidden_dim,)``.
            cond: Conditioning vector, shape ``(cond_dim,)``.
        """
        h = self.mlp(self.film(h, cond))
        return h


class DiffusionMLP(DiffusionModel):
    """FiLM-conditioned MLP for diffusion on flat vector data.

    Processes data of shape ``(batch, data_dim)`` by embedding the diffusion
    time via sinusoidal embeddings, then applying a stack of FiLM-conditioned
    residual layers.

    Attributes:
        time_embed: Sinusoidal time embedding module.
        time_proj: Projects time embedding to conditioning dimension,
            ``(time_emb_dim,) -> (cond_dim,)``.
        input_proj: Projects input data to hidden dimension,
            ``(data_dim,) -> (hidden_dim,)``.
        layers: Stack of FiLM-conditioned residual layers.
        output_norm: Final layer normalization.
        output_proj: Projects hidden state to output,
            ``(hidden_dim,) -> (data_dim,)``.
    """

    time_embed: SinusoidalEmbedding = eqx.field(static=False)
    time_proj: eqx.nn.MLP = eqx.field(static=False)
    input_proj: eqx.nn.Linear = eqx.field(static=False)
    layers: list[MLPBlock] = eqx.field(static=False)
    output_norm: eqx.nn.LayerNorm = eqx.field(static=False)
    output_proj: eqx.nn.Linear = eqx.field(static=False)

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int,
        cond_dim: int,
        prediction_kind: str,
        activation: Callable,
        max_period: float,
        key: PRNGKey,
    ):
        """Initialize a FiLM MLP.

        Args:
            data_dim: Dimensionality of the input/output data.
            hidden_dim: Dimensionality of the hidden layers.
            num_layers: Number of FiLM residual layers.
            time_emb_dim: Dimensionality of the sinusoidal time embedding.
            cond_dim: Dimensionality of the internal conditioning vector.
            prediction_kind: Type of prediction (e.g. ``'eps'``, ``'x_0'``, ``'v'``).
            activation: Elementwise activation function,
                ``(hidden_dim,) -> (hidden_dim,)``.
            max_period: Maximum period for sinusoidal time embedding.
            key: PRNG key for parameter initialization.
        """
        key_tp, key_ip, key_op, *layer_keys = jax.random.split(key, 3 + num_layers)

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
        self.input_proj = eqx.nn.Linear(data_dim, hidden_dim, key=key_ip)
        self.layers = [
            MLPBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                activation=activation,
                key=k,
            )
            for k in layer_keys
        ]
        self.output_norm = eqx.nn.LayerNorm(hidden_dim)
        self.output_proj = eqx.nn.Linear(hidden_dim, data_dim, key=key_op)

    def _forward_single(
        self,
        x: Float[Array, " data_dim"],
        t: Float[Array, ""],
    ) -> Prediction:
        """Forward pass for a single sample.

        Args:
            x: Input data, shape ``(data_dim,)``.
            t: Diffusion time, scalar.

        Returns:
            Prediction for this sample.
        """
        # Time conditioning
        t_emb = self.time_embed(t[None])[0]  # (time_emb_dim,)
        cond = self.time_proj(t_emb)  # (cond_dim,)

        # Input projection
        h = self.input_proj(x)  # (hidden_dim,)

        # FiLM residual layers
        for layer in self.layers:
            h = h + layer(h, cond)

        # Output projection
        h = self.output_norm(h)  # (hidden_dim,)
        out = self.output_proj(h)  # (data_dim,)
        return Prediction(value=out, kind=self.prediction_kinds)

    def __call__(
        self,
        x: ContinuousData,
        s: Time | None,
        t: Time,
        cond: Conditioning | None,
        aux: AuxInfo | None,
    ) -> Prediction:
        """Forward pass over a batch of samples.

        Args:
            x: Noisy data, shape ``(batch, data_dim)``.
            s: Unused (pass ``None``).
            t: Diffusion times, shape ``(batch,)``.
            cond: Unused (pass ``None``).
            aux: Unused (pass ``None``).

        Returns:
            Predictions, shape ``(batch, data_dim)``.
        """

        assert cond is None, "Conditioning is not supported for MLP"
        return jax.vmap(self._forward_single)(x, t)


class ConditionedDiffusionMLP(DiffusionModel):
    """FiLM-conditioned MLP for diffusion on flat vector data.

    Processes data of shape ``(batch, data_dim)`` by embedding the diffusion
    time via sinusoidal embeddings, then applying a stack of FiLM-conditioned
    residual layers. Adds a variable to condition on.

    Attributes:
        time_embed: Sinusoidal time embedding module.
        time_proj: Projects time embedding to conditioning dimension,
            ``(time_emb_dim,) -> (cond_dim,)``.
        cond_proj: Projects conditioning variable to conditioning dimension,
            ``(cond_emb_dim,) -> (cond_dim,)``.
        input_proj: Projects input data to hidden dimension,
            ``(data_dim,) -> (hidden_dim,)``.
        layers: Stack of FiLM-conditioned residual layers.
        output_norm: Final layer normalization.
        output_proj: Projects hidden state to output,
            ``(hidden_dim,) -> (data_dim,)``.
    """

    time_embed: SinusoidalEmbedding = eqx.field(static=False)
    time_proj: eqx.nn.MLP = eqx.field(static=False)
    cond_proj: eqx.nn.MLP = eqx.field(static=False)
    input_proj: eqx.nn.Linear = eqx.field(static=False)
    layers: list[MLPBlock] = eqx.field(static=False)
    output_norm: eqx.nn.LayerNorm = eqx.field(static=False)
    output_proj: eqx.nn.Linear = eqx.field(static=False)
    null_cond: Float[Array, " cond_emb_dim"] = eqx.field(static=False)

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        num_layers: int,
        time_emb_dim: int,
        cond_emb_dim: int,
        cond_dim: int,
        prediction_kind: str,
        activation: Callable,
        max_period: float,
        key: PRNGKey,
    ):
        """Initialize a FiLM MLP.

        Args:
            data_dim: Dimensionality of the input/output data.
            hidden_dim: Dimensionality of the hidden layers.
            num_layers: Number of FiLM residual layers.
            time_emb_dim: Dimensionality of the sinusoidal time embedding.
            cond_emb_dim: Dimensionality of the conditioning variable before projection into cond_dim.
            cond_dim: Dimensionality of the internal conditioning vector.
            prediction_kind: Type of prediction (e.g. ``'eps'``, ``'x_0'``, ``'v'``).
            activation: Elementwise activation function,
                ``(hidden_dim,) -> (hidden_dim,)``.
            max_period: Maximum period for sinusoidal time embedding.
            key: PRNG key for parameter initialization.
        """
        key_tp, key_ip, key_cp, key_op, key_nc, *layer_keys = jax.random.split(
            key, 5 + num_layers
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
        self.input_proj = eqx.nn.Linear(
            in_features=data_dim,
            out_features=hidden_dim,
            key=key_ip,
        )
        self.layers = [
            MLPBlock(
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                activation=activation,
                key=k,
            )
            for k in layer_keys
        ]
        self.output_norm = eqx.nn.LayerNorm(hidden_dim)
        self.output_proj = eqx.nn.Linear(hidden_dim, data_dim, key=key_op)
        self.null_cond = jax.random.normal(key_nc, (cond_emb_dim,))

    def _forward_single(
        self,
        x: Float[Array, " data_dim"],
        t: Float[Array, ""],
        cond: Float[Array, " initial_cond_dim"],
    ) -> Prediction:
        """Forward pass for a single sample.

        Args:
            x: Input data, shape ``(data_dim,)``.
            t: Diffusion time, scalar.
            cond: Conditioning variable, shape ``(initial_cond_dim,)``.

        Returns:
            Prediction for this sample.
        """
        # Time conditioning
        t_emb = self.time_embed(t[None])[0]  # (time_emb_dim)
        cond = self.time_proj(t_emb) + self.cond_proj(cond)  # (cond_dim,)

        # Input projection
        h = self.input_proj(x)  # (hidden_dim,)

        # FiLM residual layers
        for layer in self.layers:
            h = h + layer(h, cond)

        # Output projection
        h = self.output_norm(h)  # (hidden_dim,)
        out = self.output_proj(h)  # (data_dim,)
        return Prediction(value=out, kind=self.prediction_kinds)

    def __call__(
        self,
        x: ContinuousData,
        s: Time | None,
        t: Time,
        cond: Conditioning | None,
        aux: AuxInfo | None,
    ) -> Prediction:
        """Forward pass over a batch of samples.

        Args:
            x: Noisy data, shape ``(batch, data_dim)``.
            s: Unused (pass ``None``).
            t: Diffusion times, shape ``(batch,)``.
            cond: Conditioning variable, shape ``(batch, cond_emb_dim)``.
            aux: Unused (pass ``None``).

        Returns:
            Predictions, shape ``(batch, data_dim)``.
        """

        if cond is None:
            batch_size, _ = x.shape
            cond_emb_dim = self.cond_proj.in_size
            cond = jnp.broadcast_to(self.null_cond, (batch_size, cond_emb_dim))
        return jax.vmap(self._forward_single)(x, t, cond)
