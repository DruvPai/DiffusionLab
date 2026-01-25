"""Simplicial diffusion processes for categorical data (Dirichlet corruption).

This module provides diffusion processes that corrupt discrete
(integer-valued) token sequences to soft categorical distributions via
Dirichlet noise.  The forward process is:

    x_t ~ Dir(τ · (h(t) · δ(x_0) + π))

where:

- δ(x_0) is the one-hot encoding of x_0
- π is the invariant probability distribution
- τ is a temperature parameter
- h(t) = (1 − β(t)) / max(β(t), ε), with β(t) ∈ [0, 1] being the
  cumulative corruption schedule

At t = 0, β(0) = 0 so h(0) → +∞ and the Dirichlet concentrates at
δ(x_0) (clean data).
At t = 1, β(1) = 1 so h(1) = 0 and the Dirichlet is Dir(τπ)
(invariant distribution).

Two invariant distributions are supported via factory class methods:

- **Mask (absorbing)**: π = (0, …, 0, 1) with K + 1 categories
  (K regular + 1 mask).  At t = 1 all mass is on the mask category.
- **Uniform**: π = (1/K, …, 1/K) with K categories.
  At t = 1 the distribution is a symmetric Dirichlet.

The corrupted data ``x_t`` is a **log-probability vector** (a sample from
the Dirichlet in log-space), providing a "soft" corruption that preserves
differentiability.

Concrete schedule variants:

- SimplicialLinear: β(t) = t
- SimplicialCosine: β(t) = 1 − cos(πt/2)

"""

from abc import abstractmethod

import jax
from jax import numpy as jnp

from diffusionlab.processes.base import CorruptionProcess
from diffusionlab.typing import (
    AuxInfo,
    Coefficients,
    DiscreteData,
    Prediction,
    PRNGKey,
    ProbabilisticData,
    Scalar,
    Time,
)
from diffusionlab.utils.tree_ops import bcast_right

# ===========================================================================
# Abstract base
# ===========================================================================


class SimplicialCorruptionProcess(CorruptionProcess):
    """Abstract base class for simplicial (Dirichlet) corruption processes.

    Operates on integer-valued token sequences
    x ∈ {0, …, vocab_size − 1}^seq_shape.  The forward process corrupts
    clean discrete tokens to log-probability vectors (log-Dirichlet
    samples) over ``vocab_size`` categories.

    The corruption schedule is parameterised by a cumulative corruption
    probability β(t) ∈ [0, 1] with β(0) = 0 and β(1) = 1, exactly as
    in the discrete process.  The derived concentration ratio is
    h(t) = (1 − β(t)) / max(β(t), ε).

    **Padding / unused tokens.**  If ``padding_token`` is not ``None``,
    positions where ``x_0 == padding_token`` are preserved in the
    output (the log-probability entries are replaced by the padding
    value).  Note that ``sample_from_terminal`` has no
    knowledge of which positions are padding — the caller must restore
    padding positions after sampling.

    Subclasses must implement the corruption schedule:

    - ``beta(t)``, ``beta_prime(t)``

    Available prediction kinds for ``convert_prediction``:

    - ``"logits_x0"``: unnormalised log-probabilities (logits) over
      categories, shape ``(batch, *seq_shape, vocab_size)``
    - ``"x_0"``: probability distribution over categories,
      shape ``(batch, *seq_shape, vocab_size)``

    Attributes:
        seq_shape: Shape of a single sequence (excluding batch dimension).
        invariant_probs: Invariant probability distribution π as a JAX
            array of shape ``(vocab_size,)``.
        temperature: Temperature τ controlling Dirichlet sharpness.
        padding_token: Token value for unused / padding positions.
            If ``None`` (default), no padding handling is performed.
        safety_epsilon: Small constant ε used as a floor for the denominator of
            h(t) to avoid division by zero.

    """

    def __init__(
        self,
        seq_shape: tuple[int, ...],
        invariant_probs: jax.Array,
        temperature: float = 1.0,
        padding_token: int | None = None,
        safety_epsilon: float = 1e-6,
    ):
        """Initialize the simplicial corruption process.

        Args:
            seq_shape: Shape of a single sequence (excluding batch dim).
            invariant_probs: Invariant probability distribution π as a
                1-D JAX array of shape ``(vocab_size,)``.
                For mask: ``[0, …, 0, 1]`` with K + 1 entries.
                For uniform: ``[1/K, …, 1/K]`` with K entries.
            temperature: Temperature τ for Dirichlet concentration.
                Default 1.0.
            padding_token: Optional token value for padding positions.
                Must lie outside {0, …, vocab_size − 1}.
            safety_epsilon: Small constant to prevent division by zero
                in h(t) = (1 − β(t)) / max(β(t), ε).  Default 1e-6.
        """
        self.seq_shape: tuple[int, ...] = seq_shape
        self.invariant_probs: jax.Array = invariant_probs  # (vocab_size,)
        self.temperature: float = temperature
        self.padding_token: int | None = padding_token
        self.safety_epsilon: float = safety_epsilon

        V = self.vocab_size
        if padding_token is not None and 0 <= padding_token < V:
            raise ValueError(
                f"padding_token must be outside {{0, …, {V - 1}}}. Got: {padding_token}"
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of Dirichlet categories (= ``invariant_probs.shape[0]``).

        Equals K for uniform, K + 1 for masking (where K is the number
        of regular token classes).
        """
        return self.invariant_probs.shape[0]

    @property
    def is_masking(self) -> bool:
        """Whether this is a masking process (π = (0, …, 0, 1))."""
        expected = jnp.zeros(self.vocab_size).at[-1].set(1.0)
        return bool(jnp.array_equal(self.invariant_probs, expected))

    # ------------------------------------------------------------------
    # Schedule interface (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def beta(self, t: Time) -> Scalar:
        """Cumulative corruption probability β(t).

        Must satisfy β(0) = 0 and β(1) = 1.

        Args:
            t: Time, shape (batch,).

        Returns:
            β(t), shape (batch,).
        """
        ...

    @abstractmethod
    def beta_prime(self, t: Time) -> Scalar:
        """Time derivative of the cumulative corruption probability β'(t).

        Args:
            t: Time, shape (batch,).

        Returns:
            β'(t), shape (batch,).
        """
        ...

    # ------------------------------------------------------------------
    # Derived schedule functions
    # ------------------------------------------------------------------

    def h(self, t: Time) -> Scalar:
        """Concentration ratio h(t) = max((1 − β(t)) / max(β(t), ε), 0).

        Controls how concentrated the Dirichlet is around x_0.
        h(0) → +∞ (clean), h(1) = 0 (fully corrupted).

        The result is clamped to ≥ 0 for numerical safety: some
        schedules (e.g. cosine) may produce slightly negative values
        at boundary points due to floating-point rounding, which would
        otherwise lead to invalid (negative) Dirichlet concentration
        parameters.

        Args:
            t: Time, shape (batch,).

        Returns:
            h(t), shape (batch,).
        """
        denominator = jnp.maximum(self.beta(t), self.safety_epsilon)
        return jnp.maximum((1.0 - self.beta(t)) / denominator, 0.0)

    # ------------------------------------------------------------------
    # Dirichlet sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_log_dirichlet(
        key: PRNGKey, alpha: ProbabilisticData
    ) -> ProbabilisticData:
        """Sample from Dirichlet(α) in log-space.

        Uses Gamma sampling with log-normalisation for numerical
        stability.  Categories with α ≈ 0 receive very negative
        log-probabilities (effectively probability 0).

        Args:
            key: PRNG key.
            alpha: Concentration parameters, shape (…, M).

        Returns:
            Log-probability vector, shape (…, M).
        """
        gammas = jax.random.gamma(key, alpha)  # (…, M)
        log_gammas = jnp.log(jnp.maximum(gammas, 1e-30))  # avoid log(0)
        return log_gammas - jax.nn.logsumexp(log_gammas, axis=-1, keepdims=True)

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def _forward_core(
        self,
        key: PRNGKey,
        x: DiscreteData,
        t: Time,
    ) -> tuple[ProbabilisticData, AuxInfo]:
        """Core forward pass: compute Dirichlet corruption.

        Args:
            key: PRNG key for Dirichlet sampling.
            x: Clean tokens x_0, shape (batch, *seq_shape).
            t: Time, shape (batch,).

        Returns:
            Tuple of (x_t, aux_info).

            - x_t: Log-Dirichlet sample, shape
              (batch, *seq_shape, vocab_size).
            - aux_info: Dict containing ``x`` (original tokens),
              ``t`` (time), ``padding_mask``, and schedule info
              (beta, beta_prime, h).
        """
        V = self.vocab_size

        # One-hot encode x_0: (batch, *seq_shape, V)
        x_oh = jax.nn.one_hot(x, V)

        # h(t): (batch,) → (batch, 1, …, 1) for broadcasting
        h_t = self.h(t)  # (batch,)
        h_t_bc = bcast_right(h_t, x.ndim)  # (batch, 1, …)

        # Dirichlet concentration parameters:
        # τ · (h(t) · δ(x_0) + π), shape (batch, *seq_shape, V)
        dirichlet_params = self.temperature * (
            h_t_bc[..., None] * x_oh + self.invariant_probs
        )

        # Sample x_t in log-space: (batch, *seq_shape, V)
        x_t = self._sample_log_dirichlet(key, dirichlet_params)

        # Handle padding: preserve padding_token at padding positions
        if self.padding_token is not None:
            padding_mask = x == self.padding_token  # (batch, *seq_shape)
            x_t = jnp.where(
                padding_mask[..., None],
                float(self.padding_token),
                x_t,
            )
        else:
            padding_mask = jnp.zeros(x.shape, dtype=jnp.bool_)

        coefficients = self.get_coefficients(t)
        aux: AuxInfo = {
            "x": x,
            "t": t,
            "padding_mask": padding_mask,
        } | coefficients

        return x_t, aux

    # ------------------------------------------------------------------
    # CorruptionProcess interface
    # ------------------------------------------------------------------

    def forward(
        self,
        key: PRNGKey,
        x: DiscreteData,
        t: Time,
    ) -> tuple[ProbabilisticData, AuxInfo]:
        """Corrupt tokens x at time t via Dirichlet noise.

        Each non-padding position is independently corrupted by sampling
        from Dir(τ · (h(t) · δ(x_0) + π)) in log-space.

        Args:
            key: PRNG key.
            x: Clean tokens x_0, shape (batch, *seq_shape).
            t: Time, shape (batch,).

        Returns:
            Tuple of (x_t, aux_info).

            - x_t: Log-probability vector,
              shape (batch, *seq_shape, vocab_size).
            - aux_info: Dict with ``x``, ``t``,
              ``padding_mask``, and schedule info
              (``beta``, ``beta_prime``, ``h``).
        """
        return self._forward_core(key, x, t)

    def forward_multiple(
        self,
        key: PRNGKey,
        x: DiscreteData,
        ts: list[Time],
    ) -> list[tuple[ProbabilisticData, AuxInfo]]:
        """Corrupt tokens x at each time with independent Dirichlet noise.

        Unlike the discrete process (which shares uniform draws for
        monotonic corruption), simplicial corruption uses fresh
        randomness for each time step because the Dirichlet
        concentration parameters change with time.

        Args:
            key: PRNG key.
            x: Clean tokens x_0, shape (batch, *seq_shape).
            ts: List of times.

        Returns:
            List of (x_t, aux_info) pairs, one per time.
        """
        n = len(ts)
        keys = jax.random.split(key, n)
        stacked_ts = jnp.stack(ts)  # (num_times, batch)

        stacked_x_t, stacked_aux = jax.vmap(
            lambda ki, ti: self._forward_core(ki, x, ti)
        )(keys, stacked_ts)

        stacked = (stacked_x_t, stacked_aux)
        return [jax.tree_util.tree_map(lambda v: v[i], stacked) for i in range(n)]

    def sample_from_terminal(
        self,
        key: PRNGKey,
        batch_size: int,
    ) -> tuple[ProbabilisticData, AuxInfo]:
        """Sample from the terminal distribution at t = 1.

        At t = 1, h(1) = 0, so x_1 ~ Dir(τ · π).

        - For masking: concentrates all mass on the mask category.
        - For uniform: symmetric Dirichlet Dir(τ/K, …, τ/K).

        Note: this method has no knowledge of padding positions.  The
        caller must restore padding tokens after sampling if needed.

        Args:
            key: PRNG key.
            batch_size: Number of samples.

        Returns:
            Tuple of (x_1, aux_info).

            - x_1: Log-probability vector,
              shape (batch_size, *seq_shape, vocab_size).
        """
        shape = (batch_size, *self.seq_shape)  # (batch, *seq_shape)
        V = self.vocab_size

        # Dir(τ · π): broadcast to (batch, *seq_shape, V)
        invariant_params = self.temperature * self.invariant_probs
        invariant_params_bc = jnp.broadcast_to(invariant_params, (*shape, V))

        x_1 = self._sample_log_dirichlet(key, invariant_params_bc)
        t = jnp.ones((batch_size,))
        return x_1, {"t": t} | self.get_coefficients(t)

    def convert_prediction(
        self,
        prediction: Prediction,
        x_t: ProbabilisticData,
        s: Time | None,
        t: Time,
        to_kind: str,
    ) -> Prediction:
        """Convert a model prediction to the requested kind.

        Supported conversions:

        - ``"logits_x0"`` → ``"x_0"``: softmax normalisation.
        - ``"x_0"`` → ``"logits_x0"``: log of probabilities.

        Args:
            prediction: The prediction to convert.
            x_t: Corrupted log-probabilities at time t,
                shape (batch, *seq_shape, vocab_size).
            s: Optional earlier time (unused for simplicial).
            t: Current time, shape (batch,).
            to_kind: Target prediction kind.

        Returns:
            Prediction of the requested kind.

        Raises:
            NotImplementedError: If the requested conversion is not
                supported.
        """
        if prediction.kind == to_kind:
            return prediction

        match (prediction.kind, to_kind):
            # --- From logits_x0 ---
            case ("logits_x0", "x_0"):
                probs = jax.nn.softmax(prediction.value, axis=-1)
                return Prediction(value=probs, kind="x_0")

            # --- From x_0 ---
            case ("x_0", "logits_x0"):
                logits = jnp.log(jnp.maximum(prediction.value, 1e-20))
                return Prediction(value=logits, kind="logits_x0")

            case _:
                raise NotImplementedError(
                    f"Conversion from {prediction.kind!r} to {to_kind!r} "
                    f"is not implemented for simplicial processes."
                )

    def get_coefficients(self, t: Time) -> Coefficients:
        """Get corruption coefficients for time t.

        Args:
            t: Time, shape (batch,).

        Returns:
            Corruption coefficients for the given time; contains the following keys:
            - ``"beta"``: Cumulative corruption probability β(t).
            - ``"beta_prime"``: Time derivative of the cumulative corruption probability β'(t).
            - ``"h"``: Concentration ratio h(t).
        """
        return {
            "beta": self.beta(t),
            "beta_prime": self.beta_prime(t),
            "h": self.h(t),
        }

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def mask_process(
        cls,
        unmasked_vocab_size: int,
        seq_shape: tuple[int, ...],
        temperature: float = 1.0,
        padding_token: int | None = None,
        safety_epsilon: float = 1e-6,
    ) -> "SimplicialCorruptionProcess":
        """Create a mask simplicial process with π = (0, …, 0, 1).

        The invariant distribution puts all mass on a dedicated mask
        category (index K), resulting in K + 1 total categories
        (``vocab_size = unmasked_vocab_size + 1``).

        Args:
            unmasked_vocab_size: Number of regular token classes K.  Must be ≥ 1.
            seq_shape: Shape of a single sequence.
            temperature: Temperature τ for Dirichlet concentration.
                Default 1.0.
            padding_token: Optional padding token index.
            safety_epsilon: Numerical stability constant ε.

        Returns:
            A SimplicialCorruptionProcess with mask invariant.

        Raises:
            ValueError: If unmasked_vocab_size < 1.
        """
        if unmasked_vocab_size < 1:
            raise ValueError(
                f"unmasked_vocab_size must be positive. Got: {unmasked_vocab_size}"
            )
        invariant_probs = jnp.zeros(unmasked_vocab_size + 1).at[-1].set(1.0)
        return cls(
            seq_shape,
            invariant_probs,
            temperature=temperature,
            padding_token=padding_token,
            safety_epsilon=safety_epsilon,
        )

    @classmethod
    def uniform_process(
        cls,
        unmasked_vocab_size: int,
        seq_shape: tuple[int, ...],
        temperature: float = 1.0,
        padding_token: int | None = None,
        safety_epsilon: float = 1e-6,
    ) -> "SimplicialCorruptionProcess":
        """Create a uniform simplicial process with π = (1/K, …, 1/K).

        The invariant distribution is uniform over K categories
        (``vocab_size = unmasked_vocab_size``).

        Args:
            unmasked_vocab_size: Number of token classes K.  Must be ≥ 1.
            seq_shape: Shape of a single sequence.
            temperature: Temperature τ for Dirichlet concentration.
                Default 1.0.
            padding_token: Optional padding token index.
            safety_epsilon: Numerical stability constant ε.

        Returns:
            A SimplicialCorruptionProcess with uniform invariant.

        Raises:
            ValueError: If unmasked_vocab_size < 1.
        """
        if unmasked_vocab_size < 1:
            raise ValueError(
                f"unmasked_vocab_size must be positive. Got: {unmasked_vocab_size}"
            )
        invariant_probs = jnp.full(unmasked_vocab_size, 1.0 / unmasked_vocab_size)
        return cls(
            seq_shape,
            invariant_probs,
            temperature=temperature,
            padding_token=padding_token,
            safety_epsilon=safety_epsilon,
        )


# ===========================================================================
# Concrete schedules
# ===========================================================================


class SimplicialLinear(SimplicialCorruptionProcess):
    """Simplicial diffusion with linear corruption schedule: β(t) = t.

    The corruption probability increases linearly from 0 to 1.  The
    concentration ratio h(t) = (1 − t) / (t + ε), which decreases
    monotonically and reaches 0 at t = 1.

    Attributes:
        seq_shape: Shape of a single sequence (excluding batch dim).
        invariant_probs: Invariant distribution π.
        temperature: Temperature τ.
        padding_token: Optional padding token index.
        safety_epsilon: Numerical stability constant ε.
    """

    def beta(self, t: Time) -> Scalar:
        """β(t) = t.

        Args:
            t: Time, shape (batch,).

        Returns:
            β(t), shape (batch,).
        """
        return t

    def beta_prime(self, t: Time) -> Scalar:
        """β'(t) = 1.

        Args:
            t: Time, shape (batch,).

        Returns:
            β'(t), shape (batch,).
        """
        return jnp.ones_like(t)


class SimplicialCosine(SimplicialCorruptionProcess):
    """Simplicial diffusion with cosine corruption schedule: β(t) = 1 − cos(πt/2).

    Starts slow (gentle corruption near t = 0) and accelerates toward
    t = 1, matching the popular cosine schedule.

    Attributes:
        seq_shape: Shape of a single sequence (excluding batch dim).
        invariant_probs: Invariant distribution π.
        temperature: Temperature τ.
        padding_token: Optional padding token index.
        safety_epsilon: Numerical stability constant ε.
    """

    def beta(self, t: Time) -> Scalar:
        """β(t) = 1 − cos(πt/2).

        Args:
            t: Time, shape (batch,).

        Returns:
            β(t), shape (batch,).
        """
        return 1.0 - jnp.cos(jnp.pi / 2 * t)

    def beta_prime(self, t: Time) -> Scalar:
        """β'(t) = (π/2) sin(πt/2).

        Args:
            t: Time, shape (batch,).

        Returns:
            β'(t), shape (batch,).
        """
        return jnp.pi / 2 * jnp.sin(jnp.pi / 2 * t)
