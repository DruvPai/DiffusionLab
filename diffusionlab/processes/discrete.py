"""Discrete diffusion processes for categorical data.

This module provides diffusion processes that operate on discrete
(integer-valued) token sequences. The forward process corrupts tokens
according to a cumulative corruption schedule β(t) ∈ [0, 1], t ∈ [0, 1].

Two corruption mechanisms are supported:

- **Mask (absorbing)**: tokens are independently replaced by a special
  [MASK] token (index = vocab_size) with probability β(t).
- **Uniform**: tokens are independently replaced by a uniformly random
  token from {0, …, K−1} with probability β(t).

Concrete schedule variants:

- MaskLinear / UniformLinear: β(t) = t
- MaskCosine / UniformCosine: β(t) = 1 − cos(πt/2)
"""

from abc import abstractmethod

import jax
from jax import numpy as jnp

from diffusionlab.processes.base import CorruptionProcess
from diffusionlab.typing import (
    AuxInfo,
    Coefficients,
    ContinuousData,
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


class DiscreteCorruptionProcess(CorruptionProcess):
    """Abstract base class for discrete (categorical) diffusion processes.

    Operates on integer-valued token sequences
    x ∈ {0, …, vocab_size − 1}^seq_shape.  The corruption schedule is
    parameterised by a cumulative corruption probability β(t) ∈ [0, 1]
    with β(0) = 0 and β(1) = 1.

    At each time t, each token position is independently corrupted with
    probability β(t).  The replacement mechanism (mask token vs uniform
    random) is determined by subclasses.

    **Padding / unused tokens.**  If ``padding_token`` is not ``None``,
    positions where ``x_0 == padding_token`` are never corrupted and
    remain equal to ``padding_token`` throughout the forward process.
    Note that ``sample_from_terminal`` has no
    knowledge of which positions are padding — the caller must restore
    padding positions after sampling.

    Subclasses must implement:

    - Schedule: ``beta(t)``, ``beta_prime(t)``
    - Corruption: ``_replacement_tokens``, ``_sample_terminal``,
      ``_denoising_posterior_probs``

    Available prediction kinds for ``convert_prediction``:

    - ``"logits_x0"``: unnormalised log-probabilities (logits) over
      clean data, shape ``(batch, *seq_shape, vocab_size)``
    - ``"x_0"``: probability distribution over clean data,
      shape ``(batch, *seq_shape, vocab_size)``
    - ``"logits_x_s"``: unnormalised log-probabilities (logits) over
      the denoising posterior q(x_s | x_t, x̂_0),
      shape ``(batch, *seq_shape, output_classes)``.  Requires *s*.
    - ``"x_s"``: denoising posterior q(x_s | x_t, x̂_0) (normalised),
      shape ``(batch, *seq_shape, output_classes)``.  Requires *s*.

    Attributes:
        vocab_size: Number of distinct regular token values K.
        seq_shape: Shape of a single sequence (excluding batch dimension).
        padding_token: Token value that marks unused / padding positions.
            If ``None`` (default), no padding handling is performed.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_shape: tuple[int, ...],
        padding_token: int | None = None,
    ):
        """Initialize the discrete corruption process.

        Args:
            vocab_size: Number of distinct regular token values K.
            seq_shape: Shape of a single sequence (excluding batch dim).
            padding_token: Optional token value for padding positions.
                Positions where ``x_0 == padding_token`` are never corrupted.
                Must lie outside {0, …, vocab_size − 1} (and outside the
                mask-token index for mask diffusion).
        """
        self.vocab_size: int = vocab_size
        self.seq_shape: tuple[int, ...] = seq_shape
        self.padding_token: int | None = padding_token

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
    # Corruption mechanism (abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def _replacement_tokens(self, r: DiscreteData) -> DiscreteData:
        """Get tokens used to replace corrupted positions.

        For mask diffusion this returns all-MASK tokens; for uniform
        diffusion this returns the random tokens *r*.

        Args:
            r: Random tokens sampled uniformly from {0, …, K−1},
               shape (batch, *seq_shape).

        Returns:
            Replacement tokens, shape (batch, *seq_shape).
        """
        ...

    @abstractmethod
    def _sample_terminal(self, key: PRNGKey, batch_size: int) -> DiscreteData:
        """Sample from the terminal distribution at t = 1.

        Args:
            key: PRNG key.
            batch_size: Number of samples.

        Returns:
            Terminal samples, shape (batch_size, *seq_shape).
        """
        ...

    @abstractmethod
    def _denoising_posterior_probs(
        self,
        probs_x0: ProbabilisticData,
        x_t: DiscreteData,
        s: Time,
        t: Time,
    ) -> ProbabilisticData:
        """Compute the denoising posterior q(x_s | x_t, x̂_0).

        Padding positions (where ``x_t == padding_token``) are **not**
        handled here — the caller should mask them out via ``padding_mask``.

        Args:
            probs_x0: Predicted probability distribution over clean data x_0,
                shape (batch, *seq_shape, vocab_size).
            x_t: Corrupted tokens at time t, shape (batch, *seq_shape).
            s: Target (earlier) time, shape (batch,).
            t: Current time, shape (batch,).

        Returns:
            Posterior probabilities over x_s,
            shape (batch, *seq_shape, output_classes).
        """
        ...

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def _forward_with_uniform(
        self,
        x_0: DiscreteData,
        u: ContinuousData,
        r: DiscreteData,
        t: Time,
    ) -> tuple[DiscreteData, AuxInfo]:
        """Corrupt tokens deterministically given pre-sampled randomness.

        A token at position *i* is corrupted when ``u[i] < β(t)``,
        unless that position is padding.

        Args:
            x_0: Clean tokens, shape (batch, *seq_shape).
            u: Uniform noise ∈ [0, 1] for corruption decisions,
               shape (batch, *seq_shape).
            r: Random replacement tokens, shape (batch, *seq_shape).
            t: Time, shape (batch,).

        Returns:
            Tuple of (x_t, aux_info).
        """
        beta_t = self.beta(t)  # (batch,)
        ndim = x_0.ndim
        corrupt_probs = bcast_right(beta_t, ndim)  # (batch, 1, ...)
        corrupted_mask = u < corrupt_probs  # (batch, *seq_shape)

        # Padding: never corrupt padding positions
        if self.padding_token is not None:
            padding_mask = x_0 == self.padding_token  # (batch, *seq_shape)
            corrupted_mask = corrupted_mask & ~padding_mask
        else:
            padding_mask = jnp.zeros(x_0.shape, dtype=jnp.bool_)

        replacement = self._replacement_tokens(r)  # (batch, *seq_shape)
        x_t = jnp.where(corrupted_mask, replacement, x_0)
        coefficients = self.get_coefficients(t)
        return x_t, {
            "x": x_0,
            "t": t,
            "corrupted_mask": corrupted_mask,
            "padding_mask": padding_mask,
        } | coefficients

    # ------------------------------------------------------------------
    # CorruptionProcess interface
    # ------------------------------------------------------------------

    def forward(
        self, key: PRNGKey, x: DiscreteData, t: Time
    ) -> tuple[DiscreteData, AuxInfo]:
        """Corrupt tokens x at time t.

        Each non-padding token position is independently corrupted with
        probability β(t).

        Args:
            key: PRNG key.
            x: Clean tokens x_0, shape (batch, *seq_shape).
            t: Time, shape (batch,).

        Returns:
            Tuple of (x_t, aux_info).  aux_info contains:

            - ``x``: original clean tokens
            - ``t``: time
            - ``corrupted_mask``: bool array marking corrupted positions
            - ``padding_mask``: bool array marking padding positions
            - ``beta``, ``beta_prime``: schedule info
        """
        key1, key2 = jax.random.split(key)
        u = jax.random.uniform(key1, x.shape)
        r = jax.random.randint(key2, x.shape, 0, self.vocab_size)
        return self._forward_with_uniform(x, u, r, t)

    def forward_multiple(
        self, key: PRNGKey, x: DiscreteData, ts: list[Time]
    ) -> list[tuple[DiscreteData, AuxInfo]]:
        """Corrupt tokens x at each time with shared randomness.

        Uses the same uniform draws across all times, ensuring monotonic
        corruption: if a position is corrupted at time s it is also
        corrupted at every t > s (because β is non-decreasing).

        Args:
            key: PRNG key.
            x: Clean tokens x_0, shape (batch, *seq_shape).
            ts: List of times.

        Returns:
            List of (x_t, aux_info) pairs, one per time.
        """
        key1, key2 = jax.random.split(key)
        u = jax.random.uniform(key1, x.shape)  # (batch, *seq_shape)
        r = jax.random.randint(key2, x.shape, 0, self.vocab_size)
        stacked_ts = jnp.stack(ts)  # (num_times, batch)
        stacked_x_t, stacked_aux = jax.vmap(
            lambda ti: self._forward_with_uniform(x, u, r, ti)
        )(stacked_ts)
        stacked = (stacked_x_t, stacked_aux)
        return [jax.tree_util.tree_map(lambda v: v[i], stacked) for i in range(len(ts))]

    def sample_from_terminal(
        self, key: PRNGKey, batch_size: int
    ) -> tuple[DiscreteData, AuxInfo]:
        """Sample from the terminal distribution at t = 1.

        For mask diffusion returns all-MASK tokens; for uniform diffusion
        returns uniformly random tokens.

        Note: this method has no knowledge of padding positions.  The
        caller must restore padding tokens after sampling if needed.

        Args:
            key: PRNG key.
            batch_size: Number of samples.

        Returns:
            Tuple of (x_1, aux_info).
        """
        x_1 = self._sample_terminal(key, batch_size)
        t = jnp.ones((batch_size,))
        return x_1, {"t": t} | self.get_coefficients(t)

    def convert_prediction(
        self,
        prediction: Prediction,
        x_t: DiscreteData,
        s: Time | None,
        t: Time,
        to_kind: str,
    ) -> Prediction:
        """Convert a model prediction to the requested kind.

        Supported conversions:

        - ``"logits_x0"`` → ``"x_0"``: softmax normalisation.
        - ``"logits_x0"`` → ``"x_s"``: softmax then denoising posterior.
        - ``"logits_x0"`` → ``"logits_x_s"``: softmax, posterior, then log.
        - ``"x_0"`` → ``"x_s"``: compute denoising posterior
          q(x_s | x_t, p(x_0)).  Requires *s*.
        - ``"x_0"`` → ``"logits_x_s"``: posterior then log.  Requires *s*.
        - ``"logits_x_s"`` → ``"x_s"``: softmax normalisation.
        - ``"x_s"`` → ``"logits_x_s"``: log of probabilities.

        Args:
            prediction: The prediction to convert.
            x_t: Corrupted tokens at time t, shape (batch, *seq_shape).
            s: Optional earlier time for x_s conversion, shape (batch,).
            t: Current time, shape (batch,).
            to_kind: Target prediction kind.

        Returns:
            Prediction of the requested kind.

        Raises:
            NotImplementedError: If the requested conversion is not supported.
        """
        if prediction.kind == to_kind:
            return prediction

        match (prediction.kind, to_kind):
            # --- From logits_x0 ---
            case ("logits_x0", "x_0"):
                probs = jax.nn.softmax(prediction.value, axis=-1)
                return Prediction(value=probs, kind="x_0")
            case ("logits_x0", "x_s"):
                assert s is not None, "Time s is required for x_s conversion."
                probs = jax.nn.softmax(prediction.value, axis=-1)
                posterior = self._denoising_posterior_probs(probs, x_t, s, t)
                return Prediction(value=posterior, kind="x_s")
            case ("logits_x0", "logits_x_s"):
                assert s is not None, "Time s is required for logits_x_s conversion."
                probs = jax.nn.softmax(prediction.value, axis=-1)
                posterior = self._denoising_posterior_probs(probs, x_t, s, t)
                logits = jnp.log(jnp.maximum(posterior, 1e-20))
                return Prediction(value=logits, kind="logits_x_s")

            # --- From x_0 ---
            case ("x_0", "x_s"):
                assert s is not None, "Time s is required for x_s conversion."
                posterior = self._denoising_posterior_probs(prediction.value, x_t, s, t)
                return Prediction(value=posterior, kind="x_s")
            case ("x_0", "logits_x_s"):
                assert s is not None, "Time s is required for logits_x_s conversion."
                posterior = self._denoising_posterior_probs(prediction.value, x_t, s, t)
                logits = jnp.log(jnp.maximum(posterior, 1e-20))
                return Prediction(value=logits, kind="logits_x_s")
            case ("x_0", "logits_x0"):
                logits = jnp.log(jnp.maximum(prediction.value, 1e-20))
                return Prediction(value=logits, kind="logits_x0")

            # --- From logits_x_s ---
            case ("logits_x_s", "x_s"):
                probs = jax.nn.softmax(prediction.value, axis=-1)
                return Prediction(value=probs, kind="x_s")

            # --- From x_s ---
            case ("x_s", "logits_x_s"):
                logits = jnp.log(jnp.maximum(prediction.value, 1e-20))
                return Prediction(value=logits, kind="logits_x_s")

            case _:
                raise NotImplementedError(
                    f"Conversion from {prediction.kind!r} to {to_kind!r} "
                    f"is not implemented for discrete processes."
                )

    def get_coefficients(self, t: Time) -> Coefficients:
        """Get corruption coefficients for time t.

        Args:
            t: Time, shape (batch,).

        Returns:
            Dict with keys ``"beta"`` and ``"beta_prime"``.
        """
        return {
            "beta": self.beta(t),
            "beta_prime": self.beta_prime(t),
        }


# ===========================================================================
# Mask (absorbing) diffusion
# ===========================================================================


class MaskDiffusion(DiscreteCorruptionProcess):
    """Discrete diffusion with mask / absorbing corruption.

    Each token independently transitions to a special [MASK] token
    (index = ``vocab_size``) with probability β(t).  Tokens that are
    not corrupted remain unchanged.

    Forward process ``q(x_t | x_0)`` for each position::

        P(x_t = x_0)   = 1 − β(t)
        P(x_t = MASK)   = β(t)

    Denoising posterior ``q(x_s | x_t, x_0)`` for s < t:

    - If x_t ≠ MASK (token survived): x_s = x_t deterministically.
    - If x_t = MASK::

        P(x_s = j)     = (β(t) − β(s)) / β(t) · P(x_0 = j)   for j < K
        P(x_s = MASK)  = β(s) / β(t)

    Attributes:
        vocab_size: Number of regular token classes K.
        seq_shape: Shape of a single sequence (excluding batch dim).
        padding_token: Optional padding token index.
        mask_index: Index of the MASK token (= vocab_size).
        output_classes: Total number of output classes (K + 1).
    """

    def __init__(
        self,
        vocab_size: int,
        seq_shape: tuple[int, ...],
        padding_token: int | None = None,
    ):
        """Initialize mask diffusion.

        Args:
            vocab_size: Number of regular token classes K.
            seq_shape: Shape of a single sequence (excluding batch dim).
            padding_token: Optional padding token index.
        """
        super().__init__(vocab_size, seq_shape, padding_token=padding_token)
        self.mask_index: int = vocab_size
        self.output_classes: int = vocab_size + 1

    def _replacement_tokens(self, r: DiscreteData) -> DiscreteData:
        """Return MASK tokens (ignoring random input *r*).

        Args:
            r: Unused random tokens, shape (batch, *seq_shape).

        Returns:
            All-MASK tokens, shape (batch, *seq_shape).
        """
        return jnp.full_like(r, self.mask_index)

    def _sample_terminal(self, key: PRNGKey, batch_size: int) -> DiscreteData:
        """Sample all-MASK terminal state.

        Args:
            key: PRNG key (unused; terminal state is deterministic).
            batch_size: Number of samples.

        Returns:
            All-MASK tokens, shape (batch_size, *seq_shape).
        """
        return jnp.full((batch_size, *self.seq_shape), self.mask_index, dtype=jnp.int32)

    def _denoising_posterior_probs(
        self,
        probs_x0: ProbabilisticData,
        x_t: DiscreteData,
        s: Time,
        t: Time,
    ) -> ProbabilisticData:
        """Compute denoising posterior for mask diffusion.

        Returns probabilities over K + 1 classes (MASK at index K).

        For masked positions::

            P(x_s = j)    = (β(t) − β(s)) / β(t) · P(x_0 = j)   (j < K)
            P(x_s = MASK) = β(s) / β(t)

        For unmasked positions::

            P(x_s = x_t) = 1   (deterministic)

        Args:
            probs_x0: Predicted distribution over x_0,
                shape (batch, *seq_shape, vocab_size).
            x_t: Corrupted tokens at time t, shape (batch, *seq_shape).
            s: Target (earlier) time, shape (batch,).
            t: Current time, shape (batch,).

        Returns:
            Posterior probabilities, shape (batch, *seq_shape, vocab_size + 1).
        """
        beta_s = self.beta(s)  # (batch,)
        beta_t = self.beta(t)  # (batch,)
        K = self.vocab_size

        # Broadcast scalars to spatial dims of x_t: (batch, 1, ...)
        spatial_ndim = x_t.ndim  # 1 + len(seq_shape)
        beta_s_bc = bcast_right(beta_s, spatial_ndim)  # (batch, 1, ...)
        beta_t_bc = bcast_right(beta_t, spatial_ndim)  # (batch, 1, ...)

        is_masked = x_t == self.mask_index  # (batch, *seq_shape)

        # Guard against β(t) = 0 (no masking ⇒ masked branch unused)
        safe_beta_t = jnp.maximum(beta_t_bc, 1e-20)

        # Weights for the masked-position posterior
        unmask_weight = (safe_beta_t - beta_s_bc) / safe_beta_t  # (batch, 1, ...)
        stay_mask_weight = beta_s_bc / safe_beta_t  # (batch, 1, ...)

        # probs_x0: (batch, *seq_shape, K)
        # unmask_weight[..., None]: (batch, 1, ..., 1) broadcasts over seq + class
        probs_regular = unmask_weight[..., None] * probs_x0  # (batch, *seq_shape, K)

        # Broadcast stay_mask_weight to (batch, *seq_shape, 1) for concat
        prob_mask = jnp.broadcast_to(
            stay_mask_weight[..., None], (*x_t.shape, 1)
        )  # (batch, *seq_shape, 1)

        masked_posterior = jnp.concatenate(
            [probs_regular, prob_mask], axis=-1
        )  # (batch, *seq_shape, K+1)

        # Unmasked positions: one-hot on x_t
        unmasked_posterior = jax.nn.one_hot(x_t, K + 1)  # (batch, *seq_shape, K+1)

        # Select based on whether position is masked
        posterior = jnp.where(
            is_masked[..., None], masked_posterior, unmasked_posterior
        )  # (batch, *seq_shape, K+1)

        return posterior


# ===========================================================================
# Uniform diffusion
# ===========================================================================


class UniformDiffusion(DiscreteCorruptionProcess):
    """Discrete diffusion with uniform corruption.

    Each token independently transitions to a uniformly random token
    from {0, …, K−1} with probability β(t).  Tokens that are not
    corrupted remain unchanged.  (The random replacement may happen
    to equal the original token.)

    Forward process ``q(x_t | x_0)`` for each position::

        P(x_t = x_0)     = 1 − β(t) + β(t)/K  = 1 − β(t)(K−1)/K
        P(x_t = j ≠ x_0) = β(t)/K

    Denoising posterior ``q(x_s | x_t, x̂_0)``::

        q(x_s = j | x_t, x̂_0) ∝ q(x_t | x_s = j) · q(x_s = j | x̂_0)

    where::

        q(x_t | x_s = j) = γ(s,t) δ_{x_t,j} + (1−γ(s,t))/K
        q(x_s = j | x̂_0) = (1−β(s)) p(x_0 = j) + β(s)/K
        γ(s,t) = (1−β(t)) / (1−β(s))

    Attributes:
        vocab_size: Number of token classes K.
        seq_shape: Shape of a single sequence (excluding batch dim).
        padding_token: Optional padding token index.
    """

    def _replacement_tokens(self, r: DiscreteData) -> DiscreteData:
        """Return the random replacement tokens.

        Args:
            r: Random tokens ∈ {0, …, K−1}, shape (batch, *seq_shape).

        Returns:
            Same tokens *r*, shape (batch, *seq_shape).
        """
        return r

    def _sample_terminal(self, key: PRNGKey, batch_size: int) -> DiscreteData:
        """Sample uniformly random terminal state.

        Args:
            key: PRNG key.
            batch_size: Number of samples.

        Returns:
            Uniform random tokens, shape (batch_size, *seq_shape).
        """
        return jax.random.randint(
            key, (batch_size, *self.seq_shape), 0, self.vocab_size
        )

    def _denoising_posterior_probs(
        self,
        probs_x0: ProbabilisticData,
        x_t: DiscreteData,
        s: Time,
        t: Time,
    ) -> ProbabilisticData:
        """Compute denoising posterior for uniform diffusion.

        Returns probabilities over K classes.

        Uses Bayes' rule::

            q(x_s = j | x_t, x̂_0)
              ∝ [γ δ_{x_t,j} + (1−γ)/K] · [(1−β_s) p(x_0=j) + β_s/K]

        Args:
            probs_x0: Predicted distribution over x_0,
                shape (batch, *seq_shape, vocab_size).
            x_t: Corrupted tokens at time t, shape (batch, *seq_shape).
            s: Target (earlier) time, shape (batch,).
            t: Current time, shape (batch,).

        Returns:
            Posterior probabilities, shape (batch, *seq_shape, vocab_size).
        """
        beta_s = self.beta(s)  # (batch,)
        beta_t = self.beta(t)  # (batch,)
        K = self.vocab_size

        # γ(s,t) = (1 − β(t)) / (1 − β(s))
        safe_denom = jnp.maximum(1.0 - beta_s, 1e-20)
        gamma_st = (1.0 - beta_t) / safe_denom  # (batch,)

        # Broadcast to probability tensor dims: (batch, *seq_shape, K)
        probs_ndim = probs_x0.ndim  # 1 + len(seq_shape) + 1
        gamma_bc = bcast_right(gamma_st, probs_ndim)  # (batch, 1, ..., 1)
        beta_s_bc = bcast_right(beta_s, probs_ndim)  # (batch, 1, ..., 1)

        # q(x_t | x_s = j) = γ δ_{x_t,j} + (1−γ)/K
        x_t_onehot = jax.nn.one_hot(x_t, K)  # (batch, *seq_shape, K)
        q_xt_given_xs = (
            gamma_bc * x_t_onehot + (1.0 - gamma_bc) / K
        )  # (batch, *seq_shape, K)

        # q(x_s = j | x̂_0) = (1−β(s)) p(x_0 = j) + β(s)/K
        q_xs_marginal = (
            1.0 - beta_s_bc
        ) * probs_x0 + beta_s_bc / K  # (batch, *seq_shape, K)

        # Posterior: element-wise product, then normalise
        unnorm = q_xt_given_xs * q_xs_marginal  # (batch, *seq_shape, K)
        posterior = unnorm / jnp.sum(
            unnorm, axis=-1, keepdims=True
        )  # (batch, *seq_shape, K)

        return posterior


# ===========================================================================
# Concrete schedules – Mask
# ===========================================================================


class MaskLinear(MaskDiffusion):
    """Mask diffusion with linear corruption schedule: β(t) = t.

    Attributes:
        vocab_size: Number of regular token classes K.
        seq_shape: Shape of a single sequence (excluding batch dim).
        padding_token: Optional padding token index.
        mask_index: Index of the MASK token (= vocab_size).
        output_classes: Total number of output classes (K + 1).
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


class MaskCosine(MaskDiffusion):
    """Mask diffusion with cosine corruption schedule: β(t) = 1 − cos(πt/2).

    Starts slow (gentle corruption near t = 0) and accelerates toward t = 1.

    Attributes:
        vocab_size: Number of regular token classes K.
        seq_shape: Shape of a single sequence (excluding batch dim).
        padding_token: Optional padding token index.
        mask_index: Index of the MASK token (= vocab_size).
        output_classes: Total number of output classes (K + 1).
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


# ===========================================================================
# Concrete schedules – Uniform
# ===========================================================================


class UniformLinear(UniformDiffusion):
    """Uniform diffusion with linear corruption schedule: β(t) = t.

    Attributes:
        vocab_size: Number of token classes K.
        seq_shape: Shape of a single sequence (excluding batch dim).
        padding_token: Optional padding token index.
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


class UniformCosine(UniformDiffusion):
    """Uniform diffusion with cosine corruption schedule: β(t) = 1 − cos(πt/2).

    Starts slow and accelerates, matching cosine noise schedules.

    Attributes:
        vocab_size: Number of token classes K.
        seq_shape: Shape of a single sequence (excluding batch dim).
        padding_token: Optional padding token index.
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
