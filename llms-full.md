# diffusionlab

> No-frills JAX library providing core abstractions for diffusion models.

## Overview

# DiffusionLab

<div align="center">

![DiffusionLab Logo](https://druvpai.github.io/DiffusionLab/diffusionlab_logo.svg)

`pip install diffusionlab` • [`llms.txt`](https://druvpai.github.io/DiffusionLab/llms-full.md)

[![PyPI version](https://img.shields.io/pypi/v/diffusionlab)](https://pypi.org/project/diffusionlab/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/downloads/)

</div>

A no-frills JAX library providing core abstractions for diffusion models. DiffusionLab focuses on corruption/forward processes for continuous, discrete, and simplicial (soft-categorical) data, designed to be easy to understand, modify, and extend for research.

The API is **PyTree-first**: data, times, and predictions can be arbitrarily nested (e.g. multimodal dicts), and everything composes naturally with JAX transformations.

## Features

- **Corruption processes** for continuous (Gaussian), discrete (absorbing/uniform), and simplicial (Dirichlet) data
- **Reverse-process samplers** including Euler, DDM, and discrete ancestral sampling
- **Training losses** with configurable SNR weighting and time-sampling schedules
- **Reference model architectures**: MLP and DiT (Diffusion Transformer)
- **Classifier-free guidance** out of the box
- **Multimodal support**: jointly corrupt and denoise mixed continuous/discrete data with a single model

## Installation

Install from PyPI:

```bash
pip install diffusionlab
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add diffusionlab
```

### From source

```bash
git clone https://github.com/druvpai/DiffusionLab.git
cd DiffusionLab
uv sync
```

## Quick start

Set up and sample from a flow-matching model on 2D data in ~30 lines:

```python
import jax
import jax.numpy as jnp
from diffusionlab.processes.gaussian import FlowMatching
from diffusionlab.models.mlp import DiffusionMLP
from diffusionlab.sampling.continuous import EulerSampler
from diffusionlab.sampling.base import sample
from diffusionlab.sampling.schedules import uniform_schedule
from diffusionlab.training.denoising import DenoisingLoss
from diffusionlab.training.schedules import uniform_time_with_eps

# Define process and model
process = FlowMatching(data_shape=(2,))
model = DiffusionMLP(
    data_dim=2, hidden_dim=128, num_layers=4,
    time_emb_dim=64, cond_dim=64, prediction_kind="v_t",
    key=jax.random.key(0),
)

# Compute loss on a batch
loss_obj = DenoisingLoss(process=process)
key = jax.random.key(1)
key_t, key_fwd = jax.random.split(key)
t = uniform_time_with_eps(key_t, batch=256)
x_t, aux = process.forward(key_fwd, x_0, t)
out = loss_obj(predictor=model, x_0=x_0, x_t=x_t, t=t, aux=aux, cond=None)

# Sample from the trained model
sampler = EulerSampler(process=process)
schedule = uniform_schedule(num_steps=100, batch_size=512)
samples = sample(key=jax.random.key(2), sampler=sampler, predictor=model,
                 batch_size=512, time_schedule=schedule)
```

## Examples

The `examples/` directory contains complete, runnable scripts:

| Example                                               | Description                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------- |
| [`two_moons.py`](examples/two_moons.py)               | Flow matching on 2D synthetic data with trajectory visualization        |
| [`mnist_dit.py`](examples/mnist_dit.py)               | Class-conditional DiT on MNIST with classifier-free guidance            |
| [`mnist_multimodal.py`](examples/mnist_multimodal.py) | Joint image + label generation with mixed continuous/discrete diffusion |

Run any example with:

```bash
uv run python examples/two_moons.py
uv run python examples/mnist_dit.py
uv run python examples/mnist_multimodal.py
```

The MNIST examples download data automatically on first run.

## Project structure

```
diffusionlab/
├── processes/              # Forward corruption processes
│   ├── base.py             #   CorruptionProcess, MultimodalCorruptionProcess
│   ├── interpolation.py    #   InterpolationProcess (abstract)
│   ├── gaussian.py         #   VPCosine, VPLinear, VE, EDM, FlowMatching
│   ├── discrete.py         #   MaskLinear, MaskCosine, UniformLinear, UniformCosine
│   └── simplicial.py       #   SimplicialLinear, SimplicialCosine
├── sampling/               # Reverse-process samplers
│   ├── base.py             #   Sampler, MultimodalSampler, sample, sample_trajectory
│   ├── continuous.py       #   EulerSampler, DDMSampler
│   ├── discrete.py         #   DiscreteAncestralSampler
│   ├── guidance.py         #   CFG (classifier-free guidance)
│   └── schedules.py        #   uniform_schedule, edm_schedule
├── training/               # Training losses and utilities
│   ├── base.py             #   Loss, MultimodalLoss, mse_loss, ce_loss, kl_loss
│   ├── denoising.py        #   DenoisingLoss
│   ├── distillation.py     #   DistillationLoss
│   ├── consistency.py      #   ConsistencyLoss
│   ├── weighting.py        #   SNR-based weight functions
│   └── schedules.py        #   Time sampling strategies
├── models/                 # Reference neural network architectures
│   ├── base.py             #   DiffusionModel (abstract)
│   ├── components.py       #   SinusoidalEmbedding, FiLM, AdaLNZero
│   ├── mlp.py              #   DiffusionMLP, ConditionedDiffusionMLP
│   └── dit.py              #   DiT, ConditionedDiT
├── typing.py               # Prediction dataclass, type aliases
└── utils/
    └── tree_ops.py         # PyTree utilities (bcast_right, tree_map_with_key)
```

## Development

### Prerequisites

- Python >= 3.13
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Setup

```bash
git clone https://github.com/druvpai/DiffusionLab.git
cd DiffusionLab
uv sync
```

### Running tests

```bash
# Run the full test suite
uv run pytest tests/

# Run a specific test file
uv run pytest tests/processes/test_gaussian.py

# Run tests matching a pattern
uv run pytest tests/ -k "test_forward"
```

### Linting and formatting

```bash
uv run ruff format diffusionlab/ tests/
uv run ruff check diffusionlab/ tests/
```

### Type checking

```bash
uv run ty check diffusionlab/
```

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository and clone your fork
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b my-feature
   ```
3. **Install** development dependencies:
   ```bash
   uv sync
   ```
4. **Make your changes** — keep them focused and minimal
5. **Add tests** for any new functionality in `tests/`
6. **Run the checks** before submitting:
   ```bash
   uv run ruff format diffusionlab/ tests/
   uv run ruff check diffusionlab/ tests/
   uv run pytest tests/
   ```
7. **Open a pull request** with a clear description of your changes

## Citation

If you use DiffusionLab in your research, please consider citing it:

```bibtex
@Misc{pai25diffusionlab,
    author = {Pai, Druv},
    title = {DiffusionLab},
    howpublished = {\url{https://github.com/DruvPai/DiffusionLab}},
    year = {2025}
}
```

Many thanks!

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

This package was greatly inspired by [Hackable Diffusion](https://github.com/google/hackable_diffusion), but the implementations and some core primitives are very different, leading to greater flexibility and extensibility (IMO).

## API Reference

## `diffusionlab.models.base`

Abstract base classes for diffusion model architectures.

### `DiffusionModel`

**class** `DiffusionModel(prediction_kinds: jaxtyping.PyTree[str]) -> None`

Abstract base class for diffusion model architectures.

Attributes:
    prediction_kinds: PyTree of the kind of predictions made at each leaf of the data PyTree.

## `diffusionlab.models.components`

Shared components for diffusion model architectures.

Provides reusable building blocks: sinusoidal time embeddings,
FiLM conditioning layers, and adaptive layer normalization (AdaLN-Zero).

### `AdaLNZero`

**class** `AdaLNZero(hidden_dim: int, cond_dim: int, key: jaxtyping.Key[Array, ''])`

Adaptive Layer Normalization with zero-initialization (AdaLN-Zero).

Produces modulation vectors (gamma, beta, alpha) from a conditioning vector.
from a conditioning vector. The projection is zero-initialized so the model
starts near identity.

Attributes:
    proj: Zero-initialized MLP projection, ``(cond_dim,) -> (3 * hidden_dim,)``.
    norm: RMS normalization for the hidden state.

### `FiLM`

**class** `FiLM(hidden_dim: int, cond_dim: int, key: jaxtyping.Key[Array, ''])`

Feature-wise Linear Modulation (FiLM) layer.

Applies affine conditioning after layer normalization:
``(1 + gamma) * LayerNorm(Linear(h)) + beta``
where ``[beta, gamma] = film_proj(cond)``.

Attributes:
    proj: Conditioning projection, ``(cond_dim,) -> (2 * hidden_dim,)``.
    norm: RMS normalization over hidden_dim.

### `SinusoidalEmbedding`

**class** `SinusoidalEmbedding(emb_dim: int, max_period: float) -> None`

Sinusoidal positional/time embedding (DDPM-style).

Maps scalar values to dense vectors using fixed sinusoidal frequencies.

Attributes:
    emb_dim: Dimensionality of the output embedding. Must be even.
    max_period: Maximum period for the sinusoidal frequencies.

## `diffusionlab.models.dit`

Diffusion Transformer (DiT) architectures for sequence data.

### `ConditionedDiT`

**class** `ConditionedDiT(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, num_heads: int, head_dim: int, mlp_dim: int, max_seq_len: int, time_emb_dim: int, cond_emb_dim: int, cond_dim: int, prediction_kind: str, activation: Callable, max_period: float, key: jax.Array)`

Diffusion Transformer with external conditioning for sequence data.

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

### `DiT`

**class** `DiT(input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, num_heads: int, head_dim: int, mlp_dim: int, max_seq_len: int, time_emb_dim: int, cond_dim: int, prediction_kind: str, activation: Callable, max_period: float, key: jax.Array)`

Diffusion Transformer for sequence data.

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

### `DiTBlock`

**class** `DiTBlock(hidden_dim: int, num_heads: int, head_dim: int, mlp_dim: int, cond_dim: int, activation: Callable, key: jax.Array)`

Single transformer block with AdaLN-Zero conditioning.

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

## `diffusionlab.models.mlp`

FiLM-conditioned MLP for diffusion models on flat vector data.

### `ConditionedDiffusionMLP`

**class** `ConditionedDiffusionMLP(data_dim: int, hidden_dim: int, num_layers: int, time_emb_dim: int, cond_emb_dim: int, cond_dim: int, prediction_kind: str, activation: Callable, max_period: float, key: jaxtyping.Key[Array, ''])`

FiLM-conditioned MLP for diffusion on flat vector data.

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

### `DiffusionMLP`

**class** `DiffusionMLP(data_dim: int, hidden_dim: int, num_layers: int, time_emb_dim: int, cond_dim: int, prediction_kind: str, activation: Callable, max_period: float, key: jaxtyping.Key[Array, ''])`

FiLM-conditioned MLP for diffusion on flat vector data.

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

### `MLPBlock`

**class** `MLPBlock(hidden_dim: int, cond_dim: int, activation: Callable, key: jaxtyping.Key[Array, ''])`

MLP block with FiLM conditioning.

Attributes:
    film: FiLM module.
    mlp: MLP module, ``(hidden_dim,) -> (hidden_dim,)``.

## `diffusionlab.processes.base`

Base class for diffusion processes.

### `CorruptionProcess`

**class** `CorruptionProcess()`

Base class for data corruption / diffusion forward processes.

A corruption process defines a forward mapping (x_0 -> x_t) for a
(potentially tree-structured) data object `x` and time `t`.

### `MultimodalCorruptionProcess`

**class** `MultimodalCorruptionProcess(processes: jaxtyping.PyTree[diffusionlab.processes.base.CorruptionProcess, 'T'])`

A nested corruption process involving multiple corruption processes on independent data.

## `diffusionlab.processes.discrete`

Discrete diffusion processes for categorical data.

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

### `DiscreteCorruptionProcess`

**class** `DiscreteCorruptionProcess(vocab_size: int, seq_shape: tuple[int, ...], padding_token: int | None = None)`

Abstract base class for discrete (categorical) diffusion processes.

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

### `MaskCosine`

**class** `MaskCosine(vocab_size: int, seq_shape: tuple[int, ...], padding_token: int | None = None)`

Mask diffusion with cosine corruption schedule: β(t) = 1 − cos(πt/2).

Starts slow (gentle corruption near t = 0) and accelerates toward t = 1.

Attributes:
    vocab_size: Number of regular token classes K.
    seq_shape: Shape of a single sequence (excluding batch dim).
    padding_token: Optional padding token index.
    mask_index: Index of the MASK token (= vocab_size).
    output_classes: Total number of output classes (K + 1).

### `MaskDiffusion`

**class** `MaskDiffusion(vocab_size: int, seq_shape: tuple[int, ...], padding_token: int | None = None)`

Discrete diffusion with mask / absorbing corruption.

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

### `MaskLinear`

**class** `MaskLinear(vocab_size: int, seq_shape: tuple[int, ...], padding_token: int | None = None)`

Mask diffusion with linear corruption schedule: β(t) = t.

Attributes:
    vocab_size: Number of regular token classes K.
    seq_shape: Shape of a single sequence (excluding batch dim).
    padding_token: Optional padding token index.
    mask_index: Index of the MASK token (= vocab_size).
    output_classes: Total number of output classes (K + 1).

### `UniformCosine`

**class** `UniformCosine(vocab_size: int, seq_shape: tuple[int, ...], padding_token: int | None = None)`

Uniform diffusion with cosine corruption schedule: β(t) = 1 − cos(πt/2).

Starts slow and accelerates, matching cosine noise schedules.

Attributes:
    vocab_size: Number of token classes K.
    seq_shape: Shape of a single sequence (excluding batch dim).
    padding_token: Optional padding token index.

### `UniformDiffusion`

**class** `UniformDiffusion(vocab_size: int, seq_shape: tuple[int, ...], padding_token: int | None = None)`

Discrete diffusion with uniform corruption.

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

### `UniformLinear`

**class** `UniformLinear(vocab_size: int, seq_shape: tuple[int, ...], padding_token: int | None = None)`

Uniform diffusion with linear corruption schedule: β(t) = t.

Attributes:
    vocab_size: Number of token classes K.
    seq_shape: Shape of a single sequence (excluding batch dim).
    padding_token: Optional padding token index.

## `diffusionlab.processes.gaussian`

Gaussian diffusion processes.

Special case of interpolation processes where the source distribution
is a standard Gaussian: x(t) = α(t) x_0 + σ(t) ε, ε ~ N(0, I).

This module provides:
- GaussianDiffusion: abstract base for Gaussian-source interpolation
- VPCosine: variance-preserving with cosine schedule
- VPLinear: variance-preserving with linear beta schedule (DDPM)
- VE: variance-exploding with geometric σ schedule
- EDM: Karras et al. (2022) power-law σ schedule
- FlowMatching: linear interpolation (optimal transport / rectified flow)

### `EDM`

**class** `EDM(data_shape: tuple[int, ...], sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0)`

EDM (Karras et al. 2022) diffusion schedule.

Uses the EDM power-law noise schedule with VE-style forward process:

α(t)  = 1
σ(t)  = ((1 - t) σ_min^(1/ρ) + t σ_max^(1/ρ))^ρ

The power-law interpolation (controlled by ρ) concentrates more of
the schedule's dynamic range at higher noise levels, matching the
empirically effective training noise distribution from Karras et al.

Note: σ(0) = σ_min > 0, so the data at t=0 is slightly noisy.
For generation, σ_min should be small enough to be negligible.

Attributes:
    data_shape: Shape of a single data sample (excluding batch dim).
    sigma_min: Minimum noise level (at t=0).
    sigma_max: Maximum noise level (at t=1).
    rho: Power-law exponent controlling schedule curvature.

### `FlowMatching`

**class** `FlowMatching(data_shape: tuple[int, ...])`

Linear interpolation with Gaussian source (optimal transport / rectified flow).

α(t) = 1 - t
σ(t) = t

The simplest interpolation schedule, corresponding to optimal transport /
rectified flow with a Gaussian prior.

Boundaries: α(0) = 1, σ(0) = 0, α(1) = 0, σ(1) = 1.

Attributes:
    data_shape: Shape of a single data sample (excluding batch dim).

### `GaussianDiffusionProcess`

**class** `GaussianDiffusionProcess(data_shape: tuple[int, ...])`

Interpolation process with standard Gaussian source distribution.

x(t) = α(t) x_0 + σ(t) ε,  ε ~ N(0, I),  t ∈ [0, 1].

Since the source is Gaussian, ε (eps) replaces the generic z from
InterpolationProcess, and an additional prediction type "score" is
available via Tweedie's formula:

    score(x_t, t) ≈ ∇_{x_t} log p_t(x_t) = -ε / σ(t)

Available prediction kinds:
- x_0: the original data
- eps: the Gaussian noise ε
- v: the velocity field v(t) = α'(t) x_0 + σ'(t) ε
- score: the score function ∇ log p_t(x_t) ≈ -ε / σ(t)
- x_s: the estimated noised data at intermediate time s (requires s)
- v_st: the average velocity between s and t (requires s)

Attributes:
    data_shape: Shape of a single data sample (excluding batch dim).

### `VE`

**class** `VE(data_shape: tuple[int, ...], sigma_min: float = 0.01, sigma_max: float = 50.0)`

Variance-exploding diffusion with geometric noise schedule.

α(t) = 1
σ(t) = σ_min · (σ_max / σ_min)^t

The noise level grows geometrically from σ_min to σ_max.
At t=0, the data is approximately clean (σ_min ≈ 0).
At t=1, the terminal distribution is approximately N(0, σ_max² I).

Note: α(t) = 1 for all t, so "v_t" predictions carry the same information
as "eps" (v(t) = σ'(t) ε). All conversions remain valid.

Attributes:
    data_shape: Shape of a single data sample (excluding batch dim).
    sigma_min: Minimum noise level (at t=0).
    sigma_max: Maximum noise level (at t=1).

### `VPCosine`

**class** `VPCosine(data_shape: tuple[int, ...])`

Variance-preserving diffusion with cosine schedule.

α(t) = cos(π t / 2)
σ(t) = sin(π t / 2)

Satisfies the VP property α(t)² + σ(t)² = 1.
Boundaries: α(0) = 1, σ(0) = 0, α(1) = 0, σ(1) = 1.

Attributes:
    data_shape: Shape of a single data sample (excluding batch dim).

### `VPLinear`

**class** `VPLinear(data_shape: tuple[int, ...], beta_min: float = 0.1, beta_max: float = 20.0)`

Variance-preserving diffusion with linear beta schedule (DDPM-style).

β(t)  = β_min + (β_max - β_min) t
α(t)  = exp(-½ ∫₀ᵗ β(s) ds)
      = exp(-½ (β_min t + (β_max - β_min) t² / 2))
σ(t)  = √(1 - α(t)²)

Satisfies the VP property α(t)² + σ(t)² = 1.

Note: σ'(t) diverges as t → 0. Avoid evaluating at t = 0 exactly.

Attributes:
    data_shape: Shape of a single data sample (excluding batch dim).
    beta_min: Minimum value of the linear beta schedule.
    beta_max: Maximum value of the linear beta schedule.

## `diffusionlab.processes.interpolation`

### `InterpolationProcess`

**class** `InterpolationProcess()`

Interpolation process between two distributions:

x(t) = α(t) * x_0 + σ(t) * z, t in [0, 1],

where z (the endpoint distribution) need not be Gaussian.

Types of predictions one can make in this model (given x(t) and t):
- x_0: the original data
- z: the distribution to transport to law(x_0)
- v(t): the velocity field at time t, aka x'(t) = α'(t) * x_0 + σ'(t) * z

Given x(t) and s, t, such that s < t, one can also compute:
- x(s): the estimated noised data at time s
- v(s, t): the average velocity field between time s and t, aka (x(t) - x(s)) / (t - s)

## `diffusionlab.processes.simplicial`

Simplicial diffusion processes for categorical data (Dirichlet corruption).

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

### `SimplicialCorruptionProcess`

**class** `SimplicialCorruptionProcess(seq_shape: tuple[int, ...], invariant_probs: jax.Array, temperature: float = 1.0, padding_token: int | None = None, safety_epsilon: float = 1e-06)`

Abstract base class for simplicial (Dirichlet) corruption processes.

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

### `SimplicialCosine`

**class** `SimplicialCosine(seq_shape: tuple[int, ...], invariant_probs: jax.Array, temperature: float = 1.0, padding_token: int | None = None, safety_epsilon: float = 1e-06)`

Simplicial diffusion with cosine corruption schedule: β(t) = 1 − cos(πt/2).

Starts slow (gentle corruption near t = 0) and accelerates toward
t = 1, matching the popular cosine schedule.

Attributes:
    seq_shape: Shape of a single sequence (excluding batch dim).
    invariant_probs: Invariant distribution π.
    temperature: Temperature τ.
    padding_token: Optional padding token index.
    safety_epsilon: Numerical stability constant ε.

### `SimplicialLinear`

**class** `SimplicialLinear(seq_shape: tuple[int, ...], invariant_probs: jax.Array, temperature: float = 1.0, padding_token: int | None = None, safety_epsilon: float = 1e-06)`

Simplicial diffusion with linear corruption schedule: β(t) = t.

The corruption probability increases linearly from 0 to 1.  The
concentration ratio h(t) = (1 − t) / (t + ε), which decreases
monotonically and reaches 0 at t = 1.

Attributes:
    seq_shape: Shape of a single sequence (excluding batch dim).
    invariant_probs: Invariant distribution π.
    temperature: Temperature τ.
    padding_token: Optional padding token index.
    safety_epsilon: Numerical stability constant ε.

## `diffusionlab.sampling.base`

Sampler base class, multimodal sampler, and sampling orchestrator.

### `MultimodalSampler`

**class** `MultimodalSampler(samplers: jaxtyping.PyTree[diffusionlab.sampling.base.Sampler, 'T'])`

Sampler for multimodal data, wrapping a pytree of per-modality samplers.

Mirrors `MultimodalCorruptionProcess`.

Attributes:
    samplers: Pytree of `Sampler` instances (one per modality).

### `Sampler`

**class** `Sampler(*, process: diffusionlab.processes.base.CorruptionProcess) -> None`

Abstract base class for reverse-process samplers.

Each sampler is tied to a specific `CorruptionProcess` and
implements the three-step protocol:

1. `initialize` — wrap terminal noise into a `SamplerStep`.
2. `update` — given a model prediction, step from time *t* to *s*.
3. `finalize` — extract clean data from the last prediction.

Attributes:
    process: The corruption process this sampler reverses.

### `SamplerStep`

**class** `SamplerStep(*, x: jaxtyping.PyTree[jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], 'T'], t: jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T'], aux: jaxtyping.PyTree[dict[str, jax.Array], 'T']) -> None`

State of the sampler at a single time step.

Attributes:
    x: The current data (corrupted sample at time t).
    t: The current time.
    aux: Auxiliary schedule information at time t.

### `SamplingTransform`

**class** `SamplingTransform() -> None`

Abstract base class for sampling transforms.

Transforms intercept the sampling loop at two points:

1. ``transform_prediction`` — modify the model prediction *before*
   the sampler update (e.g. classifier-free guidance, DPS).
2. ``transform_step`` — modify the ``SamplerStep`` *after* the
   sampler update.

Subclasses override one or both hooks.  The default implementations
are identity functions.

### `sample`

**function** `sample(key: jaxtyping.Key[Array, ''], sampler: diffusionlab.sampling.base.Sampler, predictor: diffusionlab.typing.DiffusionPredictor, batch_size: int, time_schedule: list[jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']], cond: jaxtyping.PyTree[jaxtyping.Float[Array, 'batch *#?cond_shape'], 'T'] | None = None, transforms: list[diffusionlab.sampling.base.SamplingTransform] | None = None) -> jaxtyping.PyTree[jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], 'T']`

Run the full reverse-process sampling loop.

Args:
    key: PRNG key.
    sampler: A `Sampler` (or `MultimodalSampler`) object.
    predictor: A `DiffusionPredictor` object.
    batch_size: Number of samples to generate.
    time_schedule: A ``list[TimeTree]`` of length ``num_steps + 1``,
        ordered from ``t=1`` (noisy) to ``t~0`` (clean).  Typically
        produced by `uniform_schedule` or `edm_schedule`.  For
        multimodal samplers each element is a pytree of ``Time``
        arrays matching the sampler structure.
    cond: Optional conditioning data passed through to *predictor*.
    transforms: Optional list of ``SamplingTransform`` instances applied
        sequentially at each step.

Returns:
    Estimated clean data.

### `sample_trajectory`

**function** `sample_trajectory(key: jaxtyping.Key[Array, ''], sampler: diffusionlab.sampling.base.Sampler, predictor: diffusionlab.typing.DiffusionPredictor, batch_size: int, time_schedule: list[jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']], cond: jaxtyping.PyTree[jaxtyping.Float[Array, 'batch *#?cond_shape'], 'T'] | None = None, transforms: list[diffusionlab.sampling.base.SamplingTransform] | None = None) -> tuple[jaxtyping.PyTree[jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], 'T'], list[diffusionlab.sampling.base.SamplerStep]]`

Run the full reverse-process sampling loop and return the trajectory.

Identical to `sample` but additionally returns the intermediate sampler
states, useful for visualising the denoising process or diagnostics.

.. warning::
    The trajectory stores every intermediate ``SamplerStep`` in memory.
    For large batch sizes or high-dimensional data prefer `sample`.

Args:
    key: PRNG key.
    sampler: A `Sampler` (or `MultimodalSampler`) object.
    predictor: A `DiffusionPredictor` object.
    batch_size: Number of samples to generate.
    time_schedule: A ``list[TimeTree]`` of length ``num_steps + 1``,
        ordered from ``t=1`` (noisy) to ``t~0`` (clean).  Typically
        produced by `uniform_schedule` or `edm_schedule`.  For
        multimodal samplers each element is a pytree of ``Time``
        arrays matching the sampler structure.
    cond: Optional conditioning data passed through to *predictor*.
    transforms: Optional list of ``SamplingTransform`` instances applied
        sequentially at each step.

Returns:
    A tuple ``(final_data, trajectory)`` where:

    - *final_data* is the estimated clean data (same as `sample`).
    - *trajectory* is a list of `SamplerStep` objects of length
      ``num_steps``.  The first element is the initial noisy state
      (at ``t=1``), and subsequent elements are the states after each
      sampler update.

## `diffusionlab.sampling.continuous`

Samplers for Gaussian diffusion processes.

### `DDMSampler`

**class** `DDMSampler(process: diffusionlab.processes.gaussian.GaussianDiffusionProcess, eta: float = 0.0)`

Denoising Diffusion Implicit Models sampler (Song et al. 2021).

Interpolates between deterministic DDIM (`eta=0`) and stochastic
DDPM (`eta=1`) via the noise injection parameter *eta*.

Attributes:
    process: A `GaussianDiffusionProcess` object.
    eta: Stochasticity parameter.  0 = deterministic, 1 = DDPM.

### `EulerSampler`

**class** `EulerSampler(process: diffusionlab.processes.interpolation.InterpolationProcess)`

Deterministic Euler ODE integrator via velocity parameterisation.

Converts predictions to a velocity and takes an Euler step:
``x_s = x_t + (s - t) * v_hat``.

The velocity kind is inferred from the prediction: if the prediction
kind is ``"x_s"`` or ``"v_st"`` (flow-map quantities), the prediction
is converted to ``"v_st"``; otherwise it is converted to ``"v_t"``.

Attributes:
    process: An `InterpolationProcess` (or subclass) object.

## `diffusionlab.sampling.discrete`

Sampler for discrete (categorical) diffusion processes.

### `DiscreteAncestralSampler`

**class** `DiscreteAncestralSampler(process: diffusionlab.processes.discrete.DiscreteCorruptionProcess)`

Ancestral sampler for discrete corruption processes.

At each step, computes the denoising posterior `q(x_s | x_t, x_0_hat)`
via `convert_prediction` and samples from it categorically.
Finalization returns the argmax of the predicted clean distribution.

Attributes:
    process: A `DiscreteCorruptionProcess` object.

## `diffusionlab.sampling.guidance`

Guidance transforms for guided sampling.

### `CFG`

**class** `CFG(guidance_scale: float | jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T'] | Callable[[jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']], jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']])`

Classifier-free guidance transform.

Calls ``predict_fn`` a second time with ``uncond_cond`` and linearly
combines the conditional and unconditional predictions:

    pred_guided = pred_uncond + w * (pred_cond - pred_uncond)

Attributes:
    guidance_scale_fn: Guidance weight *w*, either a float or a callable
        ``(t) -> Scalar`` for time-dependent guidance.

## `diffusionlab.sampling.schedules`

Deterministic time schedules for sampling.

### `edm_schedule`

**function** `edm_schedule(num_steps: int, batch_size: int, sigma_min: float = 0.002, sigma_max: float = 80.0, rho: float = 7.0) -> list[jaxtyping.Float[Array, 'batch']]`

EDM power-law time schedule (Karras et al. 2022).

Produces a schedule of ``num_steps + 1`` time points from t=1 to t=0,
spaced according to a power-law interpolation in sigma space::

    sigma_i = (sigma_max^(1/rho) + i/N * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho

which is then converted back to the normalised time axis via
``t = (sigma^(1/rho) - sigma_min^(1/rho)) / (sigma_max^(1/rho) - sigma_min^(1/rho))``.

Args:
    num_steps: Number of sampling steps.
    batch_size: Batch dimension for each time array.
    sigma_min: Minimum noise level.
    sigma_max: Maximum noise level.
    rho: Power-law exponent (default 7.0).

Returns:
    List of ``num_steps + 1`` arrays of shape ``(batch_size,)``
    from t~=1 to t~=0.

### `uniform_schedule`

**function** `uniform_schedule(num_steps: int, batch_size: int) -> list[jaxtyping.Float[Array, 'batch']]`

Uniform time schedule from t~=1 to t~=0.

Args:
    num_steps: Number of sampling steps.
    batch_size: Batch dimension for each time array.

Returns:
    List of ``num_steps + 1`` arrays of shape ``(batch_size,)``,
    linearly spaced from t~=1 to t~=0.

## `diffusionlab.training.base`

Loss base class, LossOutput, MultimodalLoss, and standalone loss functions.

### `Loss`

**class** `Loss(*, process: diffusionlab.processes.base.CorruptionProcess, weight_fn: Callable[[jaxtyping.PyTree[dict[str, jax.Array], 'T']], jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']] | None) -> None`

Abstract base class for training losses.

Attributes:
    process: The corruption process.
    weight_fn: Optional per-sample weighting function (aux -> scalar).

### `LossOutput`

**class** `LossOutput(*, loss: jaxtyping.Float[Array, 'batch'], per_leaf: jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T'], metrics: dict[str, jaxtyping.Float[Array, 'batch'] | jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']]) -> None`

Output of a loss computation.

Attributes:
    loss: Per-sample total loss, shape (batch,).
    per_leaf: Per-modality breakdown, each shape (batch,).
    metrics: Diagnostics dict (e.g. unweighted loss, per-leaf details).

### `MultimodalLoss`

**class** `MultimodalLoss(losses: jaxtyping.PyTree[diffusionlab.training.base.Loss, 'T'], leaf_weights: jaxtyping.PyTree[float] | None = None)`

Loss for multimodal data, wrapping a pytree of per-modality losses.

Mirrors ``MultimodalSampler`` pattern.

Attributes:
    losses: Pytree of ``Loss`` instances (one per modality).
    leaf_weights: Optional per-modality contribution weights.

### `ce_loss`

**function** `ce_loss(logits: jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], target_probs: jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes']) -> jaxtyping.Float[Array, 'batch']`

Cross-entropy loss: -sum(target * log_softmax(logits)).

Averaged over spatial (non-batch, non-class) dimensions.

Args:
    logits: Unnormalized log-probabilities, shape (batch, *spatial, num_classes).
    target_probs: Target probabilities, shape (batch, *spatial, num_classes).

Returns:
    Per-sample cross-entropy, shape (batch,).

### `default_loss_fn`

**function** `default_loss_fn(process: diffusionlab.processes.base.CorruptionProcess) -> Callable[[jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes']], jaxtyping.Float[Array, 'batch']]`

Return the default loss function for a given process type.

Args:
    process: A corruption process instance.

Returns:
    ``mse_loss`` for interpolation processes,
    ``ce_loss`` for discrete processes,
    ``kl_loss`` for simplicial processes.

Raises:
    TypeError: If the process type is not recognized.

### `kl_loss`

**function** `kl_loss(logits: jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], target_probs: jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes']) -> jaxtyping.Float[Array, 'batch']`

KL divergence: sum(target * (log(target) - log_softmax(logits))).

Averaged over spatial (non-batch, non-class) dimensions.

Args:
    logits: Unnormalized log-probabilities, shape (batch, *spatial, num_classes).
    target_probs: Target probabilities, shape (batch, *spatial, num_classes).

Returns:
    Per-sample KL divergence, shape (batch,).

### `mse_loss`

**function** `mse_loss(pred: jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], target: jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes']) -> jaxtyping.Float[Array, 'batch']`

Mean squared error, averaged over non-batch dimensions.

Args:
    pred: Predicted values, shape (batch, *data_shape).
    target: Target values, shape (batch, *data_shape).

Returns:
    Per-sample MSE, shape (batch,).

## `diffusionlab.training.consistency`

Self-consistency / consistency training loss.

### `ConsistencyLoss`

**class** `ConsistencyLoss(*, process: diffusionlab.processes.base.CorruptionProcess, weight_fn: Callable[[jaxtyping.PyTree[dict[str, jax.Array], 'T']], jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']] | None, loss_fn: Callable[[jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes']], jaxtyping.Float[Array, 'batch']] | None) -> None`

Consistency training loss.

Enforces that the predictor's x_0 estimate is consistent across two
corruption levels (times t and s < t).

Attributes:
    process: The corruption process.
    weight_fn: Optional per-sample weighting function.
    loss_fn: Element-wise loss function. Auto-selected from process if None.

## `diffusionlab.training.denoising`

Standard denoising training loss.

### `DenoisingLoss`

**class** `DenoisingLoss(*, process: diffusionlab.processes.base.CorruptionProcess, weight_fn: Callable[[jaxtyping.PyTree[dict[str, jax.Array], 'T']], jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']] | None, loss_fn: Callable[[jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes']], jaxtyping.Float[Array, 'batch']] | None, target_kind: str | None) -> None`

Standard denoising loss for diffusion training.

Calls the predictor on corrupted data and compares against ground truth
in a configurable representation space.

Attributes:
    process: The corruption process.
    weight_fn: Optional per-sample weighting function.
    loss_fn: Element-wise loss function. Auto-selected from process if None.
    target_kind: Representation space for comparison.
        None means auto: use prediction.kind for continuous,
        ``"x_0"`` for discrete/simplicial.

## `diffusionlab.training.distillation`

Teacher-student distillation loss.

### `DistillationLoss`

**class** `DistillationLoss(*, process: diffusionlab.processes.base.CorruptionProcess, weight_fn: Callable[[jaxtyping.PyTree[dict[str, jax.Array], 'T']], jaxtyping.PyTree[jaxtyping.Float[Array, 'batch'], 'T']] | None, loss_fn: Callable[[jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes']], jaxtyping.Float[Array, 'batch']] | None, comparison_kind: str) -> None`

Teacher-student distillation loss.

Both student and teacher predictions are converted to a common
``comparison_kind`` space before computing the loss.

Attributes:
    process: The corruption process.
    weight_fn: Optional per-sample weighting function.
    loss_fn: Element-wise loss function. Auto-selected from process if None.
    comparison_kind: Representation space to compare in (e.g. ``"x_0"``, ``"z"``).

## `diffusionlab.training.schedules`

Stochastic time schedules for training.

### `logit_normal_time`

**function** `logit_normal_time(key: jaxtyping.Key[Array, ''], batch_size: int, mean: float = 0.0, std: float = 1.0) -> jaxtyping.Float[Array, 'batch']`

Sample times via logit-normal distribution (SD3 / rectified flow).

Draws z ~ N(mean, std^2) and returns sigmoid(z).

Args:
    key: PRNG key.
    batch_size: Number of times to sample.
    mean: Mean of the underlying normal distribution.
    std: Standard deviation of the underlying normal distribution.

Returns:
    Array of shape ``(batch_size,)`` with values in (0, 1).

### `logsnr_uniform_time`

**function** `logsnr_uniform_time(key: jaxtyping.Key[Array, ''], batch_size: int, logsnr_inverse: Callable, logsnr_min: float = -10.0, logsnr_max: float = 10.0) -> jaxtyping.Float[Array, 'batch']`

Sample times uniformly in log-SNR space.

Draws uniform samples in [logsnr_min, logsnr_max] and maps back to
the time domain via ``logsnr_inverse``.

Args:
    key: PRNG key.
    batch_size: Number of times to sample.
    logsnr_inverse: Callable mapping log-SNR values to times.
    logsnr_min: Minimum log-SNR value.
    logsnr_max: Maximum log-SNR value.

Returns:
    Array of shape ``(batch_size,)`` with valid time values.

### `stratified_time`

**function** `stratified_time(key: jaxtyping.Key[Array, ''], batch_size: int) -> jaxtyping.Float[Array, 'batch']`

Stratified time sampling for variance reduction.

Divides [0, 1] into ``batch_size`` equal strata and samples one
point uniformly from each stratum.

Args:
    key: PRNG key.
    batch_size: Number of times to sample.

Returns:
    Array of shape ``(batch_size,)`` with stratified values in [0, 1].

### `uniform_time`

**function** `uniform_time(key: jaxtyping.Key[Array, ''], batch_size: int) -> jaxtyping.Float[Array, 'batch']`

Sample times uniformly from U[0, 1].

Args:
    key: PRNG key.
    batch_size: Number of times to sample.

Returns:
    Array of shape ``(batch_size,)`` with values in [0, 1].

### `uniform_time_with_eps`

**function** `uniform_time_with_eps(key: jaxtyping.Key[Array, ''], batch_size: int, eps: float = 1e-05) -> jaxtyping.Float[Array, 'batch']`

Sample times uniformly from U[eps, 1-eps].

Args:
    key: PRNG key.
    batch_size: Number of times to sample.
    eps: Small offset from boundaries. Default 1e-5.

Returns:
    Array of shape ``(batch_size,)`` with values in [eps, 1-eps].

## `diffusionlab.training.weighting`

Standalone weight functions for training losses.

### `inverse_sigma_squared_weight`

**function** `inverse_sigma_squared_weight(aux: dict[str, jax.Array]) -> jaxtyping.Float[Array, 'batch']`

Inverse sigma squared weighting: 1/sigma^2.

### `make_weight_fn`

**function** `make_weight_fn(fn: Callable, **kwargs) -> Callable[[dict[str, jax.Array]], jaxtyping.Float[Array, 'batch']]`

Create a weight function with partially applied parameters.

Args:
    fn: A weight function that takes (aux, **kwargs).
    **kwargs: Parameters to partially apply.

Returns:
    A ``WeightFn`` with the given parameters baked in.

### `min_snr_weight`

**function** `min_snr_weight(aux: dict[str, jax.Array], gamma: float = 5.0) -> jaxtyping.Float[Array, 'batch']`

Min-SNR weighting (Hang et al. 2023): min(SNR, gamma) / SNR.

### `p2_weight`

**function** `p2_weight(aux: dict[str, jax.Array], k: float = 1.0, gamma: float = 1.0) -> jaxtyping.Float[Array, 'batch']`

P2 weighting (Choi et al. 2022): 1 / (k + SNR)^gamma.

### `snr_weight`

**function** `snr_weight(aux: dict[str, jax.Array]) -> jaxtyping.Float[Array, 'batch']`

SNR weighting: exp(logsnr).

### `truncated_snr_weight`

**function** `truncated_snr_weight(aux: dict[str, jax.Array], max_weight: float = 5.0) -> jaxtyping.Float[Array, 'batch']`

Truncated SNR weighting: min(exp(logsnr), max_weight).

### `uniform_weight`

**function** `uniform_weight(aux: dict[str, jax.Array]) -> jaxtyping.Float[Array, 'batch']`

Uniform (constant 1) weighting.

## `diffusionlab.typing`

Type definitions and short-hands for common data types.

### `DiffusionPredictor`

**class** `DiffusionPredictor(*args, **kwargs)`

Protocol for diffusion model predictors.

### `Prediction`

**class** `Prediction(*, value: jaxtyping.Float[Array, 'batch *#?data_shape'] | jaxtyping.Int[Array, 'batch *#?seq_len'] | jaxtyping.Float[Array, 'batch *#?seq_len num_classes'], kind: str) -> None`

A model prediction with its value and kind.

Attributes:
    value: The predicted data array.
    kind: The type of prediction (e.g. ``'eps'``, ``'x_0'``, ``'v'``). Static (not traced by JAX).

## `diffusionlab.utils.tree_ops`

### `bcast_left`

**function** `bcast_left(x: jax.Array, ndim: int) -> jax.Array`

### `bcast_right`

**function** `bcast_right(x: jax.Array, ndim: int) -> jax.Array`

### `lenient_map`

**function** `lenient_map(f: Callable[..., Any], tree: Any, *rest: Any, is_leaf: Callable[[Any], bool] | None = None)`

Like jax.tree.map but with a lenient structure matching.

The PyTree structure of the output is determined by the structure of `tree`.
The structures of `rest` are used only to determine the leaf values to be
mapped.

Example usage:
  a = [1.0, 2.0]
  b = (5.0, 6.0)
  c = lenient_map(lambda x, y: x+y, a, b)
  # c is [6.0, 8.0]

If one were to use jax.tree.map directly, one would get an error because the
structure of `a` is not the same as the structure of `b`.

Args:
  f: The function to apply to each leaf.
  tree: The tree to map.
  *rest: Additional arguments to pass to fn.
  is_leaf: A function that takes a leaf and returns True if it should be
    mapped. If None, all leaves are mapped.

Returns:
  The tree resulting from applying fn to each leaf in `tree`.

Raises:
  KeyError: If the structures of `tree` and `rest` do not match.

### `tree_map_with_key`

**function** `tree_map_with_key(f: Callable[..., Any], key: jaxtyping.Key[Array, ''], tree: jaxtyping.PyTree, *rest, is_leaf: Callable[[Any], bool] | None = None) -> jaxtyping.PyTree`

Like jax.tree.map but with a separate PRNG key for each leaf.

Args:
  f: The function to apply to each leaf. Takes the key as the first arg, i.e.
    of the form `f(key: PRNGKey, tree_leaf: Any, *rest_leafs: Any) -> Any`.
  key: The PRNG key from which to split all the leaf-keys.
  tree: The tree to map.
  *rest: Additional arguments to pass to f.
  is_leaf: A function that takes a leaf and returns True if it should be
    mapped. If None, all leaves are mapped.

Returns:
  The tree resulting from applying f to each leaf.

Raises:
  KeyError: If the structures of `tree` and `rest` do not match.
