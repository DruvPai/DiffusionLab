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
