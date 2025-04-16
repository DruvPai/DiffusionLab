# 

<div align="center">
  <img src="diffusionlab_logo.svg" alt="DiffusionLab Logo">
  
  <p>
    <a href="https://github.com/DruvPai/DiffusionLab">GitHub</a> • <code>pip install diffusionlab</code> • <a href="https://raw.githubusercontent.com/DruvPai/DiffusionLab/refs/heads/gh-pages/llms.txt"><code>llms.txt</code></a>
  </p>
  
  <img src="https://github.com/druvpai/diffusionlab/actions/workflows/testing.yml/badge.svg" alt="PyTest "> • <img src="https://github.com/druvpai/diffusionlab/actions/workflows/linting_formatting.yml/badge.svg" alt="Ruff">
</div>

## What is DiffusionLab?

<div align="center">
  <p><em><strong>TL;DR: DiffusionLab is a laboratory for quickly and easily experimenting with diffusion models.</strong></em></p>
</div>

<div>
  <p><strong>DiffusionLab IS:</strong></p>
  <ul>
    <li>A lightweight and flexible set of Jax APIs for smaller-scale diffusion model training and sampling.</li>
    <li>An implementation of the mathematical foundations of diffusion models.</li>
  </ul>
  
  <p><strong>DiffusionLab IS NOT:</strong></p>
  <ul>
    <li>A replacement for HuggingFace Diffusers.</li>
    <li>A codebase for SoTA diffusion model training or inference.</li>
  </ul>
</div>

<p><strong>Slightly longer description:</strong></p>

When I'm writing code for experimenting with diffusion models at smaller scales (e.g., to do some science or smaller-scale experiments), I often use the same abstractions and code snippets repeatedly. This codebase captures that useful code, making it reproducible. Since my research in this area is more mathematically oriented, the code is too: it focuses on an implementation which is exactly in line with the mathematical framework of diffusion models, while hopefully still being easy to read and extend. New stuff can be added if popular or in high-demand, bonus points if the idea is mathematically clean. Since the codebase is built for smaller scale exploration, I haven't optimized the multi-GPU or multi-node performance.
 
If you want to add a feature in the spirit of the above motivation, or want to make the code more efficient, feel free to make an Issue or Pull Request. I hope this project is useful in your exploration of diffusion models.

## Example

The following code compares three sample sets:
- One drawn from the ground truth distribution, which is a Gaussian mixture model;
- One sampled using DDIM with the ground-truth denoiser for the Gaussian mixture model;
- One sampled using DDIM with the ground-truth denoiser for the _empirical_ distribution of the first sample set.

```python
import jax 
from jax import numpy as jnp, vmap
from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.samplers import DDMSampler
from diffusionlab.distributions.gmm.gmm import GMM
from diffusionlab.distributions.empirical import EmpiricalDistribution
from diffusionlab.vector_fields import VectorFieldType 

key = jax.random.key(1)

dim = 10
num_samples_ground_truth = 100
num_samples_ddim = 50

num_components = 3
priors = jnp.ones(num_components) / num_components
key, subkey = jax.random.split(key)
means = jax.random.normal(subkey, (num_components, dim))
key, subkey = jax.random.split(key)
cov_factors = jax.random.normal(subkey, (num_components, dim, dim))
covs = jax.vmap(lambda A: A @ A.T)(cov_factors)

gmm = GMM(means, covs, priors)

key, subkey = jax.random.split(key)
X_ground_truth, y_ground_truth = gmm.sample(key, num_samples_ground_truth)

num_steps = 100
t_min = 0.001 
t_max = 0.999

diffusion_process = VariancePreservingProcess()
scheduler = UniformScheduler()
ts = scheduler.get_ts(t_min=t_min, t_max=t_max, num_steps=num_steps)

key, subkey = jax.random.split(key)
X_noise = jax.random.normal(subkey, (num_samples_ddim, dim))

zs = jax.random.normal(key, (num_samples_ddim, num_steps, dim))

ground_truth_sampler = DDMSampler(diffusion_process, lambda x, t: gmm.x0(x, t, diffusion_process), VectorFieldType.X0, use_stochastic_sampler=False)
X_ddim_ground_truth = jax.vmap(lambda x_init, z: ground_truth_sampler.sample(x_init, z, ts))(X_noise, zs)

empirical_distribution = EmpiricalDistribution([(X_ground_truth, y_ground_truth)])
empirical_sampler = DDMSampler(diffusion_process, lambda x, t: empirical_distribution.x0(x, t, diffusion_process), VectorFieldType.X0, use_stochastic_sampler=False)
X_ddim_empirical = jax.vmap(lambda x_init, z: empirical_sampler.sample(x_init, z, ts))(X_noise, zs)

min_distance_to_gt_empirical = vmap(lambda x: jnp.min(vmap(lambda x_gt: jnp.linalg.norm(x - x_gt))(X_ground_truth)))(X_ddim_empirical)
min_distance_to_gt_ground_truth = vmap(lambda x: jnp.min(vmap(lambda x_gt: jnp.linalg.norm(x - x_gt))(X_ground_truth)))(X_ddim_ground_truth)

print(f"Min distance to ground truth samples from DDIM samples using empirical denoiser: {min_distance_to_gt_empirical}")
print(f"Min distance to ground truth samples from DDIM samples using ground truth denoiser: {min_distance_to_gt_ground_truth}")
```

## How to Install

### Install via Pip

`pip install diffusionlab`

Requires Python >= 3.12. (If this is an issue, make a GitHub Issue --- the code should be backward-compatible without many changes).

### Install locally

Run `git clone`:
```
git clone https://github.com/DruvPai/DiffusionLab
cd DiffusionLab
```
Then (probably in a `conda` environment or a `venv`) install the codebase as a local Pip package, along with the required dependencies:
```
pip install .
```
Then feel free to use it! The import is `import diffusionlab`. You can see an example usage in `demo.py`.

## Roadmap/TODOs

<ul>
  <li>Add Diffusers-style pipelines for common tasks (e.g., training, sampling)</li>
  <li>Support latent diffusion</li>
  <li>Support conditional diffusion samplers like CFG</li>
  <li>Add patch-based optimal denoiser as in <a href="https://arxiv.org/abs/2411.19339">Niedoba et al</a></li>
</ul>

Version guide:
<ul>
  <li>Major version update (1 -> 2, etc): initial upload or major refactor.</li>
  <li>Minor version update (1.0 -> 1.1 -> 1.2, etc): breaking change or large feature integration or large update.</li>
  <li>Anything smaller (1.0.0 -> 1.0.1 -> 1.0.2, etc): non-breaking change, small feature integration, better documentation, etc.</li>
</ul>

## How to Contribute

Just clone the repository locally using
```
pip install -e ".[dev,docs]"
```
make a new branch, and make a PR when you feel ready. Here are a couple quick guidelines:
<ul>
  <li> If the function involves nontrivial dimension manipulation, please annotate each tensor with its shape in a comment beside its definition. Examples are found throughout the codebase.
  <li> Please add tests for all nontrivial code. Try to keep the coverage as high as possible.
  <li> If you want to add a new package, update the `pyproject.toml` accordingly.
  <li> We use `Ruff` for formatting, Pytest for tests, and `pytest-cov` for code coverage.
</ul>

Here "nontrivial" is left up to your judgement. A good first contribution is to add more integration tests.

## Note on Frameworks

DiffusionLab versions < 3.0 use a PyTorch backbone. Here is a permalink to the [GitHub pages](https://github.com/DruvPai/DiffusionLab/tree/1543db3453c4cc687c724eb0e01f63c109e4465a) and [llms.txt](https://raw.githubusercontent.com/DruvPai/DiffusionLab/1543db3453c4cc687c724eb0e01f63c109e4465a/llms.txt) for the old version.

DiffusionLab versions >= 3.0 use a Jax backbone.

## Citation Information

You can use the following Bibtex:
```
@Misc{pai25diffusionlab,
    author = {Pai, Druv},
    title = {DiffusionLab},
    howpublished = {\url{https://github.com/DruvPai/DiffusionLab}},
    year = {2025}
}
```
Many thanks!
