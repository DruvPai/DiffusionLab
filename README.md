
<div style="text-align:center;">

![DiffusionLab Logo](https://raw.githubusercontent.com/DruvPai/DiffusionLab/refs/heads/main/docs/diffusionlab_logo.svg)

[Documentation](https://druvpai.github.io/DiffusionLab) • `pip install diffusionlab` • [`llms.txt`](https://raw.githubusercontent.com/DruvPai/DiffusionLab/refs/heads/gh-pages/llms.txt)

![Tests](https://github.com/druvpai/diffusionlab/actions/workflows/testing.yml/badge.svg) • ![Linting and Formatting](https://github.com/druvpai/diffusionlab/actions/workflows/linting_formatting.yml/badge.svg)

</div>

## What is DiffusionLab?

TL;DR: DiffusionLab is a laboratory for quickly and easily experimenting with diffusion models.
- DiffusionLab IS:
  - A lightweight and flexible set of JAX APIs for smaller-scale diffusion model training and sampling.
  - An implementation of the mathematical foundations of diffusion models. 
- DiffusionLab IS NOT:
  - A replacement for HuggingFace Diffusers. 
  - A codebase for SoTA diffusion model training or inference. 

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

## Note on Frameworks

DiffusionLab versions < 3.0 use a PyTorch backbone. Here is a permalink to the [GitHub pages](https://github.com/DruvPai/DiffusionLab/tree/1543db3453c4cc687c724eb0e01f63c109e4465a) and [llms.txt](https://raw.githubusercontent.com/DruvPai/DiffusionLab/1543db3453c4cc687c724eb0e01f63c109e4465a/llms.txt) for the old version.

DiffusionLab versions >= 3.0 use a JAX backbone.


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
