# 

<div align="center">
  <img src="diffusionlab_logo.svg" alt="DiffusionLab Logo">
  
  <p>
    <a href="https://github.com/DruvPai/DiffusionLab">GitHub</a> • <code>pip install diffusionlab</code> • <a href="https://raw.githubusercontent.com/DruvPai/DiffusionLab/refs/heads/gh-pages/llms.txt"><code>llms.txt</code></a>
  </p>
  
  <img src="https://github.com/druvpai/diffusionlab/actions/workflows/testing.yml/badge.svg" alt="Tests "> • <img src="https://github.com/druvpai/diffusionlab/actions/workflows/linting_formatting.yml/badge.svg" alt="Linting and Formatting">
</div>

## What is DiffusionLab?

<div align="center">
  <p><em><strong>TL;DR: DiffusionLab is a laboratory for quickly and easily experimenting with diffusion models.</strong></em></p>
</div>

<div>
  <p><strong>DiffusionLab IS:</strong></p>
  <ul>
    <li>A lightweight and flexible set of PyTorch APIs for smaller-scale diffusion model training and sampling.</li>
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

## Examples

## Examples

### Example of Sampling 

```python
from diffusionlab.diffusions import OrnsteinUhlenbeckProcess 
from diffusionlab.samplers import DDMSampler
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.vector_fields import VectorField, VectorFieldType

device = ...  # fix a device
N = 10
L = 50
D = 20
t_min = 0.01
t_max = 1.0
diffusion_process = OrnsteinUhlenbeckProcess()
sampler = DDMSampler(diffusion_process, is_stochastic=True)  # DDPM sampler; if is_stochastic==False then it becomes DDIM sampler
scheduler = UniformScheduler()
eps_predictor = get_eps_predictor(diffusion_process)  # This function doesn't exist, but you can get such a predictor by training a NN with signature (N, D*) x (N, ) -> (N, D*)
eps_vf = VectorField(eps_predictor, VectorFieldType.EPS)

ts = scheduler.get_ts(t_min=t_min, t_max=t_max, L=L).to(device)
x_init = torch.randn((N, D), device=device)
zs = torch.randn((L-1, N, D), device=device)
samples = sampler.sample(eps_vf, x_init, zs, ts)
```

### (Long) End-To-End Example of Training and Sampling

The linked file [demo.py](https://github.com/DruvPai/DiffusionLab/blob/main/demo.py) is an end-to-end example of training and sampling from a toy diffusion model on synthetic data.

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
