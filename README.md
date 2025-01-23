# DiffusionLab

Web documentation, including nice logo, coming soon. For now, please read the README.

## What is DiffusionLab?

TL;DR: DiffusionLab is a laboratory for quickly and easily experimenting with diffusion models.
- DiffusionLab IS:
  - A lightweight and flexible set of PyTorch APIs for smaller-scale diffusion model training and sampling.
  - An implementation of the mathematical foundations of diffusion models. 
- DiffusionLab IS NOT:
  - A replacement for HuggingFace Diffusers. 
  - A codebase for SoTA diffusion model training or inference. 

Slightly longer description:

When I'm writing code for experimenting with diffusion models at smaller scales (e.g., to do some science or smaller-scale experiments), I often use the same abstractions and code snippets repeatedly. This codebase captures that useful code, making it reproducible. Since my research in this area is more mathematically oriented, the code is too: it focuses on an implementation which is exactly in line with the mathematical framework of diffusion models, while hopefully still being easy to read and extend. New stuff can be added if popular or in high-demand, bonus points if the idea is mathematically clean. Since the codebase is built for smaller scale exploration, I haven't optimized the multi-GPU or multi-node performance.
 
If you want to add a feature in the spirit of the above motivation, or want to make the code more efficient, feel free to make an Issue or Pull Request. I hope this project is useful in your exploration of diffusion models.

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

- Add Diffusers-style pipelines for common tasks (e.g., training, sampling)
- Support latent diffusion
- Support conditional diffusion samplers like CFG
- Add patch-based optimal denoiser as in [Niedoba et al](https://arxiv.org/abs/2411.19339)

Version guide:
- Major version update (1 -> 2, etc): initial upload or major refactor.
- Minor version update (1.0 -> 1.1 -> 1.2, etc): breaking change or large feature integration or large update.
- Anything smaller (1.0.0 -> 1.0.1 -> 1.0.2, etc): non-breaking change, small feature integration, better documentation, etc.

## How to Contribute

Just clone the repository locally using
```
pip install -e .
```
make a new branch, and make a PR when you feel ready. Here are a couple quick guidelines:
- If the function involves nontrivial dimension manipulation, please annotate each tensor with its shape in a comment beside its definition. Examples are found throughout the codebase.
- Please add tests for all nontrivial code.
- If you want to add a new package, update the `pyproject.toml` accordingly.
- We use `Ruff` for formatting.

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
