
<div style="text-align:center;">

![DiffusionLab Logo](docs/diffusionlab_logo.svg)

[Documentation](https://druvpai.github.io/DiffusionLab) • `pip install diffusionlab` • [`llms.txt`](https://raw.githubusercontent.com/DruvPai/DiffusionLab/refs/heads/gh-pages/llms.txt)

![Tests](https://github.com/druvpai/diffusionlab/actions/workflows/testing.yml/badge.svg) • ![Linting and Formatting](https://github.com/druvpai/diffusionlab/actions/workflows/linting_formatting.yml/badge.svg)

</div>

## What is DiffusionLab?

TL;DR: DiffusionLab is a laboratory for quickly and easily experimenting with diffusion models.
- DiffusionLab IS:
  - A lightweight and flexible set of PyTorch APIs for smaller-scale diffusion model training and sampling.
  - An implementation of the mathematical foundations of diffusion models. 
- DiffusionLab IS NOT:
  - A replacement for HuggingFace Diffusers. 
  - A codebase for SoTA diffusion model training or inference. 
  
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

The file [demo.py](demo.py) is an end-to-end example of training and sampling from a toy diffusion model on synthetic data.



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
