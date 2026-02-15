# diffusionlab

> No-frills JAX library providing core abstractions for diffusion models.

## Modules

- [diffusionlab.models.base](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/models/base.py): Abstract base classes for diffusion model architectures.
- [diffusionlab.models.components](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/models/components.py): Shared components for diffusion model architectures.
- [diffusionlab.models.dit](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/models/dit.py): Diffusion Transformer (DiT) architectures for sequence data.
- [diffusionlab.models.mlp](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/models/mlp.py): FiLM-conditioned MLP for diffusion models on flat vector data.
- [diffusionlab.processes.base](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/processes/base.py): Base class for diffusion processes.
- [diffusionlab.processes.discrete](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/processes/discrete.py): Discrete diffusion processes for categorical data.
- [diffusionlab.processes.gaussian](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/processes/gaussian.py): Gaussian diffusion processes.
- [diffusionlab.processes.interpolation](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/processes/interpolation.py)
- [diffusionlab.processes.simplicial](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/processes/simplicial.py): Simplicial diffusion processes for categorical data (Dirichlet corruption).
- [diffusionlab.sampling.base](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/sampling/base.py): Sampler base class, multimodal sampler, and sampling orchestrator.
- [diffusionlab.sampling.continuous](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/sampling/continuous.py): Samplers for Gaussian diffusion processes.
- [diffusionlab.sampling.discrete](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/sampling/discrete.py): Sampler for discrete (categorical) diffusion processes.
- [diffusionlab.sampling.guidance](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/sampling/guidance.py): Guidance transforms for guided sampling.
- [diffusionlab.sampling.schedules](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/sampling/schedules.py): Deterministic time schedules for sampling.
- [diffusionlab.training.base](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/training/base.py): Loss base class, LossOutput, MultimodalLoss, and standalone loss functions.
- [diffusionlab.training.consistency](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/training/consistency.py): Self-consistency / consistency training loss.
- [diffusionlab.training.denoising](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/training/denoising.py): Standard denoising training loss.
- [diffusionlab.training.distillation](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/training/distillation.py): Teacher-student distillation loss.
- [diffusionlab.training.schedules](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/training/schedules.py): Stochastic time schedules for training.
- [diffusionlab.training.weighting](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/training/weighting.py): Standalone weight functions for training losses.
- [diffusionlab.typing](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/typing.py): Type definitions and short-hands for common data types.
- [diffusionlab.utils.tree_ops](https://github.com/druvpai/DiffusionLab/blob/main/diffusionlab/utils/tree_ops.py)
