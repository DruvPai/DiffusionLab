"""Training losses for DiffusionLab."""

from diffusionlab.training.base import (
    Loss,
    LossFn,
    LossOutput,
    MultimodalLoss,
    ce_loss,
    default_loss_fn,
    kl_loss,
    mse_loss,
)
from diffusionlab.training.consistency import ConsistencyLoss
from diffusionlab.training.denoising import DenoisingLoss
from diffusionlab.training.distillation import DistillationLoss
from diffusionlab.training.schedules import (
    logit_normal_time,
    logsnr_uniform_time,
    stratified_time,
    uniform_time,
    uniform_time_with_eps,
)
from diffusionlab.training.weighting import (
    WeightFn,
    inverse_sigma_squared_weight,
    make_weight_fn,
    min_snr_weight,
    p2_weight,
    snr_weight,
    truncated_snr_weight,
    uniform_weight,
)

__all__ = [
    "LossOutput",
    "Loss",
    "LossFn",
    "MultimodalLoss",
    "DenoisingLoss",
    "DistillationLoss",
    "ConsistencyLoss",
    "mse_loss",
    "ce_loss",
    "kl_loss",
    "default_loss_fn",
    "WeightFn",
    "uniform_weight",
    "snr_weight",
    "truncated_snr_weight",
    "min_snr_weight",
    "inverse_sigma_squared_weight",
    "p2_weight",
    "make_weight_fn",
    "uniform_time",
    "uniform_time_with_eps",
    "logit_normal_time",
    "logsnr_uniform_time",
    "stratified_time",
]
