from typing import Callable, Dict, Literal
import torch
from torch import nn, optim
from lightning import LightningModule

from diffusionlab.loss import SamplewiseDiffusionLoss
from diffusionlab.samplers import Sampler
from diffusionlab.vector_fields import VectorField, VectorFieldType


class DiffusionModel(LightningModule, VectorField):
    def __init__(
        self,
        net: nn.Module,
        sampler: Sampler,
        vector_field_type: VectorFieldType,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        val_metrics: Dict[str, nn.Module],
        t_loss_weights: Callable[[torch.Tensor], torch.Tensor],
        t_loss_probs: Callable[[torch.Tensor], torch.Tensor],
        N_noise_per_sample: int,
    ):
        super().__init__()
        self.net: nn.Module = net
        self.vector_field_type: VectorFieldType = vector_field_type
        self.sampler: Sampler = sampler
        self.optimizer: optim.Optimizer = optimizer
        self.scheduler: optim.lr_scheduler.LRScheduler = scheduler
        self.val_metrics: Dict[str, nn.Module] = val_metrics

        self.t_loss_weights: Callable[[torch.Tensor], torch.Tensor] = t_loss_weights
        self.t_loss_probs: Callable[[torch.Tensor], torch.Tensor] = t_loss_probs
        self.N_noise_per_sample: int = N_noise_per_sample

        self.t_loss_weights_precomputed: torch.Tensor = self.t_loss_weights(
            self.sampler.schedule
        )
        self.t_loss_probs_precomputed: torch.Tensor = self.t_loss_probs(
            self.sampler.schedule
        )
        self.samplewise_loss: SamplewiseDiffusionLoss = SamplewiseDiffusionLoss(
            sampler, vector_field_type
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(x, t)

    def configure_optimizers(
        self,
    ) -> Dict[
        Literal["optimizer", "lr_scheduler"],
        optim.Optimizer | optim.lr_scheduler.LRScheduler,
    ]:
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def loss(
        self, x: torch.Tensor, t: torch.Tensor, sample_weights: torch.Tensor
    ) -> torch.Tensor:
        x = torch.repeat_interleave(x, self.N_noise_per_sample, dim=0)
        t = torch.repeat_interleave(t, self.N_noise_per_sample, dim=0)
        sample_weights = torch.repeat_interleave(
            sample_weights, self.N_noise_per_sample, dim=0
        )

        eps = torch.randn_like(x)
        xt = self.sampler.add_noise(x, t, eps)
        fxt = self(xt, t)

        samplewise_loss = self.samplewise_loss(xt, fxt, x, eps, t)
        mean_loss = torch.mean(samplewise_loss * sample_weights)
        return mean_loss

    def aggregate_loss(self, x: torch.Tensor) -> torch.Tensor:
        t_idx = torch.multinomial(
            self.t_loss_probs_precomputed, x.shape[0], replacement=True
        ).to(x.device, non_blocking=True)
        t = self.sampler.schedule.to(x.device, non_blocking=True)[t_idx]
        t_weights = self.t_loss_weights_precomputed.to(x.device, non_blocking=True)[
            t_idx
        ]
        mean_loss = self.loss(x, t, t_weights)
        return mean_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, metadata = batch
        loss = self.aggregate_loss(x)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        x, metadata = batch
        loss = self.aggregate_loss(x)
        metric_values = {
            metric_name: metric(x, metadata, self)
            for metric_name, metric in self.val_metrics.items()
        }
        metric_values["val_loss"] = loss
        self.log_dict(metric_values, on_step=True, on_epoch=True, prog_bar=True)
        return metric_values
