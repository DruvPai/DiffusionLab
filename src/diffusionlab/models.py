from typing import Any, Callable, Dict
import torch
from torch import nn, optim
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from diffusionlab.losses import SamplewiseDiffusionLoss
from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.schedulers import Scheduler
from diffusionlab.vector_fields import VectorField, VectorFieldType


class DiffusionModel(LightningModule, VectorField):
    """
    A PyTorch Lightning module for training and evaluating diffusion models.

    This class implements a diffusion model that can be trained using various vector field types
    (score, x0, eps, v) and diffusion processes. It handles the training loop, loss computation,
    and evaluation metrics.

    The model inherits from both LightningModule (for training) and VectorField (for sampling),
    making it compatible with both the Lightning training framework and the diffusion sampling
    algorithms.

    Attributes:
        LOG_ON_STEP (bool): Whether to log metrics on each step.
        LOG_ON_EPOCH (bool): Whether to log metrics on each epoch.
        LOG_ON_PROGRESS_BAR (bool): Whether to display metrics on the progress bar.
        net (nn.Module): The neural network that predicts the vector field.
        vector_field_type (VectorFieldType): The type of vector field the model predicts.
        diffusion_process (DiffusionProcess): The diffusion process used for training.
        train_scheduler (Scheduler): The scheduler for generating training time steps.
        optimizer (optim.Optimizer): The optimizer for training the model.
        lr_scheduler (optim.lr_scheduler.LRScheduler): The learning rate scheduler.
        batchwise_metrics (nn.ModuleDict): Metrics computed on each batch during validation.
        batchfree_metrics (nn.ModuleDict): Metrics computed at the end of validation epoch.
        t_loss_weights (Callable): Function that weights loss at different time steps.
        t_loss_probs (Callable): Function that determines sampling probability of time steps.
        N_noise_per_sample (int): Number of noise samples per data point.
        samplewise_loss (SamplewiseDiffusionLoss): Loss function for each sample.
        train_ts (torch.Tensor): Precomputed time steps for training.
        train_ts_loss_weights (torch.Tensor): Precomputed weights for each time step.
        train_ts_loss_probs (torch.Tensor): Precomputed sampling probabilities for each time step.
    """

    LOG_ON_STEP = True
    LOG_ON_EPOCH = True
    LOG_ON_PROGRESS_BAR = True

    def __init__(
        self,
        net: nn.Module,
        diffusion_process: DiffusionProcess,
        train_scheduler: Scheduler,
        vector_field_type: VectorFieldType,
        optimizer: optim.Optimizer,
        lr_scheduler: optim.lr_scheduler.LRScheduler,
        batchwise_metrics: Dict[str, nn.Module],
        batchfree_metrics: Dict[str, nn.Module],
        train_ts_hparams: Dict[str, Any],
        t_loss_weights: Callable[[torch.Tensor], torch.Tensor],
        t_loss_probs: Callable[[torch.Tensor], torch.Tensor],
        N_noise_draws_per_sample: int,
    ):
        """
        Initialize the diffusion model.

        Args:
            net (nn.Module): Neural network that predicts the vector field.
            diffusion_process (DiffusionProcess): The diffusion process used for training.
            train_scheduler (Scheduler): Scheduler for generating training time steps.
            vector_field_type (VectorFieldType): Type of vector field the model predicts.
            optimizer (optim.Optimizer): Optimizer for training the model.
            lr_scheduler (optim.lr_scheduler.LRScheduler): Learning rate scheduler.
            batchwise_metrics (Dict[str, nn.Module]): Metrics computed on each batch during validation. Each metric takes in (x, metadata, model) and returns a dictionary of metric (name, value) pairs.
            batchfree_metrics (Dict[str, nn.Module]): Metrics computed at the end of validation epoch. Each metric takes in (model) and returns a dictionary of metric (name, value) pairs.
            train_ts_hparams (Dict[str, Any]): Parameters for the training time step scheduler.
            t_loss_weights (Callable[[torch.Tensor], torch.Tensor]): Function that weights loss at different time steps.
            t_loss_probs (Callable[[torch.Tensor], torch.Tensor]): Function that determines sampling probability of time steps.
            N_noise_draws_per_sample (int): Number of noise draws per data point.
        """
        super().__init__()
        self.net: nn.Module = net
        self.vector_field_type: VectorFieldType = vector_field_type
        self.diffusion_process: DiffusionProcess = diffusion_process
        self.train_scheduler: Scheduler = train_scheduler
        self.optimizer: optim.Optimizer = optimizer
        self.lr_scheduler: optim.lr_scheduler.LRScheduler = lr_scheduler
        self.batchwise_metrics: nn.ModuleDict = nn.ModuleDict(batchwise_metrics)
        self.batchfree_metrics: nn.ModuleDict = nn.ModuleDict(batchfree_metrics)

        self.t_loss_weights: Callable[[torch.Tensor], torch.Tensor] = t_loss_weights
        self.t_loss_probs: Callable[[torch.Tensor], torch.Tensor] = t_loss_probs
        self.N_noise_draws_per_sample: int = N_noise_draws_per_sample

        self.samplewise_loss: SamplewiseDiffusionLoss = SamplewiseDiffusionLoss(
            diffusion_process, vector_field_type
        )

        self.register_buffer("train_ts", torch.zeros((0,)))
        self.register_buffer("train_ts_loss_weights", torch.zeros((0,)))
        self.register_buffer("train_ts_loss_probs", torch.zeros((0,)))
        self.precompute_train_schedule(train_ts_hparams)

    def precompute_train_schedule(self, train_ts_hparams: Dict[str, float]) -> None:
        """
        Precompute time steps and their associated weights for training.

        This method generates the time steps used during training and computes
        the loss weights and sampling probabilities for each time step.

        Args:
            train_ts_hparams (Dict[str, float]): Parameters for the training time step scheduler.
                Typically includes t_min, t_max, and the number of steps L.
        """
        self.train_ts = self.train_scheduler.get_ts(**train_ts_hparams).to(
            self.device, non_blocking=True
        )
        self.train_ts_loss_weights: torch.Tensor = self.t_loss_weights(self.train_ts)
        self.train_ts_loss_probs: torch.Tensor = self.t_loss_probs(self.train_ts)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Passes the input through the neural network to predict the vector field.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *data_dims).
            t (torch.Tensor): Time tensor of shape (batch_size,).

        Returns:
            torch.Tensor: Predicted vector field of shape (batch_size, *data_dims).
        """
        return self.net(x, t)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure optimizers and learning rate schedulers for training.

        This method is called by PyTorch Lightning to set up the optimization process.

        Returns:
            OptimizerLRScheduler: Dictionary containing the optimizer and learning rate scheduler.
        """
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}

    def loss(
        self, x: torch.Tensor, t: torch.Tensor, sample_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the loss for a batch of data at specified time steps.

        This method:
        1. Repeats each sample N_noise_per_sample times
        2. Adds noise to the data according to the diffusion process
        3. Predicts the vector field
        4. Computes the loss between the prediction and the ground truth

        Args:
            x (torch.Tensor): Input data of shape (batch_size, *data_dims).
            t (torch.Tensor): Time steps of shape (batch_size,).
            sample_weights (torch.Tensor): Weights for each sample of shape (batch_size,).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        x = torch.repeat_interleave(x, self.N_noise_draws_per_sample, dim=0)
        t = torch.repeat_interleave(t, self.N_noise_draws_per_sample, dim=0)
        sample_weights = torch.repeat_interleave(
            sample_weights, self.N_noise_draws_per_sample, dim=0
        )

        eps = torch.randn_like(x)
        xt = self.diffusion_process.forward(x, t, eps)
        fxt = self(xt, t)

        samplewise_loss = self.samplewise_loss(xt, fxt, x, eps, t)
        mean_loss = torch.mean(samplewise_loss * sample_weights)
        return mean_loss

    def aggregate_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for a batch of data with randomly sampled time steps.

        This method:
        1. Samples time steps according to the training distribution
        2. Computes the loss at those time steps

        Args:
            x (torch.Tensor): Input data of shape (batch_size, *data_dims).

        Returns:
            torch.Tensor: Scalar loss value.
        """
        t_idx = torch.multinomial(
            self.train_ts_loss_probs, x.shape[0], replacement=True
        ).to(self.device, non_blocking=True)
        t = self.train_ts[t_idx]
        t_weights = self.train_ts_loss_weights[t_idx]
        mean_loss = self.loss(x, t, t_weights)
        return mean_loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """
        Perform a single training step.

        This method is called by PyTorch Lightning during training.

        Args:
            batch (torch.Tensor): Batch of data, typically a tuple (x, metadata).
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the batch.
        """
        x, metadata = batch
        loss = self.aggregate_loss(x)
        self.log(
            "train_loss",
            loss,
            on_step=self.LOG_ON_STEP,
            on_epoch=self.LOG_ON_EPOCH,
            prog_bar=self.LOG_ON_PROGRESS_BAR,
        )
        return loss

    def validation_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single validation step.

        This method is called by PyTorch Lightning during validation.
        It computes the loss and any batch-wise metrics.

        Args:
            batch (torch.Tensor): Batch of data, typically a tuple (x, metadata).
            batch_idx (int): Index of the current batch.

        Returns:
            Dict[str, torch.Tensor]: Dictionary of metric values.
        """
        x, metadata = batch
        loss = self.aggregate_loss(x)
        metric_values = {"val_loss": loss}
        for metric_name, metric in self.batchwise_metrics.items():
            metric_values_dict = metric(x, metadata, self)
            for key, value in metric_values_dict.items():
                metric_values[f"{metric_name}_{key}"] = value
        self.log_dict(
            metric_values,
            on_step=self.LOG_ON_STEP,
            on_epoch=self.LOG_ON_EPOCH,
            prog_bar=self.LOG_ON_PROGRESS_BAR,
        )
        return metric_values

    def on_validation_epoch_end(self) -> None:
        """
        Perform operations at the end of a validation epoch.

        This method is called by PyTorch Lightning at the end of each validation epoch.
        It computes and logs any batch-free metrics that require the entire validation set.
        """
        metric_values = {}
        for metric_name, metric in self.batchfree_metrics.items():
            metric_values_dict = metric(self)
            for key, value in metric_values_dict.items():
                metric_values[f"{metric_name}_{key}"] = value
        self.log_dict(
            metric_values,
            on_step=self.LOG_ON_STEP,
            on_epoch=self.LOG_ON_EPOCH,
            prog_bar=self.LOG_ON_PROGRESS_BAR,
        )
