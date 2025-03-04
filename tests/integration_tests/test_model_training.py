import pytest
import torch
from lightning import Trainer
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from diffusionlab.distributions.gmm import IsoHomoGMMDistribution
from diffusionlab.models import DiffusionModel
from diffusionlab.diffusions import FlowMatchingProcess
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.vector_fields import VectorFieldType


class TestModelTraining:
    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(2, 100)
            self.relu1 = nn.ReLU()
            self.w2 = nn.Linear(100, 100)
            self.relu2 = nn.ReLU()
            self.w3 = nn.Linear(100, 2)
            self.t_embed = nn.Parameter(torch.randn((100,)))

        def forward(self, x, t):
            t_embed = t[:, None] * self.t_embed[None, :]
            x = self.w1(x)
            x = self.relu1(x) + t_embed
            x = self.w2(x)
            x = self.relu2(x)
            x = self.w3(x)
            return x

    def test_model_training_without_metrics_cpu(self):
        """Test that the model can be trained on CPU without any metrics."""
        # Create network
        net = self.DummyNet()

        # Create diffusion process
        diffusion_process = FlowMatchingProcess()

        # Create scheduler
        scheduler = UniformScheduler()

        # Create optimizer
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        # Create dataset
        means = torch.randn((3, 2)) * 5
        priors = torch.rand(3)
        priors = priors / priors.sum()
        var = torch.tensor(0.5)
        dist_params = {"means": means, "var": var, "priors": priors}
        dist_hparams = {}
        X_train, y_train = IsoHomoGMMDistribution.sample(100, dist_params, dist_hparams)
        X_val, y_val = IsoHomoGMMDistribution.sample(50, dist_params, dist_hparams)
        train_dataloader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=10, shuffle=True
        )
        val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=10)

        # Create model
        ts_hparams = {"t_min": 0.001, "t_max": 0.99, "L": 100}

        def t_loss_weights(t):
            return torch.ones_like(t)

        def t_loss_probs(t):
            e = torch.zeros_like(t)
            e[4] = 1
            return e

        model = DiffusionModel(
            net=net,
            diffusion_process=diffusion_process,
            train_scheduler=scheduler,
            vector_field_type=VectorFieldType.EPS,
            optimizer=optimizer,
            lr_scheduler=optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            batchwise_metrics={},
            batchfree_metrics={},
            train_ts_hparams=ts_hparams,
            t_loss_weights=t_loss_weights,
            t_loss_probs=t_loss_probs,
            N_noise_draws_per_sample=2,
        )

        # Get validation data for loss comparison
        x, metadata = next(iter(val_dataloader))
        untrained_loss = model.aggregate_loss(x)

        # Train model
        with pytest.warns(UserWarning):
            trainer = Trainer(
                max_epochs=100,
                accelerator="cpu",
            )
            trainer.fit(model, train_dataloader, val_dataloader)

        # Check that loss decreased
        trained_loss = model.aggregate_loss(x)
        assert trained_loss < untrained_loss / 10

    def test_model_training_with_metrics_cpu(self):
        """Test that the model can be trained on CPU with metrics."""
        # Create network
        net = self.DummyNet()

        # Create diffusion process
        diffusion_process = FlowMatchingProcess()

        # Create scheduler
        scheduler = UniformScheduler()

        # Create optimizer
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        # Create dataset
        means = torch.randn((3, 2)) * 5
        priors = torch.rand(3)
        priors = priors / priors.sum()
        var = torch.tensor(0.5)
        dist_params = {"means": means, "var": var, "priors": priors}
        dist_hparams = {}
        X_train, y_train = IsoHomoGMMDistribution.sample(100, dist_params, dist_hparams)
        X_val, y_val = IsoHomoGMMDistribution.sample(50, dist_params, dist_hparams)
        train_dataloader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=10, shuffle=True
        )
        val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=10)

        # Create metrics
        class MSEMetric(nn.Module):
            def forward(self, x, metadata, model):
                # Compute MSE between model prediction and random noise
                t = torch.ones(x.shape[0], device=x.device) * 0.5
                pred = model(x, t)
                target = torch.zeros_like(pred)
                mse = torch.mean((pred - target) ** 2)
                return {"mse": mse}

        class PredictionNormMetric(nn.Module):
            def forward(self, x, metadata, model):
                # Compute norm of model predictions
                t = torch.ones(x.shape[0], device=x.device) * 0.5
                pred = model(x, t)
                norm = torch.mean(torch.norm(pred, dim=1))
                return {"norm": norm}

        class BatchFreeMetric(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, model):
                self.counter += 1
                return {"counter": torch.tensor(self.counter, device=model.device)}

        # Create model
        ts_hparams = {"t_min": 0.001, "t_max": 0.99, "L": 100}

        def t_loss_weights(t):
            return torch.ones_like(t)

        def t_loss_probs(t):
            e = torch.zeros_like(t)
            e[4] = 1
            return e

        model = DiffusionModel(
            net=net,
            diffusion_process=diffusion_process,
            train_scheduler=scheduler,
            vector_field_type=VectorFieldType.EPS,
            optimizer=optimizer,
            lr_scheduler=optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            batchwise_metrics={
                "mse": MSEMetric(),
                "pred_norm": PredictionNormMetric(),
            },
            batchfree_metrics={
                "batch_free": BatchFreeMetric(),
            },
            train_ts_hparams=ts_hparams,
            t_loss_weights=t_loss_weights,
            t_loss_probs=t_loss_probs,
            N_noise_draws_per_sample=2,
        )

        # Get validation data for loss comparison
        x, metadata = next(iter(val_dataloader))
        untrained_loss = model.aggregate_loss(x)

        # Train model
        with pytest.warns(UserWarning):
            trainer = Trainer(
                max_epochs=100,
                accelerator="cpu",
            )
            trainer.fit(model, train_dataloader, val_dataloader)

        # Check that loss decreased
        trained_loss = model.aggregate_loss(x)
        assert trained_loss < untrained_loss / 10

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_training_without_metrics_cuda(self):
        """Test that the model can be trained on CUDA without any metrics."""
        # Create network
        net = self.DummyNet()

        # Create diffusion process
        diffusion_process = FlowMatchingProcess()

        # Create scheduler
        scheduler = UniformScheduler()

        # Create optimizer
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        # Create dataset
        means = torch.randn((3, 2)) * 5
        priors = torch.rand(3)
        priors = priors / priors.sum()
        var = torch.tensor(0.5)
        dist_params = {"means": means, "var": var, "priors": priors}
        dist_hparams = {}
        X_train, y_train = IsoHomoGMMDistribution.sample(100, dist_params, dist_hparams)
        X_val, y_val = IsoHomoGMMDistribution.sample(50, dist_params, dist_hparams)
        train_dataloader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=10, shuffle=True
        )
        val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=10)

        # Create model
        ts_hparams = {"t_min": 0.001, "t_max": 0.99, "L": 100}

        def t_loss_weights(t):
            return torch.ones_like(t)

        def t_loss_probs(t):
            e = torch.zeros_like(t)
            e[4] = 1
            return e

        model = DiffusionModel(
            net=net,
            diffusion_process=diffusion_process,
            train_scheduler=scheduler,
            vector_field_type=VectorFieldType.EPS,
            optimizer=optimizer,
            lr_scheduler=optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            batchwise_metrics={},
            batchfree_metrics={},
            train_ts_hparams=ts_hparams,
            t_loss_weights=t_loss_weights,
            t_loss_probs=t_loss_probs,
            N_noise_draws_per_sample=2,
        )

        # Get validation data for loss comparison
        x, metadata = next(iter(val_dataloader))
        untrained_loss = model.aggregate_loss(x)

        # Train model
        with pytest.warns(UserWarning):
            trainer = Trainer(
                max_epochs=100,
                accelerator="cuda",
            )
            trainer.fit(model, train_dataloader, val_dataloader)

        # Check that loss decreased
        trained_loss = model.aggregate_loss(x)
        assert trained_loss < untrained_loss / 10

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_training_with_metrics_cuda(self):
        """Test that the model can be trained on CUDA with metrics."""
        # Create network
        net = self.DummyNet()

        # Create diffusion process
        diffusion_process = FlowMatchingProcess()

        # Create scheduler
        scheduler = UniformScheduler()

        # Create optimizer
        optimizer = optim.Adam(net.parameters(), lr=1e-4)

        # Create dataset
        means = torch.randn((3, 2)) * 5
        priors = torch.rand(3)
        priors = priors / priors.sum()
        var = torch.tensor(0.5)
        dist_params = {"means": means, "var": var, "priors": priors}
        dist_hparams = {}
        X_train, y_train = IsoHomoGMMDistribution.sample(100, dist_params, dist_hparams)
        X_val, y_val = IsoHomoGMMDistribution.sample(50, dist_params, dist_hparams)
        train_dataloader = DataLoader(
            TensorDataset(X_train, y_train), batch_size=10, shuffle=True
        )
        val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=10)

        # Create metrics
        class MSEMetric(nn.Module):
            def forward(self, x, metadata, model):
                # Compute MSE between model prediction and random noise
                t = torch.ones(x.shape[0], device=x.device) * 0.5
                pred = model(x, t)
                target = torch.zeros_like(pred)
                mse = torch.mean((pred - target) ** 2)
                return {"mse": mse}

        class PredictionNormMetric(nn.Module):
            def forward(self, x, metadata, model):
                # Compute norm of model predictions
                t = torch.ones(x.shape[0], device=x.device) * 0.5
                pred = model(x, t)
                norm = torch.mean(torch.norm(pred, dim=1))
                return {"norm": norm}

        class BatchFreeMetric(nn.Module):
            def __init__(self):
                super().__init__()
                self.counter = 0

            def forward(self, model):
                self.counter += 1
                return {"counter": torch.tensor(self.counter, device=model.device)}

        # Create model
        ts_hparams = {"t_min": 0.001, "t_max": 0.99, "L": 100}

        def t_loss_weights(t):
            return torch.ones_like(t)

        def t_loss_probs(t):
            e = torch.zeros_like(t)
            e[4] = 1
            return e

        model = DiffusionModel(
            net=net,
            diffusion_process=diffusion_process,
            train_scheduler=scheduler,
            vector_field_type=VectorFieldType.EPS,
            optimizer=optimizer,
            lr_scheduler=optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            batchwise_metrics={
                "mse": MSEMetric(),
                "pred_norm": PredictionNormMetric(),
            },
            batchfree_metrics={
                "batch_free": BatchFreeMetric(),
            },
            train_ts_hparams=ts_hparams,
            t_loss_weights=t_loss_weights,
            t_loss_probs=t_loss_probs,
            N_noise_draws_per_sample=2,
        )

        # Get validation data for loss comparison
        x, metadata = next(iter(val_dataloader))
        untrained_loss = model.aggregate_loss(x)

        # Train model
        with pytest.warns(UserWarning):
            trainer = Trainer(
                max_epochs=100,
                accelerator="cuda",
            )
            trainer.fit(model, train_dataloader, val_dataloader)

        # Check that loss decreased
        trained_loss = model.aggregate_loss(x)
        assert trained_loss < untrained_loss / 10
