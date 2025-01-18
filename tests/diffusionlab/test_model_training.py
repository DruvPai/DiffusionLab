from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset
import pytest
import warnings
import torch
from torch import nn, optim
from diffusionlab.distributions.gmm import GMMDistribution, IsoHomoGMMDistribution
from diffusionlab.model import DiffusionModel
from diffusionlab.loss import SamplewiseDiffusionLoss
from diffusionlab.samplers import FMSampler, Sampler
from diffusionlab.vector_fields import VectorFieldType


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


@pytest.fixture
def dummy_net():
    return DummyNet()


@pytest.fixture
def dummy_sampler():
    return FMSampler(is_stochastic=False, t_min=0.0, t_max=1.0, L=100)


@pytest.fixture
def dummy_dataset(dummy_sampler):
    means = torch.randn((3, 2)) * 5
    priors = torch.rand(3)
    priors = priors / priors.sum()
    var = torch.tensor(0.5)
    dist = IsoHomoGMMDistribution(dummy_sampler, means, var, priors)
    X_train, y_train = dist.sample(100)
    X_val, y_val = dist.sample(50)
    train_dataloader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=10, shuffle=True
    )
    val_dataloader = DataLoader(TensorDataset(X_val, y_val), batch_size=10)

    return train_dataloader, val_dataloader


@pytest.fixture
def model(dummy_net, dummy_sampler, dummy_dataset):
    def t_loss_weights(t):
        return torch.ones_like(t)

    def t_loss_probs(t):
        e = torch.zeros_like(t)
        e[4] = 1
        return e

    optimizer = optim.Adam(dummy_net.parameters(), lr=1e-4)

    return DiffusionModel(
        net=dummy_net,
        sampler=dummy_sampler,
        vector_field_type=VectorFieldType.EPS,
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.ConstantLR(optimizer, factor=1),
        val_metrics={},
        t_loss_weights=t_loss_weights,
        t_loss_probs=t_loss_probs,
        N_noise_per_sample=2,
    )


def test_model_training(model, dummy_dataset):
    train_dataloader, val_dataloader = dummy_dataset
    x, metadata = next(iter(val_dataloader))
    untrained_loss = model.aggregate_loss(x)
    with pytest.warns(UserWarning):
        trainer = Trainer(max_epochs=100, accelerator="cpu")
        trainer.fit(model, train_dataloader, val_dataloader)
    trained_loss = model.aggregate_loss(x)
    assert trained_loss < untrained_loss / 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_training_cuda(model, dummy_dataset):
    train_dataloader, val_dataloader = dummy_dataset
    x, metadata = next(iter(val_dataloader))
    untrained_loss = model.aggregate_loss(x)
    with pytest.warns(UserWarning):
        trainer = Trainer(max_epochs=100, accelerator="cuda")
        trainer.fit(model, train_dataloader, val_dataloader)
    trained_loss = model.aggregate_loss(x)
    assert trained_loss < untrained_loss / 10
