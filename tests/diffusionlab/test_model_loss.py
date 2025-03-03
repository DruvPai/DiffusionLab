import pytest
import torch
from torch import nn, optim
from diffusionlab.models import DiffusionModel
from diffusionlab.losses import SamplewiseDiffusionLoss
from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.vector_fields import VectorFieldType


class DummyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.randn(1))

    def forward(self, x, t):
        return x  # Identity for testing


class DummyDiffusionProcess(DiffusionProcess):
    def __init__(self):
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        super().__init__(alpha=alpha, sigma=sigma)


@pytest.fixture
def dummy_net():
    return DummyNet()


@pytest.fixture
def dummy_diffusion_process():
    return DummyDiffusionProcess()


@pytest.fixture
def dummy_scheduler():
    return UniformScheduler()


@pytest.fixture
def ts_hparams():
    return {"t_min": 0.001, "t_max": 0.99, "L": 100}


@pytest.fixture
def model(dummy_net, dummy_diffusion_process, dummy_scheduler, ts_hparams):
    def t_loss_weights(t):
        return torch.ones_like(t)

    def t_loss_probs(t):
        return torch.ones_like(t) / len(t)

    return DiffusionModel(
        net=dummy_net,
        diffusion_process=dummy_diffusion_process,
        train_scheduler=dummy_scheduler,
        vector_field_type=VectorFieldType.X0,
        optimizer=optim.Adam(dummy_net.parameters(), lr=1e-4),
        lr_scheduler=optim.lr_scheduler.StepLR(
            optim.Adam(dummy_net.parameters(), lr=1e-4), step_size=1
        ),
        batchwise_metrics={},
        batchfree_metrics={},
        train_ts_hparams=ts_hparams,
        t_loss_weights=t_loss_weights,
        t_loss_probs=t_loss_probs,
        N_noise_draws_per_sample=2,
    )


def test_loss_shape():
    """Test that the loss function returns correct shapes"""
    diffusion_process = DummyDiffusionProcess()
    loss_fn = SamplewiseDiffusionLoss(diffusion_process, VectorFieldType.X0)

    batch_size = 4
    channels = 3
    height = width = 32

    xt = torch.randn(batch_size, channels, height, width)
    fxt = torch.randn(batch_size, channels, height, width)
    x0 = torch.randn(batch_size, channels, height, width)
    eps = torch.randn(batch_size, channels, height, width)
    t = torch.rand(batch_size)

    loss = loss_fn(xt, fxt, x0, eps, t)
    assert loss.shape == (batch_size,), "Loss should return batch-wise scalar values"


def test_model_forward():
    """Test that the model forward pass works with correct shapes"""
    net = DummyNet()
    ts_hparams = {"t_min": 0.0, "t_max": 1.0, "L": 100}
    model = DiffusionModel(
        net=net,
        diffusion_process=DummyDiffusionProcess(),
        train_scheduler=UniformScheduler(),
        vector_field_type=VectorFieldType.X0,
        optimizer=optim.Adam(net.parameters(), lr=1e-4),
        lr_scheduler=optim.lr_scheduler.StepLR(
            optim.Adam(net.parameters(), lr=1e-4), step_size=1
        ),
        batchwise_metrics={},
        batchfree_metrics={},
        train_ts_hparams=ts_hparams,
        t_loss_weights=lambda t: torch.ones_like(t),
        t_loss_probs=lambda t: torch.ones_like(t) / len(t),
        N_noise_draws_per_sample=2,
    )

    batch_size = 4
    channels = 3
    height = width = 32

    x = torch.randn(batch_size, channels, height, width)
    t = torch.rand(batch_size)

    output = model(x, t)
    assert output.shape == x.shape, "Model output shape should match input shape"


def test_model_loss():
    """Test that the model loss computation works"""
    net = DummyNet()
    ts_hparams = {"t_min": 0.0, "t_max": 1.0, "L": 100}
    model = DiffusionModel(
        net=net,
        diffusion_process=DummyDiffusionProcess(),
        train_scheduler=UniformScheduler(),
        vector_field_type=VectorFieldType.X0,
        optimizer=optim.Adam(net.parameters(), lr=1e-4),
        lr_scheduler=optim.lr_scheduler.StepLR(
            optim.Adam(net.parameters(), lr=1e-4), step_size=1
        ),
        batchwise_metrics={},
        batchfree_metrics={},
        train_ts_hparams=ts_hparams,
        t_loss_weights=lambda t: torch.ones_like(t),
        t_loss_probs=lambda t: torch.ones_like(t) / len(t),
        N_noise_draws_per_sample=2,
    )

    batch_size = 4
    channels = 3
    height = width = 32

    x = torch.randn(batch_size, channels, height, width)
    t = torch.rand(batch_size)
    weights = torch.ones(batch_size)

    loss = model.loss(x, t, weights)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"


def test_loss_different_targets():
    """Test that loss works with different vector field types"""
    diffusion_process = DummyDiffusionProcess()
    batch_size = 4
    channels = 3
    height = width = 32

    xt = torch.randn(batch_size, channels, height, width)
    fxt = torch.randn(batch_size, channels, height, width)
    x0 = torch.randn(batch_size, channels, height, width)
    eps = torch.randn(batch_size, channels, height, width)
    t = torch.rand(batch_size)

    for vf_type in [VectorFieldType.X0, VectorFieldType.EPS, VectorFieldType.V]:
        loss_fn = SamplewiseDiffusionLoss(diffusion_process, vf_type)
        loss = loss_fn(xt, fxt, x0, eps, t)
        assert loss.shape == (batch_size,), f"Loss shape incorrect for {vf_type}"
        assert torch.all(loss >= 0), f"Loss should be non-negative for {vf_type}"
