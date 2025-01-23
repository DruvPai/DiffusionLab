import torch
import pytest
from diffusionlab.samplers import Sampler, VESampler, VPSampler, FMSampler
from diffusionlab.vector_fields import VectorField, VectorFieldType


def test_sampler_initialization():
    # Test basic sampler initialization
    alpha = lambda t: torch.ones_like(t)
    sigma = lambda t: t
    schedule_params = {"t_min": 0.0, "t_max": 1.0, "L": 100}

    sampler = Sampler(True, alpha, sigma, schedule_params)
    assert sampler.is_stochastic
    assert len(sampler.schedule) == schedule_params["L"]
    assert torch.allclose(sampler.schedule[0], torch.tensor(1.0))
    assert torch.allclose(sampler.schedule[-1], torch.tensor(0.0))


def test_sampler_add_noise():
    # Test noise addition
    alpha = lambda t: torch.ones_like(t)
    sigma = lambda t: t
    schedule_params = {"t_min": 0.0, "t_max": 1.0, "L": 100}

    sampler = Sampler(True, alpha, sigma, schedule_params)

    batch_size = 10
    data_dim = 3
    x = torch.randn(batch_size, data_dim)
    t = torch.ones(batch_size) * 0.5
    eps = torch.randn(batch_size, data_dim)

    noisy_x = sampler.add_noise(x, t, eps)
    assert noisy_x.shape == x.shape

    # Test consistency with forward process equation
    expected = alpha(t).unsqueeze(-1) * x + sigma(t).unsqueeze(-1) * eps
    assert torch.allclose(noisy_x, expected)


def test_ve_sampler():
    # Test VE sampler initialization and properties
    sampler = VESampler(True, 0.0, 1.0, 100)

    # Test alpha and sigma functions
    t = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(sampler.alpha(t), torch.ones_like(t))
    assert torch.allclose(sampler.sigma(t), t)
    assert torch.allclose(sampler.alpha_prime(t), torch.zeros_like(t))
    assert torch.allclose(sampler.sigma_prime(t), torch.ones_like(t))

    # Test noise addition
    batch_size = 10
    data_dim = 3
    x = torch.randn(batch_size, data_dim)
    t = torch.ones(batch_size) * 0.5
    eps = torch.randn(batch_size, data_dim)

    noisy_x = sampler.add_noise(x, t, eps)
    expected = x + t.unsqueeze(-1) * eps
    assert torch.allclose(noisy_x, expected)


def test_vp_sampler():
    # Test VP sampler initialization and properties
    sampler = VPSampler(True, 0.0, 1.0, 100)

    # Test alpha and sigma functions
    t = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(sampler.alpha(t), torch.sqrt(1 - t**2))
    assert torch.allclose(sampler.sigma(t), t)
    assert torch.allclose(sampler.alpha_prime(t), -t / torch.sqrt(1 - t**2))
    assert torch.allclose(sampler.sigma_prime(t), torch.ones_like(t))

    # Test noise addition
    batch_size = 10
    data_dim = 3
    x = torch.randn(batch_size, data_dim)
    t = torch.ones(batch_size) * 0.5
    eps = torch.randn(batch_size, data_dim)

    noisy_x = sampler.add_noise(x, t, eps)
    expected = torch.sqrt(1 - t**2).unsqueeze(-1) * x + t.unsqueeze(-1) * eps
    assert torch.allclose(noisy_x, expected)


def test_fm_sampler():
    # Test FM sampler initialization and properties
    sampler = FMSampler(True, 0.0, 1.0, 100)

    # Test alpha and sigma functions
    t = torch.tensor([0.0, 0.5, 1.0])
    assert torch.allclose(sampler.alpha(t), 1 - t)
    assert torch.allclose(sampler.sigma(t), t)
    assert torch.allclose(sampler.alpha_prime(t), -torch.ones_like(t))
    assert torch.allclose(sampler.sigma_prime(t), torch.ones_like(t))

    # Test noise addition
    batch_size = 10
    data_dim = 3
    x = torch.randn(batch_size, data_dim)
    t = torch.ones(batch_size) * 0.5
    eps = torch.randn(batch_size, data_dim)

    noisy_x = sampler.add_noise(x, t, eps)
    expected = (1 - t).unsqueeze(-1) * x + t.unsqueeze(-1) * eps
    assert torch.allclose(noisy_x, expected)


def test_sampler_with_score():
    # Test sampling with score function
    sampler = VPSampler(True, 0.01, 0.99, 10)

    batch_size = 10
    data_dim = 3
    num_steps = 10

    # Create a simple score function
    def score_fn(x, t):
        return -x  # Simple score for standard normal

    score = VectorField(score_fn, VectorFieldType.SCORE)

    # Test sampling
    x0 = torch.randn(batch_size, data_dim)
    zs = torch.randn(num_steps - 1, batch_size, data_dim)  # L-1 noise vectors

    # Test single sample
    x = sampler.sample(score, x0, zs)
    assert x.shape == (batch_size, data_dim)

    # Test trajectory
    xs = sampler.sample_trajectory(score, x0, zs)

    assert xs.shape == (num_steps, batch_size, data_dim)  # L steps
    assert torch.allclose(xs[0], x0)  # First step should be x0
    assert torch.allclose(xs[-1], x)  # Last step should match single sample


def test_invalid_schedule_params():
    lambda t: torch.ones_like(t)

    # Test invalid t_min, t_max
    with pytest.raises(AssertionError):
        VPSampler(True, -0.1, 1.0, 100)

    with pytest.raises(AssertionError):
        VPSampler(True, 0.0, 1.1, 100)

    with pytest.raises(AssertionError):
        VPSampler(True, 0.5, 0.3, 100)  # t_min > t_max
