import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from diffusionlab.distributions.empirical import EmpiricalDistribution
from diffusionlab.diffusions import DiffusionProcess

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def diffusion_process():
    """Create a diffusion process for testing."""
    return DiffusionProcess(
        alpha=lambda t: torch.cos((t * torch.pi) / 2),
        sigma=lambda t: torch.sin((t * torch.pi) / 2),
    )


@pytest.fixture
def ts_hparams():
    """Create timestep params for testing."""
    return {"t_min": 0.001, "t_max": 0.99, "L": 100}


@pytest.fixture
def dummy_data():
    """Create a simple 2D dataset with 2 clusters for testing."""
    N = 100
    D = 2

    # First cluster
    X1 = torch.randn(N // 2, D) * 0.1 + torch.tensor([1.0, 1.0])
    y1 = torch.zeros(N // 2)

    # Second cluster
    X2 = torch.randn(N // 2, D) * 0.1 + torch.tensor([-1.0, -1.0])
    y2 = torch.ones(N // 2)

    # Combine clusters
    X = torch.cat([X1, X2])
    y = torch.cat([y1, y2])

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)


# ============================================================================
# Validation Tests
# ============================================================================


def test_empirical_validation():
    """Test validation of hyperparameters for EmpiricalDistribution."""
    # Test missing data
    with pytest.raises(AssertionError):
        EmpiricalDistribution.validate_hparams({})

    # Test invalid data type
    with pytest.raises(AssertionError):
        EmpiricalDistribution.validate_hparams({"labeled_data": None})

    # Test empty data loader
    empty_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
    empty_loader = DataLoader(empty_dataset, batch_size=1)
    with pytest.raises(AssertionError):
        EmpiricalDistribution.validate_hparams({"labeled_data": empty_loader})


# ============================================================================
# Sampling Tests
# ============================================================================


@pytest.mark.skip(
    reason="This test requires a sampler implementation which is not part of this update"
)
def test_empirical_sampling_with_sampler(sampler, dummy_data, ts_hparams):
    """Test that sampling using VPSampler recovers the training distribution."""
    # This test is skipped because it requires a sampler implementation
    # which is not part of the current update to use diffusion instead of sampler
    pass


# ============================================================================
# Vector Field Tests
# ============================================================================


def test_empirical_x0_shape(diffusion_process, dummy_data):
    """Test x0 prediction shape for empirical distribution."""
    N = 10
    D = 2
    x_t = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    x0_hat = EmpiricalDistribution.x0(
        x_t, t, diffusion_process, {}, {"labeled_data": dummy_data}
    )
    assert x0_hat.shape == (N, D)


def test_empirical_x0_numerical_stability(diffusion_process, dummy_data):
    """Test numerical stability in edge cases for empirical distribution."""
    N = 10
    D = 2
    x_t = torch.randn(N, D)
    torch.ones(N) * 0.5

    # Test with very small sigma
    t_small = torch.ones(N) * 0.001
    x0_hat_small = EmpiricalDistribution.x0(
        x_t, t_small, diffusion_process, {}, {"labeled_data": dummy_data}
    )
    assert not torch.any(torch.isnan(x0_hat_small))
    assert not torch.any(torch.isinf(x0_hat_small))

    # Test with very large sigma
    t_large = torch.ones(N) * 0.999
    x0_hat_large = EmpiricalDistribution.x0(
        x_t, t_large, diffusion_process, {}, {"labeled_data": dummy_data}
    )
    assert not torch.any(torch.isnan(x0_hat_large))
    assert not torch.any(torch.isinf(x0_hat_large))


def test_empirical_vector_field_types(diffusion_process, dummy_data):
    """Test vector field type conversions for empirical distribution."""
    N = 10
    D = 2
    x_t = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    dist_params = {}
    dist_hparams = {"labeled_data": dummy_data}

    # Test each vector field type
    x0_hat = EmpiricalDistribution.x0(
        x_t, t, diffusion_process, dist_params, dist_hparams
    )
    eps_hat = EmpiricalDistribution.eps(
        x_t, t, diffusion_process, dist_params, dist_hparams
    )
    v_hat = EmpiricalDistribution.v(
        x_t, t, diffusion_process, dist_params, dist_hparams
    )
    score_hat = EmpiricalDistribution.score(
        x_t, t, diffusion_process, dist_params, dist_hparams
    )

    # Check shapes
    assert x0_hat.shape == (N, D)
    assert eps_hat.shape == (N, D)
    assert v_hat.shape == (N, D)
    assert score_hat.shape == (N, D)

    # Check consistency
    x_from_x0 = (
        diffusion_process.alpha(t)[:, None] * x0_hat
        + diffusion_process.sigma(t)[:, None] * eps_hat
    )
    assert torch.allclose(x_t, x_from_x0, rtol=1e-5)


# ============================================================================
# Device Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empirical_device_movement(diffusion_process, dummy_data):
    """Test empirical distribution works with different devices."""
    device = torch.device("cuda:0")
    N = 10
    D = 2
    x_t = torch.randn(N, D, device=device)
    t = torch.ones(N, device=device) * 0.5

    x0_hat = EmpiricalDistribution.x0(
        x_t, t, diffusion_process, {}, {"labeled_data": dummy_data}
    )
    assert x0_hat.device == device


# ============================================================================
# Error Case Tests
# ============================================================================


def test_empirical_error_cases(diffusion_process, dummy_data):
    """Test error cases for empirical distribution."""
    N = 10
    D = 2
    x_t = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    # Test with mismatched batch sizes
    with pytest.raises(RuntimeError):
        t_wrong = torch.ones(N + 1) * 0.5
        EmpiricalDistribution.x0(
            x_t, t_wrong, diffusion_process, {}, {"labeled_data": dummy_data}
        )

    # Test with mismatched dimensions
    with pytest.raises(RuntimeError):
        x_wrong = torch.randn(N, D + 1)
        EmpiricalDistribution.x0(
            x_wrong, t, diffusion_process, {}, {"labeled_data": dummy_data}
        )
