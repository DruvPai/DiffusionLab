import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from diffusionlab.distributions.empirical import EmpiricalDistribution
from diffusionlab.samplers import VPSampler
from diffusionlab.vector_fields import VectorField, VectorFieldType

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sampler():
    """Create a VP sampler for testing."""
    return VPSampler(is_stochastic=False, t_min=0.001, t_max=0.99, L=100)


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


def test_empirical_sampling_with_sampler(sampler, dummy_data):
    """Test that sampling using VPSampler recovers the training distribution."""
    # Collect all training data
    X_train = []
    y_train = []
    for X_batch, y_batch in dummy_data:
        X_train.append(X_batch)
        y_train.append(y_batch)
    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)

    # Sample using the sampler
    N = 100  # Sample more points for better statistical comparison
    dist_hparams = {"labeled_data": dummy_data}
    x0_est = VectorField(
        lambda x, t: EmpiricalDistribution.x0(x, t, sampler, {}, dist_hparams),
        vector_field_type=VectorFieldType.X0,
    )
    zs = torch.randn((sampler.schedule.shape[0] - 1, N, X_train.shape[-1]))
    x0 = torch.randn((N, X_train.shape[-1]))
    X_sampled = sampler.sample(x0_est, x0, zs)

    # Compute statistics of both distributions
    train_mean = X_train.mean(0)
    train_std = X_train.std(0)
    sampled_mean = X_sampled.mean(0)
    sampled_std = X_sampled.std(0)

    # Verify the sampled distribution matches training distribution
    assert torch.allclose(train_mean, sampled_mean, atol=0.2), (
        f"Means differ: train={train_mean}, sampled={sampled_mean}"
    )
    assert torch.allclose(train_std, sampled_std, atol=0.2), (
        f"Standard deviations differ: train={train_std}, sampled={sampled_std}"
    )

    # Verify samples are close to training points
    # For each sampled point, check if it's close to at least one training point
    dists = torch.cdist(X_sampled, X_train)
    min_dists = dists.min(dim=1).values
    assert torch.mean(min_dists) < 0.5, (
        "Sampled points are too far from training points"
    )


# ============================================================================
# Vector Field Tests
# ============================================================================


def test_empirical_x0_shape(sampler, dummy_data):
    """Test x0 prediction shape."""
    N = 10
    D = 2
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    x0_hat = EmpiricalDistribution.x0(x, t, sampler, {}, {"labeled_data": dummy_data})
    assert x0_hat.shape == (N, D)


def test_empirical_x0_numerical_stability(sampler, dummy_data):
    """Test numerical stability of x0 predictions."""
    N = 10
    D = 2
    x = torch.randn(N, D)

    # Test with different time values
    for t_val in [0.01, 0.5, 0.99]:
        t = torch.ones(N) * t_val
        x0_hat = EmpiricalDistribution.x0(
            x, t, sampler, {}, {"labeled_data": dummy_data}
        )

        # Check for numerical issues
        assert not torch.any(torch.isnan(x0_hat))
        assert not torch.any(torch.isinf(x0_hat))
        assert torch.all(torch.abs(x0_hat) < 100)


def test_empirical_vector_field_types(sampler, dummy_data):
    """Test all vector field types work correctly."""
    N = 10
    D = 2
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    dist_hparams = {"labeled_data": dummy_data}
    dist_params = {}

    # Test each vector field type
    x0_hat = EmpiricalDistribution.x0(x, t, sampler, dist_params, dist_hparams)
    eps_hat = EmpiricalDistribution.eps(x, t, sampler, dist_params, dist_hparams)
    v_hat = EmpiricalDistribution.v(x, t, sampler, dist_params, dist_hparams)
    score_hat = EmpiricalDistribution.score(x, t, sampler, dist_params, dist_hparams)

    # Check shapes
    assert x0_hat.shape == (N, D)
    assert eps_hat.shape == (N, D)
    assert v_hat.shape == (N, D)
    assert score_hat.shape == (N, D)

    # Check consistency between vector field types
    x_from_x0 = sampler.alpha(t)[:, None] * x0_hat + sampler.sigma(t)[:, None] * eps_hat
    assert torch.allclose(x, x_from_x0, rtol=1e-5)


# ============================================================================
# Device Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empirical_device_movement(sampler, dummy_data):
    """Test distribution works with different devices."""
    device = torch.device("cuda:0")

    # Test x0 prediction on CUDA
    N = 10
    D = 2
    x = torch.randn(N, D, device=device)
    t = torch.ones(N, device=device) * 0.5

    x0_hat = EmpiricalDistribution.x0(x, t, sampler, {}, {"labeled_data": dummy_data})
    assert x0_hat.device == device


# ============================================================================
# Error Case Tests
# ============================================================================


def test_empirical_error_cases(sampler, dummy_data):
    """Test error handling."""
    N = 10
    x = torch.randn(N, 3)  # 3D data when distribution expects 2D
    t = torch.ones(N) * 0.5

    # Test x0 with mismatched dimensions
    with pytest.raises(RuntimeError):
        EmpiricalDistribution.x0(x, t, sampler, {}, {"labeled_data": dummy_data})

    # Test x0 with mismatched batch sizes
    x = torch.randn(N, 2)
    t = torch.ones(N + 1) * 0.5  # Different batch size than x
    with pytest.raises(RuntimeError):
        EmpiricalDistribution.x0(x, t, sampler, {}, {"labeled_data": dummy_data})
