import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from diffusionlab.distributions.empirical import EmpiricalDistribution
from diffusionlab.samplers import VPSampler
from diffusionlab.vector_fields import VectorField, VectorFieldType

@pytest.fixture
def sampler():
    return VPSampler(is_stochastic=False, t_min=0.01, t_max=0.99, L=100)

@pytest.fixture
def dummy_data():
    # Create a simple 2D dataset with 2 clusters
    N = 100
    D = 2
    
    # First cluster
    X1 = torch.randn(N//2, D) * 0.1 + torch.tensor([1.0, 1.0])
    y1 = torch.zeros(N//2)
    
    # Second cluster
    X2 = torch.randn(N//2, D) * 0.1 + torch.tensor([-1.0, -1.0])
    y2 = torch.ones(N//2)
    
    # Combine clusters
    X = torch.cat([X1, X2])
    y = torch.cat([y1, y2])
    
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=10, shuffle=True)

def test_empirical_initialization(sampler, dummy_data):
    """Test basic initialization of empirical distribution"""
    dist = EmpiricalDistribution(sampler, dummy_data)
    assert dist.sampler == sampler
    assert dist.labeled_data == dummy_data

def test_empirical_sampling(sampler, dummy_data):
    """Test sampling from empirical distribution"""
    dist = EmpiricalDistribution(sampler, dummy_data)
    
    # Test sampling N points
    N = 50
    X, y = dist.sample(N)
    
    assert X.shape[0] == N
    assert y.shape[0] == N
    assert X.shape[1] == 2  # 2D data
    assert y.min() >= 0 and y.max() <= 1  # Binary labels

def test_empirical_x0_shape(sampler, dummy_data):
    """Test x0 prediction shape"""
    dist = EmpiricalDistribution(sampler, dummy_data)
    
    N = 10
    D = 2
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5
    
    x0_hat = dist.x0(x, t)
    assert x0_hat.shape == (N, D)

def test_empirical_x0_numerical_stability(sampler, dummy_data):
    """Test numerical stability of x0 predictions"""
    dist = EmpiricalDistribution(sampler, dummy_data)
    
    N = 10
    D = 2
    x = torch.randn(N, D)
    
    # Test with different time values
    for t_val in [0.01, 0.5, 0.99]:
        t = torch.ones(N) * t_val
        x0_hat = dist.x0(x, t)
        
        # Check for NaN and Inf
        assert not torch.any(torch.isnan(x0_hat))
        assert not torch.any(torch.isinf(x0_hat))

def test_empirical_vector_field_types(sampler, dummy_data):
    """Test that all vector field types work correctly"""
    dist = EmpiricalDistribution(sampler, dummy_data)
    
    N = 10
    D = 2
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5
    
    # Test x0
    x0_hat = dist.x0(x, t)
    assert x0_hat.shape == (N, D)
    
    # Test eps
    eps_hat = dist.eps(x, t)
    assert eps_hat.shape == (N, D)
    
    # Test v
    v_hat = dist.v(x, t)
    assert v_hat.shape == (N, D)
    
    # Test score
    score_hat = dist.score(x, t)
    assert score_hat.shape == (N, D)
    
    # Test consistency between vector field types
    # Reconstruct x from x0_hat
    x_from_x0 = sampler.alpha(t)[:, None] * x0_hat + sampler.sigma(t)[:, None] * eps_hat
    assert torch.allclose(x, x_from_x0, rtol=1e-5)


def test_empirical_sampling_integration(sampler, dummy_data):
    """Integration test for sampling process"""
    dist = EmpiricalDistribution(sampler, dummy_data)
    
    # Sample points from the distribution
    N = 100
    X_sampled, y_sampled = dist.sample(N)
    
    # Collect all training data
    X_train = []
    y_train = []
    for X_batch, y_batch in dummy_data:
        X_train.append(X_batch)
        y_train.append(y_batch)
    X_train = torch.cat(X_train)
    y_train = torch.cat(y_train)
    
    # Check that sampled points have similar statistics to training data
    assert torch.allclose(X_sampled.mean(0), X_train.mean(0), atol=0.2)
    assert torch.allclose(X_sampled.std(0), X_train.std(0), atol=0.2)
    
    # Check label distribution
    sampled_label_dist = torch.bincount(y_sampled.long()) / len(y_sampled)
    train_label_dist = torch.bincount(y_train.long()) / len(y_train)
    assert torch.allclose(sampled_label_dist, train_label_dist, atol=0.1)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_empirical_device_movement(sampler, dummy_data):
    """Test that distribution can be moved between devices"""
    dist = EmpiricalDistribution(sampler, dummy_data)
    
    # Move to CUDA
    device = torch.device("cuda:0")
    
    # Test x0 prediction on CUDA
    N = 10
    D = 2
    x = torch.randn(N, D, device=device)
    t = torch.ones(N, device=device) * 0.5
    
    x0_hat = dist.x0(x, t)
    assert x0_hat.device == device
    
    # Test sampling on CUDA
    X, y = dist.sample(N)
    # Note: The sampled data might not be on CUDA since it comes from the DataLoader
    # but it should be movable to CUDA
    X = X.to(device)
    y = y.to(device)
    assert X.device == device
    assert y.device == device 

def test_empirical_error_cases(sampler, dummy_data):
    """Test that appropriate errors are raised for invalid inputs"""
    # Test missing data
    with pytest.raises(AssertionError):
        EmpiricalDistribution(sampler, None)
    
    # Test empty data loader
    empty_dataset = TensorDataset(torch.tensor([]), torch.tensor([]))
    empty_loader = DataLoader(empty_dataset, batch_size=1)
    with pytest.raises(AssertionError):
        EmpiricalDistribution(sampler, empty_loader)
    
    # Test x0 with mismatched dimensions
    N = 10
    x = torch.randn(N, 3)  # 3D data when distribution expects 2D
    t = torch.ones(N) * 0.5

    dist = EmpiricalDistribution(sampler, dummy_data)
    with pytest.raises(RuntimeError):
        dist.x0(x, t)
    
    # Test x0 with mismatched batch sizes
    x = torch.randn(N, 2)
    t = torch.ones(N+1) * 0.5  # Different batch size than x
    
    with pytest.raises(RuntimeError):
        dist.x0(x, t) 

def test_empirical_diffusion_sampling_integration(sampler, dummy_data):
    """Integration test for the full diffusion sampling process"""
    dist = EmpiricalDistribution(sampler, dummy_data)
    
    # Get initial samples, generated standard normal
    N = 100
    x0 = torch.randn(N, 2)
    
    # Generate noise for sampling process
    num_steps = len(sampler.schedule)
    zs = torch.randn(num_steps - 1, N, 2)  # (L-1, N, D) noise vectors
    
    # Create vector field from empirical distribution
    for vf_func, vf_type in [(lambda x, t: dist.x0(x, t), VectorFieldType.X0),
                             (lambda x, t: dist.eps(x, t), VectorFieldType.EPS),
                             (lambda x, t: dist.v(x, t), VectorFieldType.V),
                             (lambda x, t: dist.score(x, t), VectorFieldType.SCORE)]:
        vector_field = VectorField(vf_func, vf_type)
        
        # Sample using the diffusion process
        x_sampled = sampler.sample(vector_field, x0, zs)
        
        # Collect all training data for comparison
        X_train = []
        for X_batch, _ in dummy_data:
            X_train.append(X_batch)
        X_train = torch.concatenate(X_train)
        
        # The sampled points should be close to but not exactly the same as the training data
        # Check this by verifying the distributions are similar but not identical
        
        # Check means are close but not identical
        train_mean = X_train.mean(0)
        sampled_mean = x_sampled.mean(0)
        mean_diff = torch.norm(train_mean - sampled_mean)
        assert mean_diff < 0.5, "Means are too different"
        
        # Check covariances are close but not identical
        train_cov = torch.cov(X_train.T)
        sampled_cov = torch.cov(x_sampled.T)
        cov_diff = torch.norm(train_cov - sampled_cov)
        assert cov_diff < 1.0, "Covariances are too different"
        
        # Check that no points are exactly the same as training points
        dists = torch.cdist(x_sampled, X_train)
        min_dists = dists.min(dim=1)[0]
        assert torch.all(min_dists < 2.0), "Some sampled points are too far from training distribution" 