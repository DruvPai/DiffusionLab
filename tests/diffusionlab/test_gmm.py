import pytest
import torch
from diffusionlab.distributions.gmm import GMMDistribution, IsoHomoGMMDistribution
from diffusionlab.samplers import VPSampler
from diffusionlab.vector_fields import VectorFieldType

@pytest.fixture
def sampler():
    return VPSampler(is_stochastic=False, t_min=0.01, t_max=0.99, L=100)

@pytest.fixture
def gmm_params():
    K = 3  # number of components
    D = 2  # dimension
    device = torch.device("cpu")
    
    means = torch.tensor([
        [1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0]
    ], device=device)
    
    covs = torch.stack([
        torch.eye(D, device=device) * 0.1,
        torch.eye(D, device=device) * 0.2,
        torch.eye(D, device=device) * 0.3
    ])
    
    priors = torch.tensor([0.3, 0.3, 0.4], device=device)
    
    return means, covs, priors

@pytest.fixture
def iso_homo_gmm_params(gmm_params):
    means, _, priors = gmm_params
    var = torch.tensor(0.2)
    return means, var, priors

def test_gmm_initialization(sampler, gmm_params):
    means, covs, priors = gmm_params
    dist = GMMDistribution(sampler, means, covs, priors)
    
    assert torch.allclose(dist.means, means)
    assert torch.allclose(dist.covs, covs)
    assert torch.allclose(dist.priors, priors)

def test_iso_homo_gmm_initialization(sampler, iso_homo_gmm_params):
    means, var, priors = iso_homo_gmm_params
    dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    assert torch.allclose(dist.means, means)
    assert torch.allclose(dist.var, var)
    assert torch.allclose(dist.priors, priors)

def test_gmm_sampling(sampler, gmm_params):
    means, covs, priors = gmm_params
    dist = GMMDistribution(sampler, means, covs, priors)
    
    N = 1000
    X, y = dist.sample(N)
    
    assert X.shape == (N, means.shape[1])
    assert y.shape == (N,)
    assert y.min() >= 0 and y.max() < means.shape[0]
    
    # Test if samples roughly match the priors
    for k in range(means.shape[0]):
        count = (y == k).sum()
        ratio = count / N
        assert abs(ratio - priors[k]) < 0.1  # Allow some sampling variance

def test_iso_homo_gmm_sampling(sampler, iso_homo_gmm_params):
    means, var, priors = iso_homo_gmm_params
    dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    N = 1000
    X, y = dist.sample(N)
    
    assert X.shape == (N, means.shape[1])
    assert y.shape == (N,)
    assert y.min() >= 0 and y.max() < means.shape[0]
    
    # Test if samples roughly match the priors
    for k in range(means.shape[0]):
        count = (y == k).sum()
        ratio = count / N
        assert abs(ratio - priors[k]) < 0.1  # Allow some sampling variance

def test_gmm_x0_shape(sampler, gmm_params):
    means, covs, priors = gmm_params
    dist = GMMDistribution(sampler, means, covs, priors)
    
    N = 10
    D = means.shape[1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5
    
    x0_hat = dist.x0(x, t)
    assert x0_hat.shape == (N, D)

def test_iso_homo_gmm_x0_shape(sampler, iso_homo_gmm_params):
    means, var, priors = iso_homo_gmm_params
    dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    N = 10
    D = means.shape[1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5
    
    x0_hat = dist.x0(x, t)
    assert x0_hat.shape == (N, D)

def test_gmm_iso_homo_equivalence(sampler, iso_homo_gmm_params):
    """Test that GMMDistribution with isotropic homogeneous covariances gives same results as IsoHomoGMMDistribution"""
    means, var, priors = iso_homo_gmm_params
    K, D = means.shape
    
    # Create equivalent covariance matrices
    covs = torch.stack([torch.eye(D) * var] * K)
    
    gmm_dist = GMMDistribution(sampler, means, covs, priors)
    iso_homo_dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    # Test x0 predictions
    N = 10
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5
    
    x0_gmm = gmm_dist.x0(x, t)
    x0_iso = iso_homo_dist.x0(x, t)
    
    assert torch.allclose(x0_gmm, x0_iso, rtol=1e-5)
    
    # Test sampling
    N = 100
    torch.manual_seed(42)  # For reproducibility
    X_gmm, y_gmm = gmm_dist.sample(N)
    
    torch.manual_seed(42)  # Reset seed
    X_iso, y_iso = iso_homo_dist.sample(N)
    
    assert torch.allclose(X_gmm, X_iso, rtol=1e-5)
    assert torch.all(y_gmm == y_iso)

def test_gmm_sampling_with_x0(sampler, gmm_params):
    """Test that sampling works with the provided x0"""
    means, covs, priors = gmm_params
    dist = GMMDistribution(sampler, means, covs, priors)
    
    # Sample initial points
    N = 10
    X, _ = dist.sample(N)
    
    # Create noise for sampling
    t = torch.ones(N) * 0.5
    eps = torch.randn_like(X)
    
    # Add noise to samples
    x_t = sampler.add_noise(X, t, eps)
    
    # Predict x0
    x0_hat = dist.x0(x_t, t)
    
    # Check that predicted x0 is close to original samples
    # Note: This is not exact due to the probabilistic nature of GMMs
    assert torch.norm(X - x0_hat) / torch.norm(X) < 0.5  # Allow some reconstruction error

def test_iso_homo_gmm_sampling_with_x0(sampler, iso_homo_gmm_params):
    """Test that sampling works with the provided x0"""
    means, var, priors = iso_homo_gmm_params
    dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    # Sample initial points
    N = 10
    X, _ = dist.sample(N)
    
    # Create noise for sampling
    t = torch.ones(N) * 0.5
    eps = torch.randn_like(X)
    
    # Add noise to samples
    x_t = sampler.add_noise(X, t, eps)
    
    # Predict x0
    x0_hat = dist.x0(x_t, t)
    
    # Check that predicted x0 is close to original samples
    # Note: This is not exact due to the probabilistic nature of GMMs
    assert torch.norm(X - x0_hat) / torch.norm(X) < 0.5  # Allow some reconstruction error

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gmm_device_movement(sampler, gmm_params):
    """Test that distributions can be moved between devices"""
    means, covs, priors = gmm_params
    dist = GMMDistribution(sampler, means, covs, priors)
    
    # Move to CUDA
    device = torch.device("cuda:0")
    means_cuda = means.to(device)
    covs_cuda = covs.to(device)
    priors_cuda = priors.to(device)
    dist_cuda = GMMDistribution(sampler, means_cuda, covs_cuda, priors_cuda)
    
    # Test sampling on CUDA
    N = 10
    X_cuda, y_cuda = dist_cuda.sample(N)
    assert X_cuda.device == device
    assert y_cuda.device == device
    
    # Test x0 prediction on CUDA
    x = torch.randn(N, means.shape[1], device=device)
    t = torch.ones(N, device=device) * 0.5
    x0_hat = dist_cuda.x0(x, t)
    assert x0_hat.device == device

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_iso_homo_gmm_device_movement(sampler, iso_homo_gmm_params):
    """Test that isotropic homogeneous distributions can be moved between devices"""
    means, var, priors = iso_homo_gmm_params
    dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    # Move to CUDA
    device = torch.device("cuda:0")
    means_cuda = means.to(device)
    var_cuda = var.to(device)
    priors_cuda = priors.to(device)
    dist_cuda = IsoHomoGMMDistribution(sampler, means_cuda, var_cuda, priors_cuda)
    
    # Test sampling on CUDA
    N = 10
    X_cuda, y_cuda = dist_cuda.sample(N)
    assert X_cuda.device == device
    assert y_cuda.device == device
    
    # Test x0 prediction on CUDA
    x = torch.randn(N, means.shape[1], device=device)
    t = torch.ones(N, device=device) * 0.5
    x0_hat = dist_cuda.x0(x, t)
    assert x0_hat.device == device

def test_gmm_vector_field_types(sampler, gmm_params):
    """Test that all vector field types work correctly"""
    means, covs, priors = gmm_params
    dist = GMMDistribution(sampler, means, covs, priors)
    
    N = 10
    D = means.shape[1]
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

def test_iso_homo_gmm_vector_field_types(sampler, iso_homo_gmm_params):
    """Test that all vector field types work correctly for isotropic homogeneous GMM"""
    means, var, priors = iso_homo_gmm_params
    dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    N = 10
    D = means.shape[1]
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

def test_gmm_error_cases(sampler):
    """Test that appropriate errors are raised for invalid inputs"""
    # Test mismatched dimensions
    means = torch.randn(3, 2)  # 3 components, 2D
    covs = torch.stack([torch.eye(3)] * 3)  # Wrong dimension (3x3 instead of 2x2)
    priors = torch.ones(3) / 3
    
    with pytest.raises(AssertionError):
        GMMDistribution(sampler, means, covs, priors)
    
    # Test invalid priors (not summing to 1)
    means = torch.randn(3, 2)
    covs = torch.stack([torch.eye(2)] * 3)
    priors = torch.ones(3)  # Not normalized
    
    with pytest.raises(AssertionError):
        GMMDistribution(sampler, means, covs, priors)

def test_iso_homo_gmm_error_cases(sampler):
    """Test that appropriate errors are raised for invalid inputs"""
    # Test negative variance
    means = torch.randn(3, 2)
    var = torch.tensor(-1.0)  # Negative variance
    priors = torch.ones(3) / 3
    
    with pytest.raises(AssertionError):
        IsoHomoGMMDistribution(sampler, means, var, priors)
    
    # Test invalid priors (not summing to 1)
    means = torch.randn(3, 2)
    var = torch.tensor(1.0)
    priors = torch.ones(3)  # Not normalized
    
    with pytest.raises(AssertionError):
        IsoHomoGMMDistribution(sampler, means, var, priors)

def test_gmm_numerical_stability(sampler):
    """Test numerical stability in edge cases"""
    # Test with very small covariances
    means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    covs = torch.stack([torch.eye(2) * 1e-10] * 2)  # Very small covariances
    priors = torch.ones(2) / 2
    
    dist = GMMDistribution(sampler, means, covs, priors)
    
    N = 10
    x = torch.randn(N, 2)
    t = torch.ones(N) * 0.5
    
    x0_hat = dist.x0(x, t)
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    
    # Test with very large covariances
    covs = torch.stack([torch.eye(2) * 1e10] * 2)  # Very large covariances
    dist = GMMDistribution(sampler, means, covs, priors)
    
    x0_hat = dist.x0(x, t)
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))

def test_iso_homo_gmm_numerical_stability(sampler):
    """Test numerical stability in edge cases"""
    # Test with very small variance
    means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    var = torch.tensor(1e-10)  # Very small variance
    priors = torch.ones(2) / 2
    
    dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    N = 10
    x = torch.randn(N, 2)
    t = torch.ones(N) * 0.5
    
    x0_hat = dist.x0(x, t)
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    
    # Test with very large variance
    var = torch.tensor(1e10)  # Very large variance
    dist = IsoHomoGMMDistribution(sampler, means, var, priors)
    
    x0_hat = dist.x0(x, t)
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat)) 