import pytest
import torch
from diffusionlab.distributions.gmm import (
    GMMDistribution,
    IsoHomoGMMDistribution,
    IsoGMMDistribution,
    LowRankGMMDistribution,
)
from diffusionlab.samplers import VPSampler

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sampler():
    """Create a VP sampler for testing."""
    return VPSampler(is_stochastic=False, t_min=0.01, t_max=0.99, L=100)


@pytest.fixture
def sampling_gmm_params():
    """Create non-batched GMM parameters for sampling."""
    D = 2  # dimension
    device = torch.device("cpu")

    means = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=device)
    covs = torch.stack(
        [
            torch.eye(D, device=device) * 0.1,
            torch.eye(D, device=device) * 0.2,
            torch.eye(D, device=device) * 0.3,
        ]
    )
    priors = torch.tensor([0.3, 0.3, 0.4], device=device)

    return {
        "means": means,  # (K, D)
        "covs": covs,  # (K, D, D)
        "priors": priors,  # (K,)
    }


@pytest.fixture
def sampling_iso_gmm_params(sampling_gmm_params):
    """Create non-batched isotropic GMM parameters for sampling."""
    means = sampling_gmm_params["means"]
    priors = sampling_gmm_params["priors"]
    vars = torch.tensor([0.1, 0.2, 0.3])  # Different variance for each component
    return {
        "means": means,  # (K, D)
        "vars": vars,  # (K,)
        "priors": priors,  # (K,)
    }


@pytest.fixture
def denoising_iso_gmm_params(sampling_iso_gmm_params):
    """Create batched isotropic GMM parameters for denoising."""
    N = 10  # batch size
    K = sampling_iso_gmm_params["means"].shape[0]
    D = sampling_iso_gmm_params["means"].shape[1]

    # Create batch of means by adding random offsets
    means_offset = torch.randn(N, K, D) * 0.2
    means = sampling_iso_gmm_params["means"][None, ...].expand(N, -1, -1) + means_offset

    # Create batch of variances by scaling the base variances
    var_scales = torch.exp(torch.randn(N, K) * 0.2)  # Random positive scales
    vars = sampling_iso_gmm_params["vars"][None, ...].expand(N, -1) * var_scales

    # Create batch of priors by perturbing and renormalizing
    priors_logits = (
        torch.log(sampling_iso_gmm_params["priors"])[None, ...].expand(N, -1)
        + torch.randn(N, K) * 0.2
    )
    priors = torch.softmax(priors_logits, dim=-1)

    return {
        "means": means,  # (N, K, D)
        "vars": vars,  # (N, K)
        "priors": priors,  # (N, K)
    }


@pytest.fixture
def denoising_gmm_params(sampling_gmm_params):
    """Create batched GMM parameters for denoising with varying parameters across batch."""
    N = 10  # batch size
    K = sampling_gmm_params["means"].shape[0]
    D = sampling_gmm_params["means"].shape[1]

    # Create batch of means by adding random offsets
    means_offset = torch.randn(N, K, D) * 0.2
    means = sampling_gmm_params["means"][None, ...].expand(N, -1, -1) + means_offset

    # Create batch of covariances by scaling the base covariances
    cov_scales = torch.exp(torch.randn(N, K) * 0.2)  # Random positive scales
    covs = (
        sampling_gmm_params["covs"][None, ...].expand(N, -1, -1, -1)
        * cov_scales[..., None, None]
    )

    # Create batch of priors by perturbing and renormalizing
    priors_logits = (
        torch.log(sampling_gmm_params["priors"])[None, ...].expand(N, -1)
        + torch.randn(N, K) * 0.2
    )
    priors = torch.softmax(priors_logits, dim=-1)

    return {
        "means": means,  # (N, K, D)
        "covs": covs,  # (N, K, D, D)
        "priors": priors,  # (N, K)
    }


@pytest.fixture
def sampling_iso_homo_gmm_params(sampling_gmm_params):
    """Create non-batched isotropic homogeneous GMM parameters for sampling."""
    means = sampling_gmm_params["means"]
    priors = sampling_gmm_params["priors"]
    var = torch.tensor(0.2)
    return {
        "means": means,  # (K, D)
        "var": var,  # ()
        "priors": priors,  # (K,)
    }


@pytest.fixture
def denoising_iso_homo_gmm_params(sampling_iso_homo_gmm_params):
    """Create batched isotropic homogeneous GMM parameters for denoising."""
    N = 10  # batch size
    K = sampling_iso_homo_gmm_params["means"].shape[0]
    D = sampling_iso_homo_gmm_params["means"].shape[1]

    # Create batch of means by adding random offsets
    means_offset = torch.randn(N, K, D) * 0.2
    means = (
        sampling_iso_homo_gmm_params["means"][None, ...].expand(N, -1, -1)
        + means_offset
    )

    # Create batch of variances by scaling the base variance
    var_scales = torch.exp(torch.randn(N) * 0.2)  # Random positive scales
    var = sampling_iso_homo_gmm_params["var"] * var_scales

    # Create batch of priors by perturbing and renormalizing
    priors_logits = (
        torch.log(sampling_iso_homo_gmm_params["priors"])[None, ...].expand(N, -1)
        + torch.randn(N, K) * 0.2
    )
    priors = torch.softmax(priors_logits, dim=-1)

    return {
        "means": means,  # (N, K, D)
        "var": var,  # (N,)
        "priors": priors,  # (N, K)
    }


@pytest.fixture
def sampling_low_rank_gmm_params():
    """Create non-batched low-rank GMM parameters for sampling."""
    D = 2  # dimension
    device = torch.device("cpu")

    means = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]], device=device)
    # Create low-rank factors that would result in diagonal covariances
    covs_factors = torch.stack(
        [
            torch.tensor([[0.3162]], device=device).expand(1, D).T,  # sqrt(0.1)
            torch.tensor([[0.4472]], device=device).expand(1, D).T,  # sqrt(0.2)
            torch.tensor([[0.5477]], device=device).expand(1, D).T,  # sqrt(0.3)
        ]
    )  # (K, D, R)
    priors = torch.tensor([0.3, 0.3, 0.4], device=device)

    return {
        "means": means,  # (K, D)
        "covs_factors": covs_factors,  # (K, D, P)
        "priors": priors,  # (K,)
    }


@pytest.fixture
def denoising_low_rank_gmm_params(sampling_low_rank_gmm_params):
    """Create batched low-rank GMM parameters for denoising."""
    N = 10  # batch size
    K = sampling_low_rank_gmm_params["means"].shape[0]
    D = sampling_low_rank_gmm_params["means"].shape[1]

    # Create batch of means by adding random offsets
    means_offset = torch.randn(N, K, D) * 0.2
    means = (
        sampling_low_rank_gmm_params["means"][None, ...].expand(N, -1, -1)
        + means_offset
    )

    # Create batch of factors by scaling the base factors
    covs_factors_scales = torch.exp(torch.randn(N, K) * 0.2)[
        :, :, None, None
    ]  # Random positive scales
    covs_factors = (
        sampling_low_rank_gmm_params["covs_factors"][None, ...].expand(N, -1, -1, -1)
        * covs_factors_scales
    )

    # Create batch of priors by perturbing and renormalizing
    priors_logits = (
        torch.log(sampling_low_rank_gmm_params["priors"])[None, ...].expand(N, -1)
        + torch.randn(N, K) * 0.2
    )
    priors = torch.softmax(priors_logits, dim=-1)

    return {
        "means": means,  # (N, K, D)
        "covs_factors": covs_factors,  # (N, K, D, P)
        "priors": priors,  # (N, K)
    }


# ============================================================================
# Validation Tests
# ============================================================================


def test_gmm_validation():
    """Test validation of parameters for GMM distribution."""
    D = 2
    K = 3
    N = 3

    # Test valid sampling parameters (non-batched)
    sampling_params = {
        "means": torch.randn(K, D),
        "covs": torch.stack([torch.eye(D)] * K),
        "priors": torch.ones(K) / K,
    }
    GMMDistribution.validate_params(sampling_params)

    # Test valid denoising parameters (batched)
    denoising_params = {
        "means": torch.randn(N, K, D),
        "covs": torch.stack([torch.eye(D)] * K)[None].expand(N, -1, -1, -1),
        "priors": torch.ones(N, K) / K,
    }
    GMMDistribution.validate_params(denoising_params)

    # Test error cases
    with pytest.raises(AssertionError):
        GMMDistribution.validate_params(
            {"means": torch.randn(K, D)}
        )  # Missing parameters

    invalid_params = sampling_params.copy()
    invalid_params["covs"] = torch.stack([torch.eye(3)] * K)  # Wrong dimension
    with pytest.raises(AssertionError):
        GMMDistribution.validate_params(invalid_params)

    invalid_params = denoising_params.copy()
    invalid_params["priors"] = torch.ones(N, K)  # Not normalized
    with pytest.raises(AssertionError):
        GMMDistribution.validate_params(invalid_params)

    invalid_params = denoising_params.copy()
    invalid_covs = torch.tensor([[1.0, 2.0], [2.0, 1.0]])[None, None].expand(
        N, K, -1, -1
    )
    invalid_params["covs"] = invalid_covs  # Non-positive definite
    with pytest.raises(AssertionError):
        GMMDistribution.validate_params(invalid_params)


def test_iso_homo_gmm_validation():
    """Test validation of parameters for isotropic homogeneous GMM."""
    D = 2
    K = 3
    N = 3

    # Test valid sampling parameters (non-batched)
    sampling_params = {
        "means": torch.randn(K, D),
        "var": torch.tensor(0.1),
        "priors": torch.ones(K) / K,
    }
    IsoHomoGMMDistribution.validate_params(sampling_params)

    # Test valid denoising parameters (batched)
    denoising_params = {
        "means": torch.randn(N, K, D),
        "var": torch.full((N,), 0.1),
        "priors": torch.ones(N, K) / K,
    }
    IsoHomoGMMDistribution.validate_params(denoising_params)

    # Test error cases
    with pytest.raises(AssertionError):
        IsoHomoGMMDistribution.validate_params(
            {"means": torch.randn(K, D)}
        )  # Missing parameters

    invalid_params = sampling_params.copy()
    invalid_params["var"] = torch.tensor(-1.0)  # Negative variance
    with pytest.raises(AssertionError):
        IsoHomoGMMDistribution.validate_params(invalid_params)

    invalid_params = denoising_params.copy()
    invalid_params["priors"] = torch.ones(N, K)  # Not normalized
    with pytest.raises(AssertionError):
        IsoHomoGMMDistribution.validate_params(invalid_params)


def test_iso_gmm_validation():
    """Test validation of parameters for isotropic GMM."""
    D = 2
    K = 3
    N = 3

    # Test valid sampling parameters (non-batched)
    sampling_params = {
        "means": torch.randn(K, D),
        "vars": torch.ones(K) * 0.1,
        "priors": torch.ones(K) / K,
    }
    IsoGMMDistribution.validate_params(sampling_params)

    # Test valid denoising parameters (batched)
    denoising_params = {
        "means": torch.randn(N, K, D),
        "vars": torch.ones(N, K) * 0.1,
        "priors": torch.ones(N, K) / K,
    }
    IsoGMMDistribution.validate_params(denoising_params)

    # Test error cases
    with pytest.raises(AssertionError):
        IsoGMMDistribution.validate_params(
            {"means": torch.randn(K, D)}
        )  # Missing parameters

    invalid_params = sampling_params.copy()
    invalid_params["vars"] = torch.tensor([-1.0] * K)  # Negative variances
    with pytest.raises(AssertionError):
        IsoGMMDistribution.validate_params(invalid_params)

    invalid_params = denoising_params.copy()
    invalid_params["priors"] = torch.ones(N, K)  # Not normalized
    with pytest.raises(AssertionError):
        IsoGMMDistribution.validate_params(invalid_params)


def test_low_rank_gmm_validation():
    """Test validation of parameters for low-rank GMM."""
    D = 2
    K = 3
    R = 1
    N = 3

    # Test valid sampling parameters (non-batched)
    sampling_params = {
        "means": torch.randn(K, D),
        "covs_factors": torch.randn(K, D, R),
        "priors": torch.ones(K) / K,
    }
    LowRankGMMDistribution.validate_params(sampling_params)

    # Test valid denoising parameters (batched)
    denoising_params = {
        "means": torch.randn(N, K, D),
        "covs_factors": torch.randn(N, K, D, R),
        "priors": torch.ones(N, K) / K,
    }
    LowRankGMMDistribution.validate_params(denoising_params)

    # Test error cases
    with pytest.raises(AssertionError):
        LowRankGMMDistribution.validate_params(
            {"means": torch.randn(K, D)}
        )  # Missing parameters

    invalid_params = sampling_params.copy()
    invalid_params["covs_factors"] = torch.randn(K, D + 1, R)  # Wrong dimension
    with pytest.raises(AssertionError):
        LowRankGMMDistribution.validate_params(invalid_params)

    invalid_params = denoising_params.copy()
    invalid_params["priors"] = torch.ones(N, K)  # Not normalized
    with pytest.raises(AssertionError):
        LowRankGMMDistribution.validate_params(invalid_params)


# ============================================================================
# Sampling Tests
# ============================================================================


def test_gmm_sampling(sampling_gmm_params):
    """Test sampling from GMM distribution."""
    N = 1000
    X, y = GMMDistribution.sample(N, sampling_gmm_params, {})

    # Check shapes and ranges
    assert X.shape == (N, sampling_gmm_params["means"].shape[1])
    assert y.shape == (N,)
    assert y.min() >= 0 and y.max() < sampling_gmm_params["means"].shape[0]

    # Check component proportions match priors
    for k in range(sampling_gmm_params["means"].shape[0]):
        count = (y == k).sum()
        ratio = count / N
        assert abs(ratio - sampling_gmm_params["priors"][k]) < 0.1

    # Check component distributions
    for k in range(sampling_gmm_params["means"].shape[0]):
        mask = y == k
        if mask.sum() > 0:
            component_samples = X[mask]
            mean = component_samples.mean(0)
            cov = torch.cov(component_samples.T)

            # Check statistics
            assert (
                torch.norm(mean - sampling_gmm_params["means"][k])
                / torch.norm(sampling_gmm_params["means"][k])
                < 0.5
            )
            assert (
                torch.norm(cov - sampling_gmm_params["covs"][k])
                / torch.norm(sampling_gmm_params["covs"][k])
                < 0.5
            )


def test_iso_homo_gmm_sampling(sampling_iso_homo_gmm_params):
    """Test sampling from isotropic homogeneous GMM."""
    N = 1000
    X, y = IsoHomoGMMDistribution.sample(N, sampling_iso_homo_gmm_params, {})

    # Check shapes and ranges
    assert X.shape == (N, sampling_iso_homo_gmm_params["means"].shape[1])
    assert y.shape == (N,)
    assert y.min() >= 0 and y.max() < sampling_iso_homo_gmm_params["means"].shape[0]

    # Check component proportions match priors
    for k in range(sampling_iso_homo_gmm_params["means"].shape[0]):
        count = (y == k).sum()
        ratio = count / N
        assert abs(ratio - sampling_iso_homo_gmm_params["priors"][k]) < 0.1

    # Check component distributions
    for k in range(sampling_iso_homo_gmm_params["means"].shape[0]):
        mask = y == k
        if mask.sum() > 0:
            component_samples = X[mask]
            mean = component_samples.mean(0)
            cov = torch.cov(component_samples.T)

            # Check statistics
            assert torch.allclose(
                mean, sampling_iso_homo_gmm_params["means"][k], atol=0.5
            )
            expected_cov = torch.eye(2) * sampling_iso_homo_gmm_params["var"]
            assert torch.allclose(cov, expected_cov, atol=0.5)


def test_iso_gmm_sampling(sampling_iso_gmm_params):
    """Test sampling from isotropic GMM."""
    N = 1000
    X, y = IsoGMMDistribution.sample(N, sampling_iso_gmm_params, {})

    # Check shapes and ranges
    assert X.shape == (N, sampling_iso_gmm_params["means"].shape[1])
    assert y.shape == (N,)
    assert y.min() >= 0 and y.max() < sampling_iso_gmm_params["means"].shape[0]

    # Check component proportions match priors
    for k in range(sampling_iso_gmm_params["means"].shape[0]):
        count = (y == k).sum()
        ratio = count / N
        assert abs(ratio - sampling_iso_gmm_params["priors"][k]) < 0.1

    # Check component distributions
    for k in range(sampling_iso_gmm_params["means"].shape[0]):
        mask = y == k
        if mask.sum() > 0:
            component_samples = X[mask]
            mean = component_samples.mean(0)
            cov = torch.cov(component_samples.T)

            # Check statistics
            assert torch.allclose(mean, sampling_iso_gmm_params["means"][k], atol=0.5)
            expected_cov = torch.eye(2) * sampling_iso_gmm_params["vars"][k]
            assert torch.allclose(cov, expected_cov, atol=0.5)


def test_low_rank_gmm_sampling(sampling_low_rank_gmm_params):
    """Test sampling from low-rank GMM."""
    N = 1000
    X, y = LowRankGMMDistribution.sample(N, sampling_low_rank_gmm_params, {})

    # Check shapes and ranges
    assert X.shape == (N, sampling_low_rank_gmm_params["means"].shape[1])
    assert y.shape == (N,)
    assert y.min() >= 0 and y.max() < sampling_low_rank_gmm_params["means"].shape[0]

    # Check component proportions match priors
    for k in range(sampling_low_rank_gmm_params["means"].shape[0]):
        count = (y == k).sum()
        ratio = count / N
        assert abs(ratio - sampling_low_rank_gmm_params["priors"][k]) < 0.1

    # Check component distributions
    for k in range(sampling_low_rank_gmm_params["means"].shape[0]):
        mask = y == k
        if mask.sum() > 0:
            component_samples = X[mask]
            mean = component_samples.mean(0)
            cov = torch.cov(component_samples.T)

            # Check statistics
            assert torch.allclose(
                mean, sampling_low_rank_gmm_params["means"][k], atol=0.5
            )
            covs_factors = sampling_low_rank_gmm_params["covs_factors"][k]  # (D, P)
            expected_cov = covs_factors @ covs_factors.T
            assert torch.allclose(cov, expected_cov, atol=0.5)


def test_low_rank_gmm_equals_full_gmm():
    """Test that LowRankGMMDistribution equals GMMDistribution when covariances are low-rank."""
    N = 10  # batch size
    K = 3  # number of components
    D = 2  # dimension
    R = 1  # rank

    # Create parameters that would result in the same distribution
    means = torch.randn(N, K, D)
    covs_factors = torch.randn(N, K, D, R)  # Low-rank factors
    covs = covs_factors @ covs_factors.transpose(
        -1, -2
    )  # Construct full covariance from factors
    priors = torch.softmax(torch.randn(N, K), dim=-1)

    low_rank_params = {
        "means": means,
        "covs_factors": covs_factors,
        "priors": priors,
    }

    full_params = {
        "means": means,
        "covs": covs,
        "priors": priors,
    }

    # Test vector fields
    sampler = VPSampler(is_stochastic=False, t_min=0.01, t_max=0.99, L=100)
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    # Test x0
    x0_low_rank = LowRankGMMDistribution.x0(x, t, sampler, low_rank_params, {})
    x0_full = GMMDistribution.x0(x, t, sampler, full_params, {})
    assert torch.allclose(x0_low_rank, x0_full, atol=1e-5)

    # Test eps
    eps_low_rank = LowRankGMMDistribution.eps(x, t, sampler, low_rank_params, {})
    eps_full = GMMDistribution.eps(x, t, sampler, full_params, {})
    assert torch.allclose(eps_low_rank, eps_full, atol=1e-5)

    # Test v
    v_low_rank = LowRankGMMDistribution.v(x, t, sampler, low_rank_params, {})
    v_full = GMMDistribution.v(x, t, sampler, full_params, {})
    assert torch.allclose(v_low_rank, v_full, atol=1e-5)

    # Test score
    score_low_rank = LowRankGMMDistribution.score(x, t, sampler, low_rank_params, {})
    score_full = GMMDistribution.score(x, t, sampler, full_params, {})
    assert torch.allclose(score_low_rank, score_full, atol=1e-5)


# ============================================================================
# Vector Field Tests
# ============================================================================


def test_gmm_x0_shape(sampler, denoising_gmm_params):
    """Test x0 prediction shape for GMM."""
    N = 10
    D = denoising_gmm_params["means"].shape[-1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    x0_hat = GMMDistribution.x0(x, t, sampler, denoising_gmm_params, {})
    assert x0_hat.shape == (N, D)


def test_iso_homo_gmm_x0_shape(sampler, denoising_iso_homo_gmm_params):
    """Test x0 prediction shape for isotropic homogeneous GMM."""
    N = 10
    D = denoising_iso_homo_gmm_params["means"].shape[-1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    x0_hat = IsoHomoGMMDistribution.x0(x, t, sampler, denoising_iso_homo_gmm_params, {})
    assert x0_hat.shape == (N, D)


def test_gmm_vector_field_types(sampler, denoising_gmm_params):
    """Test all vector field types work correctly for GMM."""
    N = 10
    D = denoising_gmm_params["means"].shape[-1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    # Test each vector field type
    x0_hat = GMMDistribution.x0(x, t, sampler, denoising_gmm_params, {})
    eps_hat = GMMDistribution.eps(x, t, sampler, denoising_gmm_params, {})
    v_hat = GMMDistribution.v(x, t, sampler, denoising_gmm_params, {})
    score_hat = GMMDistribution.score(x, t, sampler, denoising_gmm_params, {})

    # Check shapes
    assert x0_hat.shape == (N, D)
    assert eps_hat.shape == (N, D)
    assert v_hat.shape == (N, D)
    assert score_hat.shape == (N, D)

    # Check consistency
    x_from_x0 = sampler.alpha(t)[:, None] * x0_hat + sampler.sigma(t)[:, None] * eps_hat
    assert torch.allclose(x, x_from_x0, rtol=1e-5)


def test_iso_homo_gmm_vector_field_types(sampler, denoising_iso_homo_gmm_params):
    """Test all vector field types work correctly for isotropic homogeneous GMM."""
    N = 10
    D = denoising_iso_homo_gmm_params["means"].shape[-1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    # Test each vector field type
    x0_hat = IsoHomoGMMDistribution.x0(x, t, sampler, denoising_iso_homo_gmm_params, {})
    eps_hat = IsoHomoGMMDistribution.eps(
        x, t, sampler, denoising_iso_homo_gmm_params, {}
    )
    v_hat = IsoHomoGMMDistribution.v(x, t, sampler, denoising_iso_homo_gmm_params, {})
    score_hat = IsoHomoGMMDistribution.score(
        x, t, sampler, denoising_iso_homo_gmm_params, {}
    )

    # Check shapes
    assert x0_hat.shape == (N, D)
    assert eps_hat.shape == (N, D)
    assert v_hat.shape == (N, D)
    assert score_hat.shape == (N, D)

    # Check consistency
    x_from_x0 = sampler.alpha(t)[:, None] * x0_hat + sampler.sigma(t)[:, None] * eps_hat
    assert torch.allclose(x, x_from_x0, rtol=1e-5)


def test_iso_gmm_x0_shape(sampler, denoising_iso_gmm_params):
    """Test x0 prediction shape for isotropic GMM."""
    N = 10
    D = denoising_iso_gmm_params["means"].shape[-1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    x0_hat = IsoGMMDistribution.x0(x, t, sampler, denoising_iso_gmm_params, {})
    assert x0_hat.shape == (N, D)


def test_iso_gmm_vector_field_types(sampler, denoising_iso_gmm_params):
    """Test all vector field types work correctly for isotropic GMM."""
    N = 10
    D = denoising_iso_gmm_params["means"].shape[-1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    # Test each vector field type
    x0_hat = IsoGMMDistribution.x0(x, t, sampler, denoising_iso_gmm_params, {})
    eps_hat = IsoGMMDistribution.eps(x, t, sampler, denoising_iso_gmm_params, {})
    v_hat = IsoGMMDistribution.v(x, t, sampler, denoising_iso_gmm_params, {})
    score_hat = IsoGMMDistribution.score(x, t, sampler, denoising_iso_gmm_params, {})

    # Check shapes
    assert x0_hat.shape == (N, D)
    assert eps_hat.shape == (N, D)
    assert v_hat.shape == (N, D)
    assert score_hat.shape == (N, D)

    # Check consistency
    x_from_x0 = sampler.alpha(t)[:, None] * x0_hat + sampler.sigma(t)[:, None] * eps_hat
    assert torch.allclose(x, x_from_x0, rtol=1e-5)


def test_low_rank_gmm_x0_shape(sampler, denoising_low_rank_gmm_params):
    """Test x0 prediction shape for low-rank GMM."""
    N = 10
    D = denoising_low_rank_gmm_params["means"].shape[-1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    x0_hat = LowRankGMMDistribution.x0(x, t, sampler, denoising_low_rank_gmm_params, {})
    assert x0_hat.shape == (N, D)


def test_low_rank_gmm_vector_field_types(sampler, denoising_low_rank_gmm_params):
    """Test all vector field types work correctly for low-rank GMM."""
    N = 10
    D = denoising_low_rank_gmm_params["means"].shape[-1]
    x = torch.randn(N, D)
    t = torch.ones(N) * 0.5

    # Test each vector field type
    x0_hat = LowRankGMMDistribution.x0(x, t, sampler, denoising_low_rank_gmm_params, {})
    eps_hat = LowRankGMMDistribution.eps(
        x, t, sampler, denoising_low_rank_gmm_params, {}
    )
    v_hat = LowRankGMMDistribution.v(x, t, sampler, denoising_low_rank_gmm_params, {})
    score_hat = LowRankGMMDistribution.score(
        x, t, sampler, denoising_low_rank_gmm_params, {}
    )

    # Check shapes
    assert x0_hat.shape == (N, D)
    assert eps_hat.shape == (N, D)
    assert v_hat.shape == (N, D)
    assert score_hat.shape == (N, D)

    # Check consistency
    x_from_x0 = sampler.alpha(t)[:, None] * x0_hat + sampler.sigma(t)[:, None] * eps_hat
    assert torch.allclose(x, x_from_x0, rtol=1e-5)


# ============================================================================
# Device Tests
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gmm_device_movement(sampler, sampling_gmm_params, denoising_gmm_params):
    """Test GMM distribution works with different devices."""
    device = torch.device("cuda:0")

    # Test sampling
    cuda_sampling_params = {k: v.to(device) for k, v in sampling_gmm_params.items()}
    N = 10
    X, y = GMMDistribution.sample(N, cuda_sampling_params, {})
    assert X.device == device
    assert y.device == device

    # Test denoising
    cuda_denoising_params = {k: v.to(device) for k, v in denoising_gmm_params.items()}
    D = denoising_gmm_params["means"].shape[-1]
    x = torch.randn(N, D, device=device)
    t = torch.ones(N, device=device) * 0.5

    x0_hat = GMMDistribution.x0(x, t, sampler, cuda_denoising_params, {})
    assert x0_hat.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_iso_homo_gmm_device_movement(
    sampler, sampling_iso_homo_gmm_params, denoising_iso_homo_gmm_params
):
    """Test isotropic homogeneous GMM distribution works with different devices."""
    device = torch.device("cuda:0")

    # Test sampling
    cuda_sampling_params = {
        k: v.to(device) for k, v in sampling_iso_homo_gmm_params.items()
    }
    N = 10
    X, y = IsoHomoGMMDistribution.sample(N, cuda_sampling_params, {})
    assert X.device == device
    assert y.device == device

    # Test denoising
    cuda_denoising_params = {
        k: v.to(device) for k, v in denoising_iso_homo_gmm_params.items()
    }
    D = denoising_iso_homo_gmm_params["means"].shape[-1]
    x = torch.randn(N, D, device=device)
    t = torch.ones(N, device=device) * 0.5

    x0_hat = IsoHomoGMMDistribution.x0(x, t, sampler, cuda_denoising_params, {})
    assert x0_hat.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_iso_gmm_device_movement(
    sampler, sampling_iso_gmm_params, denoising_iso_gmm_params
):
    """Test isotropic GMM distribution works with different devices."""
    device = torch.device("cuda:0")

    # Test sampling
    cuda_sampling_params = {k: v.to(device) for k, v in sampling_iso_gmm_params.items()}
    N = 10
    X, y = IsoGMMDistribution.sample(N, cuda_sampling_params, {})
    assert X.device == device
    assert y.device == device

    # Test denoising
    cuda_denoising_params = {
        k: v.to(device) for k, v in denoising_iso_gmm_params.items()
    }
    D = denoising_iso_gmm_params["means"].shape[-1]
    x = torch.randn(N, D, device=device)
    t = torch.ones(N, device=device) * 0.5

    x0_hat = IsoGMMDistribution.x0(x, t, sampler, cuda_denoising_params, {})
    assert x0_hat.device == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_low_rank_gmm_device_movement(
    sampler, sampling_low_rank_gmm_params, denoising_low_rank_gmm_params
):
    """Test low-rank GMM distribution works with different devices."""
    device = torch.device("cuda:0")

    # Test sampling
    cuda_sampling_params = {
        k: v.to(device) for k, v in sampling_low_rank_gmm_params.items()
    }
    N = 10
    X, y = LowRankGMMDistribution.sample(N, cuda_sampling_params, {})
    assert X.device == device
    assert y.device == device

    # Test denoising
    cuda_denoising_params = {
        k: v.to(device) for k, v in denoising_low_rank_gmm_params.items()
    }
    D = denoising_low_rank_gmm_params["means"].shape[-1]
    x = torch.randn(N, D, device=device)
    t = torch.ones(N, device=device) * 0.5

    x0_hat = LowRankGMMDistribution.x0(x, t, sampler, cuda_denoising_params, {})
    assert x0_hat.device == device


# ============================================================================
# Numerical Stability Tests
# ============================================================================


def test_gmm_numerical_stability(sampler):
    """Test numerical stability in edge cases for GMM."""
    means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    priors = torch.ones(2) / 2
    N = 10  # batch size for denoising

    # Test with very small covariances
    covs = torch.stack([torch.eye(2) * 1e-10] * 2)
    denoising_params = {
        "means": means[None].expand(N, -1, -1),
        "covs": covs[None].expand(N, -1, -1, -1),
        "priors": priors[None].expand(N, -1),
    }

    N_test = 10
    x = torch.randn(N_test, 2)
    t = torch.ones(N_test) * 0.5

    x0_hat = GMMDistribution.x0(x, t, sampler, denoising_params, {})
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    assert torch.all(torch.abs(x0_hat) < 100)

    # Test with very large covariances
    denoising_params["covs"] = denoising_params["covs"] * 1e20
    x0_hat = GMMDistribution.x0(x, t, sampler, denoising_params, {})
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    assert torch.all(torch.abs(x0_hat) < 100)


def test_iso_homo_gmm_numerical_stability(sampler):
    """Test numerical stability in edge cases for isotropic homogeneous GMM."""
    means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    priors = torch.ones(2) / 2
    N = 10  # batch size for denoising

    # Test with very small variance
    denoising_params = {
        "means": means[None].expand(N, -1, -1),
        "var": torch.tensor(1e-10).expand(N),
        "priors": priors[None].expand(N, -1),
    }

    N_test = 10
    x = torch.randn(N_test, 2)
    t = torch.ones(N_test) * 0.5

    x0_hat = IsoHomoGMMDistribution.x0(x, t, sampler, denoising_params, {})
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    assert torch.all(torch.abs(x0_hat) < 100)

    # Test with very large variance
    denoising_params["var"] = denoising_params["var"] * 1e20
    x0_hat = IsoHomoGMMDistribution.x0(x, t, sampler, denoising_params, {})
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    assert torch.all(torch.abs(x0_hat) < 100)


def test_iso_gmm_numerical_stability(sampler):
    """Test numerical stability in edge cases for isotropic GMM."""
    means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    priors = torch.ones(2) / 2
    N = 10  # batch size for denoising

    # Test with very small variances
    vars = torch.ones(2) * 1e-10
    denoising_params = {
        "means": means[None].expand(N, -1, -1),
        "vars": vars[None].expand(N, -1),
        "priors": priors[None].expand(N, -1),
    }

    N_test = 10
    x = torch.randn(N_test, 2)
    t = torch.ones(N_test) * 0.5

    x0_hat = IsoGMMDistribution.x0(x, t, sampler, denoising_params, {})
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    assert torch.all(torch.abs(x0_hat) < 100)

    # Test with very large variances
    denoising_params["vars"] = denoising_params["vars"] * 1e20
    x0_hat = IsoGMMDistribution.x0(x, t, sampler, denoising_params, {})
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    assert torch.all(torch.abs(x0_hat) < 100)


def test_low_rank_gmm_numerical_stability(sampler):
    """Test numerical stability in edge cases for low-rank GMM."""
    means = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    priors = torch.ones(2) / 2
    N = 10  # batch size for denoising

    # Test with very small factors
    covs_factors = torch.ones(2, 2, 1) * 1e-5
    denoising_params = {
        "means": means[None].expand(N, -1, -1),
        "covs_factors": covs_factors[None].expand(N, -1, -1, -1),
        "priors": priors[None].expand(N, -1),
    }

    N_test = 10
    x = torch.randn(N_test, 2)
    t = torch.ones(N_test) * 0.5

    x0_hat = LowRankGMMDistribution.x0(x, t, sampler, denoising_params, {})
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    assert torch.all(torch.abs(x0_hat) < 100)

    # Test with very large factors
    denoising_params["covs_factors"] = denoising_params["covs_factors"] * 1e10
    x0_hat = LowRankGMMDistribution.x0(x, t, sampler, denoising_params, {})
    assert not torch.any(torch.isnan(x0_hat))
    assert not torch.any(torch.isinf(x0_hat))
    assert torch.all(torch.abs(x0_hat) < 100)
