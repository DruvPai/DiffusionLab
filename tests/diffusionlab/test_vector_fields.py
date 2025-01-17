import torch
from diffusionlab.utils import pad_shape_back
from diffusionlab.vector_fields import (
    VectorField,
    VectorFieldType,
    convert_vector_field_type,
)


def test_vector_field_creation():
    # Test basic vector field creation
    def f(x, t):
        return x * pad_shape_back(t, x.shape)

    vf = VectorField(f, VectorFieldType.SCORE)

    # Test calling
    x = torch.randn(10, 3)
    t = torch.ones(10)
    assert torch.allclose(vf(x, t), f(x, t))

    # Test type property
    assert vf.vector_field_type == VectorFieldType.SCORE


def test_vector_field_device_movement():
    # Test moving vector field to different device
    if not torch.cuda.is_available():
        return

    def f(x, t):
        return x * t

    vf = VectorField(f, VectorFieldType.SCORE)
    device = torch.device("cuda:0")
    vf = vf.to(device)

    x = torch.randn(10, 3, device=device)
    t = torch.ones(10, device=device)
    assert vf(x, t).device == device


def test_vector_field_nn_module():
    # Test vector field with nn.Module
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x, t):
            return self.linear(x) * t.unsqueeze(-1)

    net = Net()
    vf = VectorField(net, VectorFieldType.SCORE)

    x = torch.randn(10, 3)
    t = torch.ones(10)
    # Just test that it runs without error
    _ = vf(x, t)


def test_vector_field_conversion():
    batch_size = 10
    data_dim = 3

    # Create test data
    x = torch.randn(batch_size, data_dim)
    fx = torch.randn(batch_size, data_dim)
    alpha = torch.ones(batch_size)
    sigma = torch.ones(batch_size) * 0.5
    alpha_prime = -torch.ones(batch_size)
    sigma_prime = torch.ones(batch_size)

    # Test score to x0 and back
    x0_from_score = convert_vector_field_type(
        x,
        fx,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.SCORE,
        VectorFieldType.X0,
    )
    score_back = convert_vector_field_type(
        x,
        x0_from_score,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.X0,
        VectorFieldType.SCORE,
    )
    assert torch.allclose(fx, score_back)

    # Test score to eps and back
    eps_from_score = convert_vector_field_type(
        x,
        fx,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.SCORE,
        VectorFieldType.EPS,
    )
    score_back = convert_vector_field_type(
        x,
        eps_from_score,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.EPS,
        VectorFieldType.SCORE,
    )
    assert torch.allclose(fx, score_back)

    # Test score to v and back
    v_from_score = convert_vector_field_type(
        x,
        fx,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.SCORE,
        VectorFieldType.V,
    )
    score_back = convert_vector_field_type(
        x,
        v_from_score,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.V,
        VectorFieldType.SCORE,
    )
    assert torch.allclose(fx, score_back)


def test_vector_field_conversion_consistency():
    batch_size = 10
    data_dim = 3

    # Create test data
    x = torch.randn(batch_size, data_dim)
    fx = torch.randn(batch_size, data_dim)
    alpha = torch.ones(batch_size)
    sigma = torch.ones(batch_size) * 0.5
    alpha_prime = -torch.ones(batch_size)
    sigma_prime = torch.ones(batch_size)

    # Convert score to all other types
    x0 = convert_vector_field_type(
        x,
        fx,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.SCORE,
        VectorFieldType.X0,
    )
    eps = convert_vector_field_type(
        x,
        fx,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.SCORE,
        VectorFieldType.EPS,
    )
    v = convert_vector_field_type(
        x,
        fx,
        alpha,
        sigma,
        alpha_prime,
        sigma_prime,
        VectorFieldType.SCORE,
        VectorFieldType.V,
    )

    # Verify consistency equations
    # x = alpha * x0 + sigma * eps
    assert torch.allclose(x, alpha.unsqueeze(-1) * x0 + sigma.unsqueeze(-1) * eps)

    # v = alpha_prime * x0 + sigma_prime * eps
    assert torch.allclose(
        v, alpha_prime.unsqueeze(-1) * x0 + sigma_prime.unsqueeze(-1) * eps
    )
