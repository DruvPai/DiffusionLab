import torch
from diffusionlab.utils import pad_shape_back
from diffusionlab.vector_fields import (
    VectorField,
    VectorFieldType,
    convert_vector_field_type,
)


class TestVectorField:
    def test_vector_field_creation(self):
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

    def test_vector_field_nn_module(self):
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


class TestVectorFieldConversion:
    def test_vector_field_conversion_score_to_others_and_back(self):
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

    def test_vector_field_conversion_consistency(self):
        """Test that conversions between all vector field types are consistent with the equations."""
        batch_size = 10
        data_dim = 3

        # Create test data
        x = torch.randn(batch_size, data_dim)
        alpha = torch.ones(batch_size)
        sigma = torch.ones(batch_size) * 0.5
        alpha_prime = -torch.ones(batch_size)
        sigma_prime = torch.ones(batch_size)

        # Part 1: Test conversions starting from SCORE
        score = torch.randn(batch_size, data_dim)

        # Convert score to all other types
        x0_from_score = convert_vector_field_type(
            x,
            score,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.SCORE,
            VectorFieldType.X0,
        )
        eps_from_score = convert_vector_field_type(
            x,
            score,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.SCORE,
            VectorFieldType.EPS,
        )
        v_from_score = convert_vector_field_type(
            x,
            score,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.SCORE,
            VectorFieldType.V,
        )

        # Verify consistency equations for score-derived fields
        # x = alpha * x0 + sigma * eps
        assert torch.allclose(
            x,
            alpha.unsqueeze(-1) * x0_from_score + sigma.unsqueeze(-1) * eps_from_score,
        )

        # v = alpha_prime * x0 + sigma_prime * eps
        assert torch.allclose(
            v_from_score,
            alpha_prime.unsqueeze(-1) * x0_from_score
            + sigma_prime.unsqueeze(-1) * eps_from_score,
        )

        # Part 2: Test conversions starting from EPS
        eps = torch.randn(batch_size, data_dim)

        # Test eps to other types and back
        v_from_eps = convert_vector_field_type(
            x,
            eps,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.EPS,
            VectorFieldType.V,
        )
        eps_back = convert_vector_field_type(
            x,
            v_from_eps,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.V,
            VectorFieldType.EPS,
        )
        assert torch.allclose(eps, eps_back)

        x0_from_eps = convert_vector_field_type(
            x,
            eps,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.EPS,
            VectorFieldType.X0,
        )
        eps_back = convert_vector_field_type(
            x,
            x0_from_eps,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.X0,
            VectorFieldType.EPS,
        )
        assert torch.allclose(eps, eps_back)

        # Verify consistency equations for eps-derived fields
        # v = alpha_prime * x0 + sigma_prime * eps
        v_expected = (
            alpha_prime.unsqueeze(-1) * x0_from_eps + sigma_prime.unsqueeze(-1) * eps
        )
        assert torch.allclose(v_from_eps, v_expected)

        # x = alpha * x0 + sigma * eps
        x_expected = alpha.unsqueeze(-1) * x0_from_eps + sigma.unsqueeze(-1) * eps
        assert torch.allclose(x, x_expected)

        # Part 3: Test conversions starting from V
        v = torch.randn(batch_size, data_dim)

        # Test v to other types and back
        eps_from_v = convert_vector_field_type(
            x,
            v,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.V,
            VectorFieldType.EPS,
        )
        v_back = convert_vector_field_type(
            x,
            eps_from_v,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.EPS,
            VectorFieldType.V,
        )
        assert torch.allclose(v, v_back)

        x0_from_v = convert_vector_field_type(
            x,
            v,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.V,
            VectorFieldType.X0,
        )
        v_back = convert_vector_field_type(
            x,
            x0_from_v,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.X0,
            VectorFieldType.V,
        )
        assert torch.allclose(v, v_back)

        # Verify consistency equations for v-derived fields
        # v = alpha_prime * x0 + sigma_prime * eps
        v_expected = (
            alpha_prime.unsqueeze(-1) * x0_from_v
            + sigma_prime.unsqueeze(-1) * eps_from_v
        )
        assert torch.allclose(v, v_expected)

        # x = alpha * x0 + sigma * eps
        x_expected = alpha.unsqueeze(-1) * x0_from_v + sigma.unsqueeze(-1) * eps_from_v
        assert torch.allclose(x, x_expected)

        # Part 4: Test conversions starting from X0
        x0 = torch.randn(batch_size, data_dim)

        # Test x0 to other types and back
        eps_from_x0 = convert_vector_field_type(
            x,
            x0,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.X0,
            VectorFieldType.EPS,
        )
        x0_back = convert_vector_field_type(
            x,
            eps_from_x0,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.EPS,
            VectorFieldType.X0,
        )
        assert torch.allclose(x0, x0_back)

        v_from_x0 = convert_vector_field_type(
            x,
            x0,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.X0,
            VectorFieldType.V,
        )
        x0_back = convert_vector_field_type(
            x,
            v_from_x0,
            alpha,
            sigma,
            alpha_prime,
            sigma_prime,
            VectorFieldType.V,
            VectorFieldType.X0,
        )
        assert torch.allclose(x0, x0_back)

        # Verify consistency equations for x0-derived fields
        # v = alpha_prime * x0 + sigma_prime * eps
        v_expected = (
            alpha_prime.unsqueeze(-1) * x0 + sigma_prime.unsqueeze(-1) * eps_from_x0
        )
        assert torch.allclose(v_from_x0, v_expected)

        # x = alpha * x0 + sigma * eps
        x_expected = alpha.unsqueeze(-1) * x0 + sigma.unsqueeze(-1) * eps_from_x0
        assert torch.allclose(x, x_expected)
