import torch
from diffusionlab.utils import (
    scalar_derivative,
    pad_shape_front,
    pad_shape_back,
    logdet_pd,
    sqrt_psd,
)


class TestScalarDerivative:
    def test_polynomial(self):
        # Test polynomial function
        def f(x):
            return x**2 + 2 * x + 1

        f_prime = scalar_derivative(f)

        x = torch.tensor(2.0)
        assert torch.allclose(
            f_prime(x), torch.tensor(6.0)
        )  # d/dx(x^2 + 2x + 1) = 2x + 2

        x = torch.tensor(0.0)
        assert torch.allclose(f_prime(x), torch.tensor(2.0))

        x = torch.tensor(-1.0)
        assert torch.allclose(f_prime(x), torch.tensor(0.0))

    def test_exponential(self):
        def f(x):
            return torch.exp(x)

        f_prime = scalar_derivative(f)

        x = torch.tensor(0.0)
        assert torch.allclose(f_prime(x), torch.tensor(1.0))

        x = torch.tensor(1.0)
        assert torch.allclose(f_prime(x), torch.exp(torch.tensor(1.0)))

    def test_trigonometric(self):
        def f(x):
            return torch.sin(x)

        f_prime = scalar_derivative(f)

        x = torch.tensor(0.0)
        assert torch.allclose(f_prime(x), torch.tensor(1.0))

        x = torch.tensor(torch.pi / 2)
        assert torch.allclose(f_prime(x), torch.tensor(0.0), atol=1e-6)

    def test_broadcasting(self):
        def f(x):
            return x**2

        f_prime = scalar_derivative(f)

        x = torch.tensor([1.0, 2.0, 3.0])
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(f_prime(x), expected)

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expected = torch.tensor([[2.0, 4.0], [6.0, 8.0]])
        assert torch.allclose(f_prime(x), expected)

        x = torch.randn((10, 3, 5, 4, 3))
        expected = 2 * x
        assert torch.allclose(f_prime(x), expected)

    def test_composition(self):
        def f(x):
            return torch.sin(x**2)

        f_prime = scalar_derivative(f)

        x = torch.tensor(0.0)
        assert torch.allclose(f_prime(x), torch.tensor(0.0))

        x = torch.tensor(1.0)
        expected = 2 * torch.cos(torch.tensor(1.0))
        assert torch.allclose(f_prime(x), expected)


class TestPadShapeFront:
    def test_scalar_to_higher_dimensions(self):
        x = torch.tensor(5.0)
        target_shape = torch.Size([2, 3, 4])
        padded = pad_shape_front(x, target_shape)
        assert padded.shape == torch.Size([1, 1, 1])
        assert torch.all(padded == x)

    def test_vector_to_higher_dimensions(self):
        x = torch.randn(3)
        target_shape = torch.Size([2, 3, 4, 3])
        padded = pad_shape_front(x, target_shape)
        assert padded.shape == torch.Size([1, 1, 1, 3])
        assert torch.all(padded.squeeze() == x)

    def test_matrix_to_higher_dimensions(self):
        x = torch.randn(2, 3)
        target_shape = torch.Size([4, 5, 2, 3])
        padded = pad_shape_front(x, target_shape)
        assert padded.shape == torch.Size([1, 1, 2, 3])
        assert torch.all(padded.squeeze() == x)

    def test_memory_efficiency(self):
        x = torch.randn(3, 4)
        target_shape = torch.Size([2, 3, 4, 5])
        padded = pad_shape_front(x, target_shape)
        assert padded.data_ptr() == x.data_ptr()


class TestPadShapeBack:
    def test_scalar_to_higher_dimensions(self):
        x = torch.tensor(5.0)
        target_shape = torch.Size([2, 3, 4])
        padded = pad_shape_back(x, target_shape)
        assert padded.shape == torch.Size([1, 1, 1])
        assert torch.all(padded == x)

    def test_vector_to_higher_dimensions(self):
        x = torch.randn(3)
        target_shape = torch.Size([3, 4, 5, 6])
        padded = pad_shape_back(x, target_shape)
        assert padded.shape == torch.Size([3, 1, 1, 1])
        assert torch.all(padded.squeeze() == x)

    def test_matrix_to_higher_dimensions(self):
        x = torch.randn(2, 3)
        target_shape = torch.Size([2, 3, 4, 5])
        padded = pad_shape_back(x, target_shape)
        assert padded.shape == torch.Size([2, 3, 1, 1])
        assert torch.all(padded.squeeze() == x)

    def test_memory_efficiency(self):
        x = torch.randn(3, 4)
        target_shape = torch.Size([2, 3, 4, 5])
        padded = pad_shape_back(x, target_shape)
        assert padded.data_ptr() == x.data_ptr()


class TestLogdetPd:
    def test_2x2_case(self):
        A = torch.tensor([[2.0, 0.5], [0.5, 2.0]])
        logdet = logdet_pd(A)
        expected = torch.log(torch.tensor(3.75))  # det([[2, 0.5], [0.5, 2]]) = 3.75
        assert torch.allclose(logdet, expected)

    def test_batched_case(self):
        A = torch.tensor([[2.0, 0.5], [0.5, 2.0]])
        batch_A = torch.stack([A, 2 * A])
        batch_logdet = logdet_pd(batch_A)
        expected = torch.tensor(
            [torch.log(torch.tensor(3.75)), torch.log(torch.tensor(3.75 * 4))]
        )
        assert torch.allclose(batch_logdet, expected)


class TestSqrtPsd:
    def test_identity_matrix(self):
        A = torch.eye(3)
        sqrt_A = sqrt_psd(A)
        assert torch.allclose(sqrt_A @ sqrt_A, A)

    def test_diagonal_matrix(self):
        A = torch.diag(torch.tensor([4.0, 9.0, 16.0]))
        sqrt_A = sqrt_psd(A)
        expected = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
        assert torch.allclose(sqrt_A, expected)

    def test_symmetric_psd_matrix(self):
        A = torch.tensor([[2.0, 0.5], [0.5, 2.0]])
        sqrt_A = sqrt_psd(A)
        assert torch.allclose(sqrt_A @ sqrt_A, A)

    def test_zero_matrix(self):
        A = torch.zeros(3, 3)
        sqrt_A = sqrt_psd(A)
        assert torch.allclose(sqrt_A, torch.zeros_like(A))

    def test_small_eigenvalues(self):
        A = torch.tensor([[1e-8, 0.0], [0.0, 1e-8]])
        sqrt_A = sqrt_psd(A)
        expected = torch.tensor([[1e-4, 0.0], [0.0, 1e-4]])
        assert torch.allclose(sqrt_A, expected, atol=1e-10)

    def test_rank_deficient_matrix(self):
        A = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # rank-1 matrix
        sqrt_A = sqrt_psd(A)
        assert torch.allclose(sqrt_A @ sqrt_A, A)
        # Verify rank deficiency by checking determinant is zero
        assert torch.allclose(torch.det(sqrt_A), torch.tensor(0.0), atol=1e-6)

    def test_higher_dimensional_matrix(self):
        A = torch.tensor([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]])
        sqrt_A = sqrt_psd(A)
        assert torch.allclose(sqrt_A @ sqrt_A, A, atol=1e-6)
        assert torch.allclose(sqrt_A, sqrt_A.T)  # Result should be symmetric

    def test_batched_case(self):
        batch_A = torch.stack(
            [
                torch.eye(2),  # identity
                torch.tensor([[4.0, 0.0], [0.0, 9.0]]),  # diagonal
                torch.tensor([[2.0, 1.0], [1.0, 2.0]]),  # dense symmetric
            ]
        )
        batch_sqrt_A = sqrt_psd(batch_A)
        assert torch.allclose(batch_sqrt_A @ batch_sqrt_A.mT, batch_A)

    def test_broadcasting(self):
        A1 = torch.eye(2).expand(3, 4, 2, 2)  # Shape: (3, 4, 2, 2)
        sqrt_A1 = sqrt_psd(A1)
        assert sqrt_A1.shape == (3, 4, 2, 2)
        assert torch.allclose(sqrt_A1 @ sqrt_A1.mT, A1)
