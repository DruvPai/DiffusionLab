import torch
from diffusionlab.utils import (
    scalar_derivative,
    pad_shape_front,
    pad_shape_back,
    vector_lstsq,
    logdet_pd,
    sqrt_psd,
)

# ===== Derivative Tests =====


def test_scalar_derivative_polynomial():
    # Test polynomial function
    def f(x):
        return x**2 + 2 * x + 1

    f_prime = scalar_derivative(f)

    x = torch.tensor(2.0)
    assert torch.allclose(f_prime(x), torch.tensor(6.0))  # d/dx(x^2 + 2x + 1) = 2x + 2

    x = torch.tensor(0.0)
    assert torch.allclose(f_prime(x), torch.tensor(2.0))

    x = torch.tensor(-1.0)
    assert torch.allclose(f_prime(x), torch.tensor(0.0))


def test_scalar_derivative_exponential():
    def f(x):
        return torch.exp(x)

    f_prime = scalar_derivative(f)

    x = torch.tensor(0.0)
    assert torch.allclose(f_prime(x), torch.tensor(1.0))

    x = torch.tensor(1.0)
    assert torch.allclose(f_prime(x), torch.exp(torch.tensor(1.0)))


def test_scalar_derivative_trigonometric():
    def f(x):
        return torch.sin(x)

    f_prime = scalar_derivative(f)

    x = torch.tensor(0.0)
    assert torch.allclose(f_prime(x), torch.tensor(1.0))

    x = torch.tensor(torch.pi / 2)
    assert torch.allclose(f_prime(x), torch.tensor(0.0), atol=1e-6)


def test_scalar_derivative_broadcasting():
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


def test_scalar_derivative_composition():
    def f(x):
        return torch.sin(x**2)

    f_prime = scalar_derivative(f)

    x = torch.tensor(0.0)
    assert torch.allclose(f_prime(x), torch.tensor(0.0))

    x = torch.tensor(1.0)
    expected = 2 * torch.cos(torch.tensor(1.0))
    assert torch.allclose(f_prime(x), expected)


# ===== Shape Padding Tests =====


def test_pad_shape_front():
    # Test scalar to higher dimensions
    x = torch.tensor(5.0)
    target_shape = torch.Size([2, 3, 4])
    padded = pad_shape_front(x, target_shape)
    assert padded.shape == torch.Size([1, 1, 1])
    assert torch.all(padded == x)

    # Test vector to higher dimensions
    x = torch.randn(3)
    target_shape = torch.Size([2, 3, 4, 3])
    padded = pad_shape_front(x, target_shape)
    assert padded.shape == torch.Size([1, 1, 1, 3])
    assert torch.all(padded.squeeze() == x)

    # Test matrix to higher dimensions
    x = torch.randn(2, 3)
    target_shape = torch.Size([4, 5, 2, 3])
    padded = pad_shape_front(x, target_shape)
    assert padded.shape == torch.Size([1, 1, 2, 3])
    assert torch.all(padded.squeeze() == x)


def test_pad_shape_back():
    # Test scalar to higher dimensions
    x = torch.tensor(5.0)
    target_shape = torch.Size([2, 3, 4])
    padded = pad_shape_back(x, target_shape)
    assert padded.shape == torch.Size([1, 1, 1])
    assert torch.all(padded == x)

    # Test vector to higher dimensions
    x = torch.randn(3)
    target_shape = torch.Size([3, 4, 5, 6])
    padded = pad_shape_back(x, target_shape)
    assert padded.shape == torch.Size([3, 1, 1, 1])
    assert torch.all(padded.squeeze() == x)

    # Test matrix to higher dimensions
    x = torch.randn(2, 3)
    target_shape = torch.Size([2, 3, 4, 5])
    padded = pad_shape_back(x, target_shape)
    assert padded.shape == torch.Size([2, 3, 1, 1])
    assert torch.all(padded.squeeze() == x)


def test_pad_shape_memory_efficiency():
    x = torch.randn(3, 4)
    target_shape = torch.Size([2, 3, 4, 5])

    padded_front = pad_shape_front(x, target_shape)
    assert padded_front.data_ptr() == x.data_ptr()

    padded_back = pad_shape_back(x, target_shape)
    assert padded_back.data_ptr() == x.data_ptr()


# ===== Linear Algebra Tests =====


def test_vector_lstsq():
    # Test simple 2D case
    A = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    y = torch.tensor([6.0, 9.0, 12.0])
    x = vector_lstsq(A, y)
    assert x.shape == torch.Size([2])
    assert torch.allclose(A @ x, y, atol=1e-5)

    # Test batched case
    batch_A = torch.stack([A, 2 * A])
    batch_y = torch.stack([y, 2 * y])
    batch_x = vector_lstsq(batch_A, batch_y)
    assert batch_x.shape == torch.Size([2, 2])
    assert torch.allclose(batch_A @ batch_x[..., None], batch_y[..., None], atol=1e-5)


def test_logdet_pd():
    # Test 2x2 case
    A = torch.tensor([[2.0, 0.5], [0.5, 2.0]])
    logdet = logdet_pd(A)
    expected = torch.log(torch.tensor(3.75))  # det([[2, 0.5], [0.5, 2]]) = 3.75
    assert torch.allclose(logdet, expected)

    # Test batched case
    batch_A = torch.stack([A, 2 * A])
    batch_logdet = logdet_pd(batch_A)
    expected = torch.tensor(
        [torch.log(torch.tensor(3.75)), torch.log(torch.tensor(3.75 * 4))]
    )
    assert torch.allclose(batch_logdet, expected)


def test_sqrt_psd():
    # Test identity matrix
    A = torch.eye(3)
    sqrt_A = sqrt_psd(A)
    assert torch.allclose(sqrt_A @ sqrt_A, A)

    # Test diagonal matrix
    A = torch.diag(torch.tensor([4.0, 9.0, 16.0]))
    sqrt_A = sqrt_psd(A)
    expected = torch.diag(torch.tensor([2.0, 3.0, 4.0]))
    assert torch.allclose(sqrt_A, expected)

    # Test symmetric PSD matrix
    A = torch.tensor([[2.0, 0.5], [0.5, 2.0]])
    sqrt_A = sqrt_psd(A)
    assert torch.allclose(sqrt_A @ sqrt_A, A)

    # Test zero matrix
    A = torch.zeros(3, 3)
    sqrt_A = sqrt_psd(A)
    assert torch.allclose(sqrt_A, torch.zeros_like(A))

    # Test matrix with very small positive eigenvalues
    A = torch.tensor([[1e-8, 0.0], [0.0, 1e-8]])
    sqrt_A = sqrt_psd(A)
    expected = torch.tensor([[1e-4, 0.0], [0.0, 1e-4]])
    assert torch.allclose(sqrt_A, expected, atol=1e-10)

    # Test rank-deficient matrix
    A = torch.tensor([[1.0, 1.0], [1.0, 1.0]])  # rank-1 matrix
    sqrt_A = sqrt_psd(A)
    assert torch.allclose(sqrt_A @ sqrt_A, A)
    # Verify rank deficiency by checking determinant is zero
    assert torch.allclose(torch.det(sqrt_A), torch.tensor(0.0), atol=1e-6)

    # Test higher dimensional symmetric PSD matrix
    A = torch.tensor([[4.0, 1.0, 0.0], [1.0, 5.0, 2.0], [0.0, 2.0, 6.0]])
    sqrt_A = sqrt_psd(A)
    assert torch.allclose(sqrt_A @ sqrt_A, A, atol=1e-6)
    assert torch.allclose(sqrt_A, sqrt_A.T)  # Result should be symmetric

    # Test batched case with different matrices
    batch_A = torch.stack(
        [
            torch.eye(2),  # identity
            torch.tensor([[4.0, 0.0], [0.0, 9.0]]),  # diagonal
            torch.tensor([[2.0, 1.0], [1.0, 2.0]]),  # dense symmetric
        ]
    )
    batch_sqrt_A = sqrt_psd(batch_A)
    assert torch.allclose(batch_sqrt_A @ batch_sqrt_A.mT, batch_A)

    # Test broadcasting with different batch dimensions
    A1 = torch.eye(2).expand(3, 4, 2, 2)  # Shape: (3, 4, 2, 2)
    sqrt_A1 = sqrt_psd(A1)
    assert sqrt_A1.shape == (3, 4, 2, 2)
    assert torch.allclose(sqrt_A1 @ sqrt_A1.mT, A1)
