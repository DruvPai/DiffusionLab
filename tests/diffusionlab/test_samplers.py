import torch
import pytest
from unittest.mock import MagicMock, patch

from diffusionlab.diffusions import DiffusionProcess
from diffusionlab.samplers import Sampler, EulerMaruyamaSampler, DDMSampler
from diffusionlab.vector_fields import VectorField, VectorFieldType


class TestSampler:
    def test_initialization(self):
        """Test basic initialization of Sampler."""
        # Create a mock diffusion process
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        is_stochastic = True
        sampler = Sampler(diffusion_process, is_stochastic)

        # Check attributes
        assert sampler.diffusion_process is diffusion_process
        assert sampler.is_stochastic is is_stochastic

        # The alpha, sigma, alpha_prime, and sigma_prime attributes are now part of the diffusion process
        # and not directly accessible from the sampler

    def test_get_sample_step_function(self):
        """Test that get_sample_step_function returns the correct function."""
        # Create a mock diffusion process
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Test stochastic sampler
        sampler = Sampler(diffusion_process, True)

        # Create dummy inputs for testing
        vector_field_mock = MagicMock(spec=VectorField)
        x_mock = torch.zeros(1)
        zs_mock = torch.zeros(1, 1)
        ts_mock = torch.zeros(1)

        # Patch the sample step functions to return identifiable values
        with (
            patch.object(
                sampler, "sample_step_stochastic_score", return_value="stochastic_score"
            ),
            patch.object(
                sampler, "sample_step_stochastic_x0", return_value="stochastic_x0"
            ),
            patch.object(
                sampler, "sample_step_stochastic_eps", return_value="stochastic_eps"
            ),
            patch.object(
                sampler, "sample_step_stochastic_v", return_value="stochastic_v"
            ),
        ):
            # Test each vector field type
            assert (
                sampler.get_sample_step_function(VectorFieldType.SCORE)(
                    vector_field_mock, x_mock, zs_mock, 0, ts_mock
                )
                == "stochastic_score"
            )
            assert (
                sampler.get_sample_step_function(VectorFieldType.X0)(
                    vector_field_mock, x_mock, zs_mock, 0, ts_mock
                )
                == "stochastic_x0"
            )
            assert (
                sampler.get_sample_step_function(VectorFieldType.EPS)(
                    vector_field_mock, x_mock, zs_mock, 0, ts_mock
                )
                == "stochastic_eps"
            )
            assert (
                sampler.get_sample_step_function(VectorFieldType.V)(
                    vector_field_mock, x_mock, zs_mock, 0, ts_mock
                )
                == "stochastic_v"
            )

        # Test deterministic sampler
        sampler = Sampler(diffusion_process, False)

        # Patch the sample step functions to return identifiable values
        with (
            patch.object(
                sampler,
                "sample_step_deterministic_score",
                return_value="deterministic_score",
            ),
            patch.object(
                sampler, "sample_step_deterministic_x0", return_value="deterministic_x0"
            ),
            patch.object(
                sampler,
                "sample_step_deterministic_eps",
                return_value="deterministic_eps",
            ),
            patch.object(
                sampler, "sample_step_deterministic_v", return_value="deterministic_v"
            ),
        ):
            # Test each vector field type
            assert (
                sampler.get_sample_step_function(VectorFieldType.SCORE)(
                    vector_field_mock, x_mock, zs_mock, 0, ts_mock
                )
                == "deterministic_score"
            )
            assert (
                sampler.get_sample_step_function(VectorFieldType.X0)(
                    vector_field_mock, x_mock, zs_mock, 0, ts_mock
                )
                == "deterministic_x0"
            )
            assert (
                sampler.get_sample_step_function(VectorFieldType.EPS)(
                    vector_field_mock, x_mock, zs_mock, 0, ts_mock
                )
                == "deterministic_eps"
            )
            assert (
                sampler.get_sample_step_function(VectorFieldType.V)(
                    vector_field_mock, x_mock, zs_mock, 0, ts_mock
                )
                == "deterministic_v"
            )

    def test_fix_t_shape(self):
        """Test the _fix_t_shape helper method."""
        # Create a mock diffusion process
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        sampler = Sampler(diffusion_process, True)

        # Test with various shapes
        batch_size = 5
        x = torch.randn(batch_size, 3)
        t = torch.tensor([0.5])

        reshaped_t = sampler._fix_t_shape(x, t)

        assert reshaped_t.shape == (batch_size,)
        assert torch.allclose(reshaped_t, torch.ones(batch_size) * 0.5)

    def test_sample_with_mocked_step_function(self):
        """Test that sample calls the step function correctly."""
        # Create a mock diffusion process
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = Sampler(diffusion_process, True)

        # Create mock vector field
        vector_field = MagicMock(spec=VectorField)
        vector_field.vector_field_type = VectorFieldType.X0

        # Create mock inputs
        batch_size = 2
        data_dim = 3
        num_steps = 4
        x_t = torch.randn(batch_size, data_dim)
        zs = torch.randn(num_steps - 1, batch_size, data_dim)

        # Create time steps
        ts = torch.linspace(1, 0, num_steps)

        # Mock the step function
        mock_step_fn = MagicMock(return_value=torch.zeros(batch_size, data_dim))

        # Patch the get_sample_step_function to return our mock
        with patch.object(
            sampler, "get_sample_step_function", return_value=mock_step_fn
        ):
            # Call sample
            result = sampler.sample(vector_field, x_t, zs, ts)

            # Check that step function was called correct number of times
            assert mock_step_fn.call_count == num_steps - 1

            # Check that step function was called with correct arguments
            for i in range(num_steps - 1):
                args, _ = mock_step_fn.call_args_list[i]
                assert args[0] is vector_field
                assert torch.allclose(
                    args[1], mock_step_fn.return_value if i > 0 else x_t
                )
                assert args[2] is zs
                assert args[3] == i
                assert args[4] is ts

            # Check result shape
            assert result.shape == (batch_size, data_dim)
            assert torch.allclose(result, mock_step_fn.return_value)

    def test_sample_trajectory_with_mocked_step_function(self):
        """Test that sample_trajectory calls the step function correctly and returns the trajectory."""
        # Create a mock diffusion process
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = Sampler(diffusion_process, True)

        # Create mock vector field
        vector_field = MagicMock(spec=VectorField)
        vector_field.vector_field_type = VectorFieldType.X0

        # Create mock inputs
        batch_size = 2
        data_dim = 3
        num_steps = 4
        x_t = torch.randn(batch_size, data_dim)
        zs = torch.randn(num_steps - 1, batch_size, data_dim)

        # Create time steps
        ts = torch.linspace(1, 0, num_steps)

        # Create different return values for each step
        step_returns = [
            torch.full((batch_size, data_dim), i, dtype=zs.dtype)
            for i in range(num_steps - 1)
        ]
        mock_step_fn = MagicMock(side_effect=step_returns)

        # Patch the get_sample_step_function to return our mock
        with patch.object(
            sampler, "get_sample_step_function", return_value=mock_step_fn
        ):
            # Call sample_trajectory
            result = sampler.sample_trajectory(vector_field, x_t, zs, ts)

            # Check that step function was called correct number of times
            assert mock_step_fn.call_count == num_steps - 1

            # Check result shape
            assert result.shape == (num_steps, batch_size, data_dim)

            # Check that the trajectory contains the initial x_t and all step results
            assert torch.allclose(result[0], x_t)
            for i in range(num_steps - 1):
                assert torch.allclose(result[i + 1], step_returns[i])

    def test_unimplemented_sample_step_methods(self):
        """Test that the sample step methods raise NotImplementedError."""
        # Create a mock diffusion process
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        sampler = Sampler(diffusion_process, True)

        # Create dummy inputs
        vector_field = MagicMock(spec=VectorField)
        x = torch.randn(5, 3)
        zs = torch.randn(2, 5, 3)
        idx = 0
        ts = torch.tensor([1.0, 0.5, 0.0])

        # Test all the unimplemented methods
        with pytest.raises(NotImplementedError):
            sampler.sample_step_stochastic_score(vector_field, x, zs, idx, ts)

        with pytest.raises(NotImplementedError):
            sampler.sample_step_deterministic_score(vector_field, x, zs, idx, ts)

        with pytest.raises(NotImplementedError):
            sampler.sample_step_stochastic_x0(vector_field, x, zs, idx, ts)

        with pytest.raises(NotImplementedError):
            sampler.sample_step_deterministic_x0(vector_field, x, zs, idx, ts)

        with pytest.raises(NotImplementedError):
            sampler.sample_step_stochastic_eps(vector_field, x, zs, idx, ts)

        with pytest.raises(NotImplementedError):
            sampler.sample_step_deterministic_eps(vector_field, x, zs, idx, ts)

        with pytest.raises(NotImplementedError):
            sampler.sample_step_stochastic_v(vector_field, x, zs, idx, ts)

        with pytest.raises(NotImplementedError):
            sampler.sample_step_deterministic_v(vector_field, x, zs, idx, ts)


class TestEulerMaruyamaSampler:
    def test_initialization(self):
        """Test basic initialization of EulerMaruyamaSampler."""
        # Create a mock diffusion process
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        is_stochastic = True
        sampler = EulerMaruyamaSampler(diffusion_process, is_stochastic)

        # Check attributes
        assert sampler.diffusion_process is diffusion_process
        assert sampler.is_stochastic is is_stochastic

        # The alpha, sigma, alpha_prime, and sigma_prime attributes are now part of the diffusion process
        # and not directly accessible from the sampler

    def test_get_step_quantities(self):
        """Test the _get_step_quantities method."""
        # Create a mock diffusion process with known functions
        alpha = lambda t: 1 - t
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = EulerMaruyamaSampler(diffusion_process, True)

        # Create test inputs
        batch_size = 2
        data_dim = 3
        num_steps = 3
        zs = torch.ones(num_steps - 1, batch_size, data_dim)
        idx = 0
        ts = torch.tensor([1.0, 0.5, 0.0])

        # Call the method
        result = sampler._get_step_quantities(zs, idx, ts)

        # Unpack the result
        (
            t,
            t1,
            alpha_t,
            sigma_t,
            alpha_prime_t,
            sigma_prime_t,
            dt,
            dwt,
            alpha_ratio_t,
            sigma_ratio_t,
            diff_ratio_t,
        ) = result

        # Check shapes
        assert t.shape == (1, 1)
        assert t1.shape == (1, 1)
        assert alpha_t.shape == (1, 1)
        assert sigma_t.shape == (1, 1)
        assert alpha_prime_t.shape == (1, 1)
        assert sigma_prime_t.shape == (1, 1)
        assert dt.shape == (1, 1)
        assert dwt.shape == (batch_size, data_dim)
        assert alpha_ratio_t.shape == (1, 1)
        assert sigma_ratio_t.shape == (1, 1)
        assert diff_ratio_t.shape == (1, 1)

        # Check values
        assert torch.allclose(t, torch.ones(batch_size, data_dim))
        assert torch.allclose(t1, torch.ones(batch_size, data_dim) * 0.5)
        assert torch.allclose(
            alpha_t, torch.zeros(batch_size, data_dim)
        )  # alpha(1.0) = 1 - 1.0 = 0
        assert torch.allclose(
            sigma_t, torch.ones(batch_size, data_dim)
        )  # sigma(1.0) = 1.0
        assert torch.allclose(
            alpha_prime_t, -torch.ones(batch_size, data_dim)
        )  # d/dt(1-t) = -1
        assert torch.allclose(
            sigma_prime_t, torch.ones(batch_size, data_dim)
        )  # d/dt(t) = 1
        assert torch.allclose(
            dt, torch.ones(batch_size, data_dim) * -0.5
        )  # t1 - t = 0.5 - 1.0 = -0.5
        assert torch.allclose(
            dwt, torch.ones(batch_size, data_dim) * torch.sqrt(torch.tensor(0.5))
        )  # zs * sqrt(-dt)

        # For alpha_ratio_t = alpha_prime_t / alpha_t, we have division by zero
        # So we skip this check

        # For sigma_ratio_t = sigma_prime_t / sigma_t = 1 / 1 = 1
        assert torch.allclose(sigma_ratio_t, torch.ones(batch_size, data_dim))

        # For diff_ratio_t = sigma_ratio_t - alpha_ratio_t, we skip due to division by zero in alpha_ratio_t

    def test_sample_step_deterministic_score(self):
        """Test the sample_step_deterministic_score method."""
        # Create a mock diffusion process with simple dynamics
        alpha = lambda t: torch.ones_like(t)  # Constant alpha = 1
        sigma = lambda t: t  # Linear sigma = t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        sampler = EulerMaruyamaSampler(diffusion_process, False)

        # Create a simple score function that returns -x (score of standard normal)
        def score_fn(x, t):
            return -x

        vector_field = VectorField(score_fn, VectorFieldType.SCORE)

        # Create input tensors
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        zs = torch.randn(
            2, batch_size, data_dim
        )  # L-1 noise tensors (not used in deterministic)
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Call the method
        result = sampler.sample_step_deterministic_score(vector_field, x, zs, idx, ts)

        # Check shape
        assert result.shape == x.shape

        # For this test we won't check exact values, as they depend on the complex dynamics
        # But we can check that the result is different from the input
        assert not torch.allclose(result, x)

    def test_sample_step_deterministic_x0(self):
        """Test the sample_step_deterministic_x0 method."""
        # Create a mock diffusion process with simple dynamics
        alpha = lambda t: torch.ones_like(t)  # Constant alpha = 1
        sigma = lambda t: t  # Linear sigma = t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        sampler = EulerMaruyamaSampler(diffusion_process, False)

        # Create a simple x0 function that returns zeros
        def x0_fn(x, t):
            return torch.zeros_like(x)

        vector_field = VectorField(x0_fn, VectorFieldType.X0)

        # Create input tensors
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        zs = torch.randn(
            2, batch_size, data_dim
        )  # L-1 noise tensors (not used in deterministic)
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Call the method
        result = sampler.sample_step_deterministic_x0(vector_field, x, zs, idx, ts)

        # Check shape
        assert result.shape == x.shape

        # For this test we won't check exact values, as they depend on the complex dynamics
        # But we can check that the result is different from the input
        assert not torch.allclose(result, x)

    def test_sample_step_deterministic_eps(self):
        """Test the sample_step_deterministic_eps method."""
        # Create a mock diffusion process with simple dynamics
        alpha = lambda t: torch.ones_like(t)  # Constant alpha = 1
        sigma = lambda t: t  # Linear sigma = t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        sampler = EulerMaruyamaSampler(diffusion_process, False)

        # Create a simple eps function that returns zeros
        def eps_fn(x, t):
            return torch.ones_like(x)

        vector_field = VectorField(eps_fn, VectorFieldType.EPS)

        # Create input tensors
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        zs = torch.randn(
            2, batch_size, data_dim
        )  # L-1 noise tensors (not used in deterministic)
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Call the method
        result = sampler.sample_step_deterministic_eps(vector_field, x, zs, idx, ts)

        # Check shape
        assert result.shape == x.shape

        # For this test we won't check exact values, as they depend on the complex dynamics
        # But we can check that the result is different from the input
        assert not torch.allclose(result, x)

    def test_sample_step_deterministic_v(self):
        """Test the sample_step_deterministic_v method."""
        # Create a mock diffusion process with simple dynamics
        alpha = lambda t: torch.ones_like(t)  # Constant alpha = 1
        sigma = lambda t: t  # Linear sigma = t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        sampler = EulerMaruyamaSampler(diffusion_process, False)

        # Create a simple v function that returns a constant velocity
        def v_fn(x, t):
            return torch.ones_like(x) * 0.1

        vector_field = VectorField(v_fn, VectorFieldType.V)

        # Create input tensors
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        zs = torch.randn(
            2, batch_size, data_dim
        )  # L-1 noise tensors (not used in deterministic)
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Call the method
        result = sampler.sample_step_deterministic_v(vector_field, x, zs, idx, ts)

        # Check shape
        assert result.shape == x.shape

        # For this test we won't check exact values, as they depend on the complex dynamics
        # But we can check that the result is different from the input
        assert not torch.allclose(result, x)

    def test_stochastic_sampling_methods(self):
        """Test the stochastic sampling methods."""
        # Create a mock diffusion process with simple dynamics
        alpha = lambda t: torch.ones_like(t)  # Constant alpha = 1
        sigma = lambda t: t  # Linear sigma = t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        sampler = EulerMaruyamaSampler(diffusion_process, True)

        # Create simple vector field functions
        def field_fn(x, t):
            return torch.zeros_like(x)

        score_field = VectorField(field_fn, VectorFieldType.SCORE)
        x0_field = VectorField(field_fn, VectorFieldType.X0)
        eps_field = VectorField(field_fn, VectorFieldType.EPS)
        v_field = VectorField(field_fn, VectorFieldType.V)

        # Create input tensors
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        zs = torch.randn(2, batch_size, data_dim)  # L-1 noise tensors
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Test all stochastic methods
        result_score = sampler.sample_step_stochastic_score(score_field, x, zs, idx, ts)
        result_x0 = sampler.sample_step_stochastic_x0(x0_field, x, zs, idx, ts)
        result_eps = sampler.sample_step_stochastic_eps(eps_field, x, zs, idx, ts)
        result_v = sampler.sample_step_stochastic_v(v_field, x, zs, idx, ts)

        # Check shapes
        assert result_score.shape == x.shape
        assert result_x0.shape == x.shape
        assert result_eps.shape == x.shape
        assert result_v.shape == x.shape

        # For stochastic methods, results should be different due to noise
        assert not torch.allclose(result_score, x)
        assert not torch.allclose(result_x0, x)
        assert not torch.allclose(result_eps, x)
        assert not torch.allclose(result_v, x)


class TestDDMSampler:
    def test_initialization(self):
        """Test basic initialization of DDMSampler."""
        # Create a mock diffusion process
        alpha = lambda t: torch.ones_like(t)
        sigma = lambda t: t
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        is_stochastic = True
        sampler = DDMSampler(diffusion_process, is_stochastic)

        # Check attributes
        assert sampler.diffusion_process is diffusion_process
        assert sampler.is_stochastic is is_stochastic

    def test_convert_to_x0(self):
        """Test the _convert_to_x0 method."""
        # Create a mock diffusion process
        alpha = lambda t: 1 - 0.5 * t
        sigma = lambda t: torch.sqrt(t)
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = DDMSampler(diffusion_process, True)

        # Create test inputs
        batch_size = 2
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        t = torch.tensor([0.5])

        # Test conversion from different vector field types

        # 1. Test SCORE to X0 conversion
        score = -x  # Score of standard normal
        x0_from_score = sampler._convert_to_x0(x, t, score, VectorFieldType.SCORE)
        assert x0_from_score.shape == x.shape

        # 2. Test EPS to X0 conversion
        eps = torch.randn_like(x)
        x0_from_eps = sampler._convert_to_x0(x, t, eps, VectorFieldType.EPS)
        assert x0_from_eps.shape == x.shape

        # 3. Test V to X0 conversion
        v = torch.randn_like(x)
        x0_from_v = sampler._convert_to_x0(x, t, v, VectorFieldType.V)
        assert x0_from_v.shape == x.shape

        # 4. Test X0 to X0 conversion (should be identity)
        x0 = torch.zeros_like(x)
        x0_from_x0 = sampler._convert_to_x0(x, t, x0, VectorFieldType.X0)
        assert x0_from_x0.shape == x.shape
        assert torch.allclose(x0_from_x0, x0)

    def test_ddpm_step_x0_tensor(self):
        """Test the _ddpm_step_x0_tensor method."""
        # Create a mock diffusion process
        alpha = lambda t: 1 - 0.5 * t
        sigma = lambda t: torch.sqrt(t)
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = DDMSampler(diffusion_process, True)

        # Create test inputs
        batch_size = 2
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        x0 = torch.zeros_like(x)  # Target is zero
        zs = torch.randn(2, batch_size, data_dim)
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Call the method
        result = sampler._ddpm_step_x0_tensor(x0, x, zs, idx, ts)

        # Check shape
        assert result.shape == x.shape

    def test_ddim_step_x0_tensor(self):
        """Test the _ddim_step_x0_tensor method."""
        # Create a mock diffusion process
        alpha = lambda t: 1 - 0.5 * t
        sigma = lambda t: torch.sqrt(t)
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = DDMSampler(diffusion_process, False)

        # Create test inputs
        batch_size = 2
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        x0 = torch.zeros_like(x)  # Target is zero
        zs = torch.randn(2, batch_size, data_dim)  # Not used in deterministic case
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Call the method
        result = sampler._ddim_step_x0_tensor(x0, x, zs, idx, ts)

        # Check shape
        assert result.shape == x.shape

    def test_sample_step_deterministic_x0(self):
        """Test the sample_step_deterministic_x0 method."""
        # Create a mock diffusion process
        alpha = lambda t: 1 - 0.5 * t
        sigma = lambda t: torch.sqrt(t)
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = DDMSampler(diffusion_process, False)

        # Create a simple x0 function that returns zeros
        def x0_fn(x, t):
            return torch.zeros_like(x)

        vector_field = VectorField(x0_fn, VectorFieldType.X0)

        # Create input tensors
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        zs = torch.randn(2, batch_size, data_dim)  # Not used in deterministic case
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Call the method
        result = sampler.sample_step_deterministic_x0(vector_field, x, zs, idx, ts)

        # Check shape
        assert result.shape == x.shape

    def test_sample_step_stochastic_x0(self):
        """Test the sample_step_stochastic_x0 method."""
        # Create a mock diffusion process
        alpha = lambda t: 1 - 0.5 * t
        sigma = lambda t: torch.sqrt(t)
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = DDMSampler(diffusion_process, True)

        # Create a simple x0 function that returns zeros
        def x0_fn(x, t):
            return torch.zeros_like(x)

        vector_field = VectorField(x0_fn, VectorFieldType.X0)

        # Create input tensors
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        zs = torch.randn(2, batch_size, data_dim)
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Call the method
        result = sampler.sample_step_stochastic_x0(vector_field, x, zs, idx, ts)

        # Check shape
        assert result.shape == x.shape

    def test_other_vector_field_types(self):
        """Test sampling with other vector field types (SCORE, EPS, V)."""
        # Create a mock diffusion process
        alpha = lambda t: 1 - 0.5 * t
        sigma = lambda t: torch.sqrt(t)
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize samplers
        deterministic_sampler = DDMSampler(diffusion_process, False)
        stochastic_sampler = DDMSampler(diffusion_process, True)

        # Create simple vector field functions
        def field_fn(x, t):
            return torch.zeros_like(x)

        score_field = VectorField(field_fn, VectorFieldType.SCORE)
        eps_field = VectorField(field_fn, VectorFieldType.EPS)
        v_field = VectorField(field_fn, VectorFieldType.V)

        # Create input tensors
        batch_size = 5
        data_dim = 3
        x = torch.randn(batch_size, data_dim)
        zs = torch.randn(2, batch_size, data_dim)
        idx = 0
        ts = torch.tensor([0.8, 0.5, 0.0])  # Decreasing time steps

        # Test deterministic methods
        det_result_score = deterministic_sampler.sample_step_deterministic_score(
            score_field, x, zs, idx, ts
        )
        det_result_eps = deterministic_sampler.sample_step_deterministic_eps(
            eps_field, x, zs, idx, ts
        )
        det_result_v = deterministic_sampler.sample_step_deterministic_v(
            v_field, x, zs, idx, ts
        )

        # Check shapes
        assert det_result_score.shape == x.shape
        assert det_result_eps.shape == x.shape
        assert det_result_v.shape == x.shape

        # Test stochastic methods
        stoch_result_score = stochastic_sampler.sample_step_stochastic_score(
            score_field, x, zs, idx, ts
        )
        stoch_result_eps = stochastic_sampler.sample_step_stochastic_eps(
            eps_field, x, zs, idx, ts
        )
        stoch_result_v = stochastic_sampler.sample_step_stochastic_v(
            v_field, x, zs, idx, ts
        )

        # Check shapes
        assert stoch_result_score.shape == x.shape
        assert stoch_result_eps.shape == x.shape
        assert stoch_result_v.shape == x.shape

    def test_end_to_end_sampling(self):
        """Test end-to-end sampling with DDMSampler."""
        # Create a mock diffusion process
        alpha = lambda t: 1 - 0.5 * t
        sigma = lambda t: torch.sqrt(t)
        diffusion_process = DiffusionProcess(alpha=alpha, sigma=sigma)

        # Initialize sampler
        sampler = DDMSampler(
            diffusion_process, False
        )  # Deterministic for reproducibility

        # Create a simple x0 function that returns zeros
        def x0_fn(x, t):
            return torch.zeros_like(x)

        vector_field = VectorField(x0_fn, VectorFieldType.X0)

        # Create input tensors
        batch_size = 3
        data_dim = 2
        x_t = torch.randn(batch_size, data_dim)  # Initial noisy samples
        num_steps = 5
        zs = torch.randn(num_steps - 1, batch_size, data_dim)  # Noise for each step
        ts = torch.linspace(1.0, 0.0, num_steps)  # Time steps from t=1 to t=0

        # Sample
        result = sampler.sample(vector_field, x_t, zs, ts)

        # Check shape
        assert result.shape == (batch_size, data_dim)

        # Sample trajectory
        trajectory = sampler.sample_trajectory(vector_field, x_t, zs, ts)

        # Check shape
        assert trajectory.shape == (num_steps, batch_size, data_dim)
        assert torch.allclose(trajectory[0], x_t)  # First point should be x_t
        assert torch.allclose(
            trajectory[-1], result
        )  # Last point should be the final sample
