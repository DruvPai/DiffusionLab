import torch
import pytest
from diffusionlab.schedulers import Scheduler, UniformScheduler


class TestScheduler:
    def test_abstract_class(self):
        """Test that the base Scheduler class cannot be instantiated directly."""
        # Instantiation should work, but get_ts should raise NotImplementedError
        scheduler = Scheduler(some_param=42)  # Test with a parameter
        with pytest.raises(NotImplementedError):
            scheduler.get_ts(t_min=0.0, t_max=1.0, L=100)


class TestUniformScheduler:
    def test_initialization(self):
        """Test basic initialization of UniformScheduler."""
        t_min = 0.0
        t_max = 1.0
        L = 100

        # Initialization should work without errors, with or without parameters
        scheduler1 = UniformScheduler()
        scheduler2 = UniformScheduler(some_param=42)

        # Check that get_ts works correctly for both instances
        ts1 = scheduler1.get_ts(t_min=t_min, t_max=t_max, L=L)
        ts2 = scheduler2.get_ts(t_min=t_min, t_max=t_max, L=L)

        assert ts1.shape == (L,)
        assert ts2.shape == (L,)
        assert torch.allclose(ts1[0], torch.tensor(t_max))
        assert torch.allclose(ts2[0], torch.tensor(t_max))
        assert torch.allclose(ts1[-1], torch.tensor(t_min))
        assert torch.allclose(ts2[-1], torch.tensor(t_min))
        assert torch.allclose(ts1, ts2)  # Both should produce the same result

    def test_get_ts(self):
        """Test the get_ts method of UniformScheduler."""
        t_min = 0.01
        t_max = 0.99
        L = 50

        # Create scheduler with optional parameters
        scheduler = UniformScheduler(optional_param="test")

        # Call get_ts with our test parameters
        ts = scheduler.get_ts(t_min=t_min, t_max=t_max, L=L)

        # Check shape
        assert ts.shape == (L,)

        # Check values
        assert torch.allclose(ts[0], torch.tensor(t_max))
        assert torch.allclose(ts[-1], torch.tensor(t_min))

        # Check that the steps are uniform by comparing with linspace
        expected_ts = torch.linspace(t_max, t_min, L)
        assert torch.allclose(ts, expected_ts)

    def test_invalid_parameters(self):
        """Test that invalid parameters raise appropriate errors."""
        scheduler = UniformScheduler(custom_param=123)

        # Test t_min < 0
        with pytest.raises(AssertionError):
            scheduler.get_ts(t_min=-0.1, t_max=1.0, L=100)

        # Test t_max > 1
        with pytest.raises(AssertionError):
            scheduler.get_ts(t_min=0.0, t_max=1.1, L=100)

        # Test t_min > t_max
        with pytest.raises(AssertionError):
            scheduler.get_ts(t_min=0.6, t_max=0.5, L=100)

        # Test L < 2
        with pytest.raises(AssertionError):
            scheduler.get_ts(t_min=0.0, t_max=1.0, L=1)

    def test_edge_cases(self):
        """Test edge cases for UniformScheduler."""
        scheduler = UniformScheduler(edge_case_param=True)

        # Test with t_min = t_max
        t_min = t_max = 0.5
        L = 10
        ts = scheduler.get_ts(t_min=t_min, t_max=t_max, L=L)

        assert ts.shape == (L,)
        assert torch.allclose(ts, torch.ones(L) * t_min)

        # Test with L = 2 (minimum allowed value)
        t_min = 0.0
        t_max = 1.0
        L = 2
        ts = scheduler.get_ts(t_min=t_min, t_max=t_max, L=L)

        assert ts.shape == (L,)
        assert torch.allclose(ts[0], torch.tensor(t_max))
        assert torch.allclose(ts[1], torch.tensor(t_min))

    def test_device_compatibility(self):
        """Test that the scheduler works with different devices."""
        if torch.cuda.is_available():
            device = torch.device("cuda")

            t_min = 0.0
            t_max = 1.0
            L = 100

            scheduler = UniformScheduler(device_param="cuda")
            ts = scheduler.get_ts(t_min=t_min, t_max=t_max, L=L)

            # Move ts to GPU
            ts_gpu = ts.to(device)

            # Check that it's on the right device
            assert ts_gpu.device == device

            # Check that values are the same
            assert torch.allclose(ts_gpu.cpu(), ts)
