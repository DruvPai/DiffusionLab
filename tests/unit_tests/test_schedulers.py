import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

from diffusionlab.schedulers import Scheduler, UniformScheduler


class TestScheduler:
    """Tests for the base Scheduler class."""

    def test_get_ts_not_implemented(self) -> None:
        """Test that the base class raises NotImplementedError."""
        scheduler = Scheduler()
        with pytest.raises(NotImplementedError):
            scheduler.get_ts()


class TestUniformScheduler:
    """Tests for the UniformScheduler class."""

    def test_get_ts_basic(self) -> None:
        """Test standard usage of get_ts."""
        scheduler = UniformScheduler()
        t_min = 0.01
        t_max = 1.0
        num_steps = 10
        expected_ts = jnp.linspace(t_min, t_max, num_steps + 1)[::-1]
        actual_ts = scheduler.get_ts(t_min=t_min, t_max=t_max, num_steps=num_steps)

        assert actual_ts.shape == (num_steps + 1,)
        assert_allclose(actual_ts, expected_ts, atol=1e-6)
        assert actual_ts[0] == t_max  # Check descending order
        assert actual_ts[-1] == t_min

    def test_get_ts_num_steps_one(self) -> None:
        """Test get_ts with num_steps=1."""
        scheduler = UniformScheduler()
        t_min = 0.0
        t_max = 0.5
        num_steps = 1
        expected_ts = jnp.array([t_max, t_min])
        actual_ts = scheduler.get_ts(t_min=t_min, t_max=t_max, num_steps=num_steps)

        assert actual_ts.shape == (2,)
        assert_allclose(actual_ts, expected_ts, atol=1e-6)

    def test_get_ts_t_min_equals_t_max(self) -> None:
        """Test get_ts when t_min equals t_max."""
        scheduler = UniformScheduler()
        t_val = 0.7
        num_steps = 5
        # linspace(0.7, 0.7, 6) -> [0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        expected_ts = jnp.full((num_steps + 1,), t_val)
        actual_ts = scheduler.get_ts(t_min=t_val, t_max=t_val, num_steps=num_steps)

        assert actual_ts.shape == (num_steps + 1,)
        assert_allclose(actual_ts, expected_ts, atol=1e-6)

    def test_get_ts_missing_params(self) -> None:
        """Test that KeyError is raised for missing parameters."""
        scheduler = UniformScheduler()
        with pytest.raises(KeyError, match="t_min"):
            scheduler.get_ts(t_max=1.0, num_steps=10)
        with pytest.raises(KeyError, match="t_max"):
            scheduler.get_ts(t_min=0.0, num_steps=10)
        with pytest.raises(KeyError, match="num_steps"):
            scheduler.get_ts(t_min=0.0, t_max=1.0)

    @pytest.mark.parametrize(
        "t_min, t_max, num_steps",
        [
            (0.1, 0.0, 10),  # t_min > t_max
            (-0.1, 1.0, 10),  # t_min < 0
            (0.0, 1.1, 10),  # t_max > 1
        ],
    )
    def test_get_ts_invalid_t_range(
        self, t_min: float, t_max: float, num_steps: int
    ) -> None:
        """Test assertion failure for invalid t_min/t_max."""
        scheduler = UniformScheduler()
        with pytest.raises(
            AssertionError, match="t_min and t_max must be in the range"
        ):
            scheduler.get_ts(t_min=t_min, t_max=t_max, num_steps=num_steps)

    @pytest.mark.parametrize("num_steps", [0, -1])
    def test_get_ts_invalid_num_steps(self, num_steps: int) -> None:
        """Test assertion failure for num_steps < 1."""
        scheduler = UniformScheduler()
        with pytest.raises(AssertionError, match="num_steps must be at least 1"):
            scheduler.get_ts(t_min=0.0, t_max=1.0, num_steps=num_steps)
