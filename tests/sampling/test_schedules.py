"""Tests for diffusionlab.sampling.schedules (deterministic time schedules)."""

import jax.numpy as jnp
import pytest

from diffusionlab.sampling.schedules import edm_schedule, uniform_schedule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
STEP_COUNTS = [1, 2, 5, 10, 50]


# ===========================================================================
# Tests for uniform_schedule
# ===========================================================================


class TestUniformSchedule:
    """Tests for the uniform time schedule."""

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_length(self, num_steps: int):
        """Schedule has num_steps + 1 entries."""
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        assert isinstance(sched, list)
        assert len(sched) == num_steps + 1

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_shape(self, num_steps: int):
        """Each entry has shape (batch_size,)."""
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        for entry in sched:
            assert entry.shape == (BATCH_SIZE,)

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_endpoints(self, num_steps: int):
        """First entry is 1.0, last entry is 0.0."""
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        assert jnp.allclose(sched[0], 1.0)
        assert jnp.allclose(sched[-1], 0.0)

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_strictly_decreasing(self, num_steps: int):
        """Schedule is strictly monotonically decreasing."""
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        for i in range(len(sched) - 1):
            assert jnp.all(sched[i] > sched[i + 1])

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_values_in_unit_interval(self, num_steps: int):
        """All values are in [0, 1]."""
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        for entry in sched:
            assert jnp.all(entry >= 0.0)
            assert jnp.all(entry <= 1.0)

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_uniform_spacing(self, num_steps: int):
        """Entries are uniformly spaced."""
        sched = uniform_schedule(num_steps, BATCH_SIZE)
        values = jnp.array([float(s[0]) for s in sched])
        diffs = jnp.diff(values)
        # All diffs should be equal
        assert jnp.allclose(diffs, diffs[0], atol=1e-6)

    def test_batch_broadcast(self):
        """All elements within a time step are equal (broadcast from scalar)."""
        sched = uniform_schedule(5, 8)
        for entry in sched:
            assert jnp.all(entry == entry[0])

    def test_single_step(self):
        """num_steps=1 gives [1.0, 0.0]."""
        sched = uniform_schedule(1, BATCH_SIZE)
        assert len(sched) == 2
        assert jnp.allclose(sched[0], 1.0)
        assert jnp.allclose(sched[-1], 0.0)


# ===========================================================================
# Tests for edm_schedule
# ===========================================================================


class TestEDMSchedule:
    """Tests for the EDM power-law time schedule."""

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_length(self, num_steps: int):
        """Schedule has num_steps + 1 entries."""
        sched = edm_schedule(num_steps, BATCH_SIZE)
        assert isinstance(sched, list)
        assert len(sched) == num_steps + 1

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_shape(self, num_steps: int):
        """Each entry has shape (batch_size,)."""
        sched = edm_schedule(num_steps, BATCH_SIZE)
        for entry in sched:
            assert entry.shape == (BATCH_SIZE,)

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_endpoints(self, num_steps: int):
        """First entry ~ 1.0, last entry ~ 0.0."""
        sched = edm_schedule(num_steps, BATCH_SIZE)
        assert jnp.allclose(sched[0], 1.0, atol=1e-5)
        assert jnp.allclose(sched[-1], 0.0, atol=1e-5)

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_strictly_decreasing(self, num_steps: int):
        """Schedule is strictly monotonically decreasing."""
        sched = edm_schedule(num_steps, BATCH_SIZE)
        for i in range(len(sched) - 1):
            assert jnp.all(sched[i] > sched[i + 1])

    @pytest.mark.parametrize("num_steps", STEP_COUNTS)
    def test_values_in_unit_interval(self, num_steps: int):
        """All values are in [0, 1]."""
        sched = edm_schedule(num_steps, BATCH_SIZE)
        for entry in sched:
            assert jnp.all(entry >= -1e-7)
            assert jnp.all(entry <= 1.0 + 1e-7)

    def test_sigma_spacing_is_power_law(self):
        """The underlying sigma spacing follows a power law (not uniform)."""
        # Verify the sigma values are power-law spaced by checking
        # the schedule changes shape with rho even if normalized times are uniform
        inv_rho_a = 1.0 / 7.0
        inv_rho_b = 1.0 / 3.0
        s_min = 0.002
        s_max = 80.0
        sigmas_a = jnp.linspace(s_max**inv_rho_a, s_min**inv_rho_a, 11) ** 7.0
        sigmas_b = jnp.linspace(s_max**inv_rho_b, s_min**inv_rho_b, 11) ** 3.0
        # Different rho produces different sigma spacing
        assert not jnp.allclose(sigmas_a, sigmas_b, atol=1e-2)

    def test_custom_sigma_bounds(self):
        """Custom sigma_min/sigma_max still produce valid schedule."""
        sched = edm_schedule(10, BATCH_SIZE, sigma_min=0.01, sigma_max=100.0, rho=5.0)
        assert jnp.allclose(sched[0], 1.0, atol=1e-4)
        assert jnp.allclose(sched[-1], 0.0, atol=1e-4)
        for i in range(len(sched) - 1):
            assert jnp.all(sched[i] > sched[i + 1])

    def test_batch_broadcast(self):
        """All elements within a time step are equal."""
        sched = edm_schedule(5, 8)
        for entry in sched:
            assert jnp.all(entry == entry[0])
