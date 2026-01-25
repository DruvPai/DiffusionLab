"""Tests for diffusionlab.training.schedules (stochastic time schedules)."""

import jax
import jax.numpy as jnp

from diffusionlab.training.schedules import (
    logit_normal_time,
    logsnr_uniform_time,
    stratified_time,
    uniform_time,
    uniform_time_with_eps,
)

BATCH_SIZE = 16


# ===========================================================================
# Stochastic (training) schedules
# ===========================================================================


class TestUniformTime:
    def test_shape(self):
        t = uniform_time(jax.random.key(0), BATCH_SIZE)
        assert t.shape == (BATCH_SIZE,)

    def test_range(self):
        t = uniform_time(jax.random.key(0), BATCH_SIZE)
        assert jnp.all(t >= 0.0)
        assert jnp.all(t <= 1.0)


class TestUniformTimeWithEps:
    def test_shape(self):
        t = uniform_time_with_eps(jax.random.key(0), BATCH_SIZE)
        assert t.shape == (BATCH_SIZE,)

    def test_range(self):
        eps = 1e-3
        t = uniform_time_with_eps(jax.random.key(0), BATCH_SIZE, eps=eps)
        assert jnp.all(t >= eps)
        assert jnp.all(t <= 1.0 - eps)


class TestLogitNormalTime:
    def test_shape(self):
        t = logit_normal_time(jax.random.key(0), BATCH_SIZE)
        assert t.shape == (BATCH_SIZE,)

    def test_range(self):
        t = logit_normal_time(jax.random.key(0), BATCH_SIZE)
        assert jnp.all(t > 0.0)
        assert jnp.all(t < 1.0)

    def test_mean_shift(self):
        key = jax.random.key(42)
        # Large positive mean should shift distribution toward 1
        t_high = logit_normal_time(key, 1000, mean=3.0)
        t_low = logit_normal_time(key, 1000, mean=-3.0)
        assert jnp.mean(t_high) > jnp.mean(t_low)


class TestLogSNRUniformTime:
    def test_shape(self):
        inv = lambda lam: jax.nn.sigmoid(-lam)  # noqa: E731
        t = logsnr_uniform_time(jax.random.key(0), BATCH_SIZE, logsnr_inverse=inv)
        assert t.shape == (BATCH_SIZE,)

    def test_range(self):
        inv = lambda lam: jnp.clip(jax.nn.sigmoid(-lam), 0, 1)  # noqa: E731
        t = logsnr_uniform_time(jax.random.key(0), BATCH_SIZE, logsnr_inverse=inv)
        assert jnp.all(t >= 0.0)
        assert jnp.all(t <= 1.0)


class TestStratifiedTime:
    def test_shape(self):
        t = stratified_time(jax.random.key(0), BATCH_SIZE)
        assert t.shape == (BATCH_SIZE,)

    def test_range(self):
        t = stratified_time(jax.random.key(0), BATCH_SIZE)
        assert jnp.all(t >= 0.0)
        assert jnp.all(t <= 1.0)

    def test_sorted_approximately(self):
        t = stratified_time(jax.random.key(0), BATCH_SIZE)
        # Stratified should be approximately sorted
        sorted_t = jnp.sort(t)
        assert jnp.allclose(t, sorted_t)

    def test_spread(self):
        t = stratified_time(jax.random.key(0), BATCH_SIZE)
        # Each sample should be in its own stratum
        for i in range(BATCH_SIZE):
            assert t[i] >= i / BATCH_SIZE
            assert t[i] <= (i + 1) / BATCH_SIZE


# ===========================================================================
# Re-export from diffusionlab.training
# ===========================================================================


class TestTrainingReexport:
    def test_training_init_import(self):
        """diffusionlab.training re-exports training schedules."""
        from diffusionlab.training import logit_normal_time as lnt2
        from diffusionlab.training import uniform_time as ut2

        assert ut2 is uniform_time
        assert lnt2 is logit_normal_time
