import jax
from jax import numpy as jnp, vmap
from diffusionlab.dynamics import VariancePreservingProcess
from diffusionlab.schedulers import UniformScheduler
from diffusionlab.samplers import DDMSampler
from diffusionlab.distributions.gmm.gmm import GMM
from diffusionlab.distributions.empirical import EmpiricalDistribution
from diffusionlab.vector_fields import VectorFieldType

jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_disable_jit", True)


def _sample_gmm():
    key = jax.random.key(1)

    dim = 2
    num_samples_ground_truth = 100
    num_samples_ddim = 50

    num_components = 3
    priors = jnp.ones(num_components) / num_components
    key, subkey = jax.random.split(key)
    means = jax.random.normal(subkey, (num_components, dim))
    key, subkey = jax.random.split(key)
    cov_factors = jax.random.normal(subkey, (num_components, dim, dim))
    covs = jax.vmap(lambda A: A @ A.T)(cov_factors)

    gmm = GMM(means, covs, priors)

    key, subkey = jax.random.split(key)
    X_ground_truth, y_ground_truth = gmm.sample(key, num_samples_ground_truth)

    num_steps = 100
    t_min = 0.001
    t_max = 0.999

    diffusion_process = VariancePreservingProcess()
    scheduler = UniformScheduler()
    ts = scheduler.get_ts(t_min=t_min, t_max=t_max, num_steps=num_steps)

    key, subkey = jax.random.split(key)
    X_noise = jax.random.normal(subkey, (num_samples_ddim, dim))

    zs = jax.random.normal(key, (num_samples_ddim, num_steps, dim))

    ground_truth_sampler = DDMSampler(
        diffusion_process,
        lambda x, t: gmm.x0(x, t, diffusion_process),
        VectorFieldType.X0,
        use_stochastic_sampler=False,
    )
    X_ddim_ground_truth = jax.vmap(
        lambda x_init, z: ground_truth_sampler.sample(x_init, z, ts)
    )(X_noise, zs)

    empirical_distribution = EmpiricalDistribution([(X_ground_truth, y_ground_truth)])
    empirical_sampler = DDMSampler(
        diffusion_process,
        lambda x, t: empirical_distribution.x0(x, t, diffusion_process),
        VectorFieldType.X0,
        use_stochastic_sampler=False,
    )
    X_ddim_empirical = jax.vmap(
        lambda x_init, z: empirical_sampler.sample(x_init, z, ts)
    )(X_noise, zs)

    return X_ground_truth, X_ddim_ground_truth, X_ddim_empirical


class TestGMMSampling:
    def test_gmm_sampling_e2e(self):
        X_ground_truth, X_ddim_ground_truth, X_ddim_empirical = _sample_gmm()
        assert not jnp.isnan(X_ground_truth).any()
        assert not jnp.isnan(X_ddim_ground_truth).any()
        assert not jnp.isnan(X_ddim_empirical).any()

        mean_min_distance = lambda X: jnp.mean(
            vmap(
                lambda x: jnp.min(
                    vmap(lambda x_gt: jnp.linalg.norm(x - x_gt))(X_ground_truth)
                )
            )(X)
        )

        mean_min_distance_gt = mean_min_distance(X_ddim_ground_truth)
        mean_min_distance_empirical = mean_min_distance(X_ddim_empirical)

        assert mean_min_distance_empirical < 0.1 * mean_min_distance_gt


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    X_ground_truth, X_ddim_ground_truth, X_ddim_empirical = _sample_gmm()

    mean_min_distance = lambda X: jnp.mean(
        vmap(
            lambda x: jnp.min(
                vmap(lambda x_gt: jnp.linalg.norm(x - x_gt))(X_ground_truth)
            )
        )(X)
    )

    mean_min_distance_gt = mean_min_distance(X_ddim_ground_truth)
    mean_min_distance_empirical = mean_min_distance(X_ddim_empirical)

    print(mean_min_distance_gt, mean_min_distance_empirical)

    fig, ax = plt.subplots()

    ax.scatter(
        X_ground_truth[:, 0], X_ground_truth[:, 1], alpha=0.7, label="Ground Truth"
    )
    ax.scatter(
        X_ddim_ground_truth[:, 0],
        X_ddim_ground_truth[:, 1],
        alpha=0.7,
        label="Ground Truth Sampler",
    )
    ax.scatter(
        X_ddim_empirical[:, 0],
        X_ddim_empirical[:, 1],
        alpha=0.7,
        label="Empirical Sampler",
    )

    ax.set_title("All Samples Overlaid")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.legend()

    plt.tight_layout()
    plt.show()
