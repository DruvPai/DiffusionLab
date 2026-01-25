"""Train a flow-matching diffusion model on synthetic 2-D two-moons data.

Demonstrates:
  1. Generating synthetic data (sklearn two-moons).
  2. Setting up a FlowMatching process + DiffusionMLP.
  3. Training with DenoisingLoss and optax.
  4. Plotting the loss curve.
  5. Sampling from the trained model and visualising sample trajectories.

Usage:
    uv run python examples/two_moons.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from sklearn.datasets import make_moons

from diffusionlab.models.mlp import DiffusionMLP
from diffusionlab.processes.gaussian import FlowMatching
from diffusionlab.sampling.base import sample_trajectory
from diffusionlab.sampling.continuous import EulerSampler
from diffusionlab.sampling.schedules import uniform_schedule
from diffusionlab.training.denoising import DenoisingLoss
from diffusionlab.training.schedules import uniform_time_with_eps

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 42
NUM_DATA = 2048
BATCH_SIZE = 256
NUM_STEPS = 3000
LR = 3e-4

HIDDEN_DIM = 128
NUM_LAYERS = 4
TIME_EMB_DIM = 64
COND_DIM = 64

SAMPLE_STEPS = 100
NUM_SAMPLES = 512

# ---------------------------------------------------------------------------
# 1. Generate two-moons data
# ---------------------------------------------------------------------------
X_np, _ = make_moons(n_samples=NUM_DATA, noise=0.06, random_state=SEED)
data = jnp.array(X_np, dtype=jnp.float32)

# ---------------------------------------------------------------------------
# 2. Create process and model
# ---------------------------------------------------------------------------
process = FlowMatching(data_shape=(2,))

key = jax.random.key(SEED)
key, key_model = jax.random.split(key)

model = DiffusionMLP(
    data_dim=2,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    time_emb_dim=TIME_EMB_DIM,
    cond_dim=COND_DIM,
    prediction_kind="v_t",
    activation=jax.nn.silu,
    max_period=10_000.0,
    key=key_model,
)

# ---------------------------------------------------------------------------
# 3. Training setup
# ---------------------------------------------------------------------------
loss_obj = DenoisingLoss(
    process=process, weight_fn=None, loss_fn=None, target_kind=None
)
optimizer = optax.adam(learning_rate=LR)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, x_0, key):
    key_t, key_fwd = jax.random.split(key)
    batch = x_0.shape[0]
    t = uniform_time_with_eps(key_t, batch)
    x_t, aux = process.forward(key_fwd, x_0, t)

    def compute_loss(m):
        out = loss_obj(predictor=m, x_0=x_0, x_t=x_t, t=t, aux=aux, cond=None)
        return jnp.mean(out.loss)

    loss, grads = eqx.filter_value_and_grad(compute_loss)(model)
    updates, opt_state_new = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    model_new = eqx.apply_updates(model, updates)
    return model_new, opt_state_new, loss


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
losses = []
print("Training...")
for step in range(NUM_STEPS):
    key, key_batch, key_step = jax.random.split(key, 3)
    idx = jax.random.randint(key_batch, (BATCH_SIZE,), 0, NUM_DATA)
    x_0 = data[idx]
    model, opt_state, loss = train_step(model, opt_state, x_0, key_step)
    losses.append(float(loss))
    if (step + 1) % 500 == 0:
        print(f"  step {step + 1:5d}  loss = {loss:.4f}")

# ---------------------------------------------------------------------------
# 5. Sampling + trajectory
# ---------------------------------------------------------------------------
print("Sampling...")
sampler = EulerSampler(process=process)
schedule = uniform_schedule(SAMPLE_STEPS, NUM_SAMPLES)

key, key_sample = jax.random.split(key)
samples, trajectory = sample_trajectory(
    key=key_sample,
    sampler=sampler,
    predictor=model,
    batch_size=NUM_SAMPLES,
    time_schedule=schedule,
)

# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Loss curve ---
ax = axes[0]
ax.plot(losses, linewidth=0.5, alpha=0.3, color="C0")
# Smoothed
window = 50
if len(losses) >= window:
    smoothed = jnp.convolve(jnp.array(losses), jnp.ones(window) / window, mode="valid")
    ax.plot(range(window - 1, len(losses)), smoothed, linewidth=1.5, color="C0")
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss")
ax.set_yscale("log")

# --- Samples vs data ---
ax = axes[1]
ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.3, label="Data", color="C1")
ax.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, label="Samples", color="C0")
ax.legend(markerscale=5)
ax.set_title("Data vs Samples")
ax.set_aspect("equal")

# --- Sample trajectories ---
ax = axes[2]
# Stack trajectory positions: (num_steps, batch, 2)
traj_xs = jnp.stack([s.x for s in trajectory])
# Plot a handful of trajectories
num_traj = 8
for i in range(num_traj):
    ax.plot(traj_xs[:, i, 0], traj_xs[:, i, 1], linewidth=0.8, alpha=0.7)
    ax.scatter(traj_xs[-1, i, 0], traj_xs[-1, i, 1], s=20, zorder=5)
ax.scatter(data[:, 0], data[:, 1], s=0.5, alpha=0.15, color="gray")
ax.set_title("Sample Trajectories (noise → data)")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("examples/two_moons.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved examples/two_moons.png")
