"""Train a class-conditional DiT on MNIST with classifier-free guidance.

Demonstrates:
  1. Patchifying MNIST into sequences for a DiT.
  2. Class-conditional training with 10% label dropout for CFG.
  3. Sampling with classifier-free guidance.
  4. Plotting loss curve + generated digits per class.

Uses only the first 100 samples per class (1000 total) to keep it fast.

Usage:
    uv run python examples/mnist_dit.py
"""

import struct
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from diffusionlab.models.dit import ConditionedDiT
from diffusionlab.processes.gaussian import FlowMatching
from diffusionlab.sampling.base import sample
from diffusionlab.sampling.continuous import DDMSampler
from diffusionlab.sampling.guidance import CFG
from diffusionlab.sampling.schedules import uniform_schedule
from diffusionlab.training.denoising import DenoisingLoss
from diffusionlab.training.schedules import logit_normal_time

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEED = 0
SAMPLES_PER_CLASS = 100
BATCH_SIZE = 64
NUM_STEPS = 5000
LR = 3e-4

PATCH_SIZE = 4  # 28/4 = 7 -> 49 patches of dim 16
SEQ_LEN = 49
PATCH_DIM = PATCH_SIZE * PATCH_SIZE  # 16

HIDDEN_DIM = 128
NUM_LAYERS = 4
NUM_HEADS = 4
HEAD_DIM = 32
MLP_DIM = 256
TIME_EMB_DIM = 64
COND_DIM = 128
COND_EMB_DIM = 10  # one-hot class label
COND_DROP_PROB = 0.1  # label dropout rate for CFG

SAMPLE_STEPS = 100
GUIDANCE_SCALE = 3.0
SAMPLES_PER_DIGIT = 8


# ---------------------------------------------------------------------------
# Data loading helpers (raw IDX files from MNIST)
# ---------------------------------------------------------------------------
def load_mnist_images(path: Path) -> jax.Array:
    """Load MNIST images from IDX file, return (N, 28, 28) float32 in [-1, 1]."""
    with open(path, "rb") as f:
        _magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        raw = jnp.frombuffer(f.read(), dtype=jnp.uint8).reshape(num, rows, cols)
    return raw.astype(jnp.float32) / 127.5 - 1.0


def load_mnist_labels(path: Path) -> jax.Array:
    """Load MNIST labels from IDX file, return (N,) int32."""
    with open(path, "rb") as f:
        _magic, _num = struct.unpack(">II", f.read(8))
        raw = jnp.frombuffer(f.read(), dtype=jnp.uint8)
    return raw.astype(jnp.int32)


def download_mnist(data_dir: Path) -> tuple[jax.Array, jax.Array]:
    """Download MNIST if not cached, return (images, labels)."""
    import gzip
    import urllib.request

    data_dir.mkdir(parents=True, exist_ok=True)
    base = "https://storage.googleapis.com/cvdf-datasets/mnist"
    files = {
        "train-images-idx3-ubyte.gz": "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte.gz": "train-labels-idx1-ubyte",
    }
    for gz_name, raw_name in files.items():
        raw_path = data_dir / raw_name
        if not raw_path.exists():
            gz_path = data_dir / gz_name
            if not gz_path.exists():
                print(f"  Downloading {gz_name}...")
                urllib.request.urlretrieve(f"{base}/{gz_name}", gz_path)
            with gzip.open(gz_path, "rb") as gz:
                raw_path.write_bytes(gz.read())

    images = load_mnist_images(data_dir / "train-images-idx3-ubyte")
    labels = load_mnist_labels(data_dir / "train-labels-idx1-ubyte")
    return images, labels


# ---------------------------------------------------------------------------
# Patchify / unpatchify
# ---------------------------------------------------------------------------
def patchify(images: jax.Array) -> jax.Array:
    """(N, 28, 28) -> (N, 49, 16): reshape into 4x4 patches.

    We pad 28 -> 28 (already divisible by 4), giving 7x7=49 patches.
    """
    n = images.shape[0]
    # Reshape into (N, 7, 4, 7, 4)
    x = images.reshape(n, 7, PATCH_SIZE, 7, PATCH_SIZE)
    # Transpose to (N, 7, 7, 4, 4)
    x = x.transpose(0, 1, 3, 2, 4)
    # Flatten patches: (N, 49, 16)
    return x.reshape(n, SEQ_LEN, PATCH_DIM)


def unpatchify(patches: jax.Array) -> jax.Array:
    """(N, 49, 16) -> (N, 28, 28): reverse of patchify."""
    n = patches.shape[0]
    x = patches.reshape(n, 7, 7, PATCH_SIZE, PATCH_SIZE)
    x = x.transpose(0, 1, 3, 2, 4)
    return x.reshape(n, 28, 28)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("Loading MNIST...")
data_dir = Path("examples/.data/mnist")
all_images, all_labels = download_mnist(data_dir)

# Take first SAMPLES_PER_CLASS per digit
indices = []
for digit in range(10):
    mask = jnp.where(all_labels == digit)[0]
    indices.append(mask[:SAMPLES_PER_CLASS])
indices = jnp.concatenate(indices)

images = all_images[indices]  # (1000, 28, 28)
labels = all_labels[indices]  # (1000,)
patches = patchify(images)  # (1000, 49, 16)
num_data = patches.shape[0]
print(f"  Using {num_data} samples ({SAMPLES_PER_CLASS} per class)")

# ---------------------------------------------------------------------------
# 2. Create process and model
# ---------------------------------------------------------------------------
process = FlowMatching(data_shape=(SEQ_LEN, PATCH_DIM))

key = jax.random.key(SEED)
key, key_model = jax.random.split(key)

model = ConditionedDiT(
    input_dim=PATCH_DIM,
    output_dim=PATCH_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    head_dim=HEAD_DIM,
    mlp_dim=MLP_DIM,
    max_seq_len=SEQ_LEN,
    time_emb_dim=TIME_EMB_DIM,
    cond_emb_dim=COND_EMB_DIM,
    cond_dim=COND_DIM,
    prediction_kind="v_t",
    activation=jax.nn.gelu,
    max_period=10_000.0,
    key=key_model,
)

num_params = sum(p.size for p in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
print(f"  Model parameters: {num_params:,}")

# ---------------------------------------------------------------------------
# 3. Training setup
# ---------------------------------------------------------------------------
loss_obj = DenoisingLoss(
    process=process, weight_fn=None, loss_fn=None, target_kind=None
)
optimizer = optax.adamw(learning_rate=LR, weight_decay=1e-4)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))


@eqx.filter_jit
def train_step(model, opt_state, x_0, cond, key):
    key_t, key_fwd, key_drop = jax.random.split(key, 3)
    batch = x_0.shape[0]
    t = logit_normal_time(key_t, batch, mean=0.0, std=1.0)
    x_t, aux = process.forward(key_fwd, x_0, t)

    # Conditioning dropout: replace label with zeros with probability COND_DROP_PROB
    drop_mask = jax.random.bernoulli(key_drop, COND_DROP_PROB, (batch, 1))
    cond_dropped = cond * (1.0 - drop_mask)

    def compute_loss(m):
        out = loss_obj(predictor=m, x_0=x_0, x_t=x_t, t=t, aux=aux, cond=cond_dropped)
        return jnp.mean(out.loss)

    loss, grads = eqx.filter_value_and_grad(compute_loss)(model)
    updates, new_opt_state = optimizer.update(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss


# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
# Pre-compute one-hot labels
labels_onehot = jax.nn.one_hot(labels, 10)  # (1000, 10)

losses = []
print("Training...")
for step in range(NUM_STEPS):
    key, key_batch, key_step = jax.random.split(key, 3)
    idx = jax.random.randint(key_batch, (BATCH_SIZE,), 0, num_data)
    x_0 = patches[idx]
    cond = labels_onehot[idx]
    model, opt_state, loss = train_step(model, opt_state, x_0, cond, key_step)
    losses.append(float(loss))
    if (step + 1) % 500 == 0:
        print(f"  step {step + 1:5d}  loss = {loss:.4f}")

# ---------------------------------------------------------------------------
# 5. Conditional sampling with CFG
# ---------------------------------------------------------------------------
print("Sampling with classifier-free guidance...")
sampler = DDMSampler(process=process, eta=0.0)
cfg = CFG(guidance_scale=GUIDANCE_SCALE)
schedule = uniform_schedule(SAMPLE_STEPS, SAMPLES_PER_DIGIT)

all_samples = []
for digit in range(10):
    key, key_sample = jax.random.split(key)
    cond = jnp.broadcast_to(
        jax.nn.one_hot(digit, 10), (SAMPLES_PER_DIGIT, COND_EMB_DIM)
    )
    gen = sample(
        key=key_sample,
        sampler=sampler,
        predictor=model,
        batch_size=SAMPLES_PER_DIGIT,
        time_schedule=schedule,
        cond=cond,
        transforms=[cfg],
    )
    all_samples.append(gen)

all_samples = jnp.stack(all_samples)  # (10, SAMPLES_PER_DIGIT, 49, 16)

# ---------------------------------------------------------------------------
# 6. Plotting
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(10, SAMPLES_PER_DIGIT + 1, figsize=(SAMPLES_PER_DIGIT + 1, 11))

# First column: real data sample for each class
for digit in range(10):
    ax = axes[digit, 0]
    mask = jnp.where(labels == digit)[0]
    real_img = images[mask[0]]
    ax.imshow(real_img, cmap="gray", vmin=-1, vmax=1)
    ax.set_ylabel(str(digit), fontsize=12, rotation=0, labelpad=15)
    ax.set_xticks([])
    ax.set_yticks([])
    if digit == 0:
        ax.set_title("Real", fontsize=9)

# Remaining columns: generated samples
for digit in range(10):
    for j in range(SAMPLES_PER_DIGIT):
        ax = axes[digit, j + 1]
        gen_img = unpatchify(all_samples[digit : digit + 1, j])[0]
        gen_img = jnp.clip(gen_img, -1, 1)
        ax.imshow(gen_img, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if digit == 0:
            ax.set_title(f"Gen {j + 1}", fontsize=9)

plt.suptitle(
    f"Class-Conditional MNIST DiT (CFG w={GUIDANCE_SCALE})",
    fontsize=13,
    y=0.98,
)
plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.savefig("examples/mnist_dit_samples.png", dpi=150, bbox_inches="tight")
print("Saved examples/mnist_dit_samples.png")

# --- Loss curve ---
fig2, ax2 = plt.subplots(figsize=(6, 3.5))
ax2.plot(losses, linewidth=0.5, alpha=0.3, color="C0")
window = 100
if len(losses) >= window:
    smoothed = jnp.convolve(jnp.array(losses), jnp.ones(window) / window, mode="valid")
    ax2.plot(range(window - 1, len(losses)), smoothed, linewidth=1.5, color="C0")
ax2.set_xlabel("Step")
ax2.set_ylabel("Loss")
ax2.set_title("MNIST DiT Training Loss")
ax2.set_yscale("log")
plt.tight_layout()
plt.savefig("examples/mnist_dit_loss.png", dpi=150, bbox_inches="tight")
print("Saved examples/mnist_dit_loss.png")
plt.show()
